use std::fs::File;
use std::path::Path;

use cudarc::cublas::sys as cublas_sys;
use gpu_host::{GpuCtxGuard, GpuCtxSpace, PinnedHostBox, TensorSliceMut};
use memmap2::Mmap;

pub(crate) mod dataloader;

#[macro_use]
pub(crate) mod kernels;
pub(crate) mod params;

use dataloader::parse_header_data;
use kernels::*;
use params::{ActivationTensors, GPT2Config, ParameterTensors};

pub struct GPT2<'ctx, NS: GpuCtxSpace> {
    pub module: &'ctx gpu_host::GpuModule<NS>,
    /// Model configuration.
    pub config: GPT2Config,

    /// Total number of parameters.
    pub num_parameters: usize,

    /// The weights (parameters) of the model.
    pub params: ParameterTensors<'ctx, NS>,

    /// Gradients of the weights.
    pub grads: Option<ParameterTensors<'ctx, NS>>,

    /// Buffer for the AdamW optimizer.
    pub m_memory: Option<TensorSliceMut<'ctx, f32, NS>>,

    /// Buffer for the AdamW optimizer.
    pub v_memory: Option<TensorSliceMut<'ctx, f32, NS>>,

    /// The activations of the model.
    pub acts: Option<ActivationTensors<'ctx, NS>>,

    /// Total number of activations.
    pub num_activations: usize,

    /// Gradients of the activations.
    pub grads_acts: Option<ActivationTensors<'ctx, NS>>,

    /// The batch size (B) of the current forward pass
    pub batch_size: usize,

    /// The sequence length (T) of the current forward pass
    pub seq_len: usize,

    /// The input tokens for the current forward pass
    pub inputs: Option<TensorSliceMut<'ctx, i32, NS>>,

    /// The target tokens for the current forward pass
    pub targets: Option<TensorSliceMut<'ctx, i32, NS>>,

    /// After a forward pass with targets, will be populated with the mean loss
    pub mean_loss: f32,

    pub cpu_losses: Option<PinnedHostBox<'ctx, [f32]>>,
}

impl<'ctx, NS: GpuCtxSpace> GPT2<'ctx, NS> {
    /// Creates a new GPT-2 model instance from a checkpoint file.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to the checkpoint file containing model parameters and configuration.
    ///
    /// # Returns
    ///
    /// A new `GPT2` model instance.
    pub fn new<'ctx_a: 'ctx>(
        ctx: &'ctx GpuCtxGuard<'ctx_a, '_, NS>,
        m: &'ctx gpu_host::GpuModule<NS>,
        checkpoint_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Read model from a checkpoint file
        let model_file = File::open(checkpoint_path).unwrap_or_else(|_| {
            panic!("Error opening model file: {:?}", checkpoint_path);
        });
        let model_data = unsafe { Mmap::map(&model_file)? };
        let (model_header, cpu_params) = parse_header_data(&model_data, 20240326, 3);
        if model_header[1] != 3 {
            panic!("Bad version in model file");
        }

        // Read in hyperparameters
        let max_seq_len = model_header[2] as usize;
        let vocab_size = model_header[3] as usize;
        let num_layers = model_header[4] as usize;
        let num_heads = model_header[5] as usize;
        let channels = model_header[6] as usize;
        let padded_vocab_size = model_header[7] as usize;

        let config = GPT2Config {
            max_seq_len,
            vocab_size,
            padded_vocab_size,
            num_layers,
            num_heads,
            channels,
        };
        println!("[GPT-2]");
        println!("max_seq_len: {}", max_seq_len);
        println!("vocab_size: {}", vocab_size);
        println!("padded_vocab_size: {}", padded_vocab_size);
        println!("num_layers: {}", num_layers);
        println!("num_heads: {}", num_heads);
        println!("channels: {}", channels);

        let param_sizes = config.get_params_sizes();
        // Count the number of parameters
        let num_parameters: usize = param_sizes.iter().sum();
        println!("num_parameters: {}", num_parameters);
        let cpu_params = &cpu_params[0..num_parameters];
        let params: ParameterTensors<'ctx, NS> =
            ParameterTensors::new(ctx, param_sizes, cpu_params);

        Ok(GPT2 {
            config,
            module: m,
            num_parameters,
            params,
            grads: None,
            m_memory: None,
            v_memory: None,
            acts: None,
            num_activations: 0,
            grads_acts: None,
            batch_size: 0,
            seq_len: 0,
            inputs: None,
            targets: None,
            mean_loss: -1.0,
            cpu_losses: None,
        })
    }

    pub fn forward<'ctx_a: 'ctx>(
        &mut self,
        ctx: &'ctx GpuCtxGuard<'ctx_a, '_, NS>,
        cublas_handle: cublas_sys::cublasHandle_t,
        cpu_inputs: &[i32],
        cpu_targets: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) {
        assert!(cpu_inputs.len() >= batch_size * seq_len);
        assert!(cpu_targets.len() >= batch_size * seq_len);
        cpu_inputs
            .iter()
            .for_each(|&x| assert!(0 <= x && (x as usize) < self.config.vocab_size, "{}", x));
        cpu_targets
            .iter()
            .for_each(|&x| assert!(0 <= x && (x as usize) < self.config.vocab_size, "{}", x));

        // allocate space for all the activations if needed (done here, lazily)
        if self.acts.is_none() {
            let act_sizes = self.config.get_act_sizes(batch_size, seq_len);
            self.num_activations = act_sizes.iter().sum();
            self.batch_size = batch_size;
            self.seq_len = seq_len;
            self.acts =
                Some(ActivationTensors::new(ctx, act_sizes, &vec![0.0f32; self.num_activations]));
            println!(
                "allocated {} MiB for activations",
                (self.num_activations * std::mem::size_of::<f32>()) / (1024 * 1024)
            );
        }

        // also create memory for caching inputs and targets
        if self.inputs.is_none() {
            self.inputs = Some(ctx.new_tensor_slice(&cpu_inputs[0..batch_size * seq_len]).unwrap());
        } else {
            self.inputs
                .as_mut()
                .unwrap()
                .copy_from_host(&cpu_inputs[0..batch_size * seq_len], batch_size * seq_len, ctx)
                .unwrap();
        }
        if cpu_targets.len() > 0 {
            if self.targets.is_none() {
                self.targets =
                    Some(ctx.new_tensor_slice(&cpu_targets[0..batch_size * seq_len]).unwrap());
            } else {
                self.targets
                    .as_mut()
                    .unwrap()
                    .copy_from_host(
                        &cpu_targets[0..batch_size * seq_len],
                        batch_size * seq_len,
                        ctx,
                    )
                    .unwrap();
            }
        }

        // validate B,T is consistent with how we've allocated the memory before
        assert!(self.batch_size == batch_size);
        assert!(self.seq_len == seq_len);

        // Sync losses from GPU to CPU
        let acts = self.acts.as_mut().unwrap().inner(ctx);
        let params = self.params.inner(ctx);
        let inputs = self.inputs.as_ref().unwrap();
        let targets = self.targets.as_ref().unwrap();
        //acts.encoded, model->inputs, params.wte, params.wpe
        encoder_forward(
            ctx,
            self.module,
            acts.encoded,
            inputs,
            params.wte,
            params.wpe,
            batch_size,
            seq_len,
            self.config.channels,
        );

        //for (int l = 0; l < L; l++) {
        let mut residual3 = acts.residual3;
        let mut residual = acts.encoded;
        let mut ln1w = params.ln1w;
        let mut ln1b = params.ln1b;
        let mut qkvw = params.qkvw;
        let mut qkvb = params.qkvb;
        let mut attprojw = params.attprojw;
        let mut attprojb = params.attprojb;
        let mut ln2w = params.ln2w;
        let mut ln2b = params.ln2b;
        let mut fcw = params.fcw;
        let mut fcb = params.fcb;
        let mut fcprojw = params.fcprojw;
        let mut fcprojb = params.fcprojb;

        let mut ln1 = acts.ln1;
        let mut ln1_mean = acts.ln1_mean;
        let mut ln1_rstd = acts.ln1_rstd;
        let mut qkvr = acts.qkvr;
        let mut atty = acts.atty;
        let mut att = acts.att;
        let mut attproj = acts.attproj;
        let mut residual2 = acts.residual2;
        let mut ln2 = acts.ln2;
        let mut ln2_mean = acts.ln2_mean;
        let mut ln2_rstd = acts.ln2_rstd;
        let mut fch = acts.fch;
        let mut fch_gelu = acts.fch_gelu;
        let mut fcproj = acts.fcproj;
        let scratch = acts.output;

        for l in 0..self.config.num_layers {
            if l != 0 {
                residual = next_tensor!(ctx, residual3, batch_size * seq_len * self.config.channels)
            };
            /*
            float* l_ln1w = params.ln1w + l * C;
            float* l_ln1b = params.ln1b + l * C;
            float* l_qkvw = params.qkvw + l * 3*C * C;
            float* l_qkvb = params.qkvb + l * 3*C;
            float* l_attprojw = params.attprojw + l * C * C;
            float* l_attprojb = params.attprojb + l * C;
            float* l_ln2w = params.ln2w + l * C;
            float* l_ln2b = params.ln2b + l * C;
            float* l_fcw = params.fcw + l * 4*C * C;
            float* l_fcb = params.fcb + l * 4*C;
            float* l_fcprojw = params.fcprojw + l * C * 4*C;
            float* l_fcprojb = params.fcprojb + l * C;

            // get the pointers of the activations for this layer
            float* l_ln1 = acts.ln1 + l * B * T * C;
            float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            float* l_qkvr = acts.qkvr + l * B * T * 3*C;
            float* l_atty = acts.atty + l * B * T * C;
            float* l_att = acts.att + l * B * NH * T * T;
            float* l_attproj = acts.attproj + l * B * T * C;
            float* l_residual2 = acts.residual2 + l * B * T * C;
            float* l_ln2 = acts.ln2 + l * B * T * C;
            float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            float* l_fch = acts.fch + l * B * T * 4*C;
            float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
            float* l_fcproj = acts.fcproj + l * B * T * C;
            float* l_residual3 = acts.residual3 + l * B * T * C;
            // these are only needed as scratchpads for the forward pass, but
            // need not be stored for backward
            float* scratch = acts.output;
            */
            let l_ln1w = next_tensor!(ctx, ln1w, self.config.channels);
            let l_ln1b = next_tensor!(ctx, ln1b, self.config.channels);
            let l_qkvw = next_tensor!(ctx, qkvw, 3 * self.config.channels * self.config.channels);
            let l_qkvb = next_tensor!(ctx, qkvb, 3 * self.config.channels);
            let l_attprojw =
                next_tensor!(ctx, attprojw, self.config.channels * self.config.channels);
            let l_attprojb = next_tensor!(ctx, attprojb, self.config.channels);
            let l_ln2w = next_tensor!(ctx, ln2w, self.config.channels);
            let l_ln2b = next_tensor!(ctx, ln2b, self.config.channels);
            let l_fcw = next_tensor!(ctx, fcw, 4 * self.config.channels * self.config.channels);
            let l_fcb = next_tensor!(ctx, fcb, 4 * self.config.channels);
            let l_fcprojw =
                next_tensor!(ctx, fcprojw, self.config.channels * 4 * self.config.channels);
            let l_fcprojb = next_tensor!(ctx, fcprojb, self.config.channels);

            let l_ln1 = next_tensor!(ctx, ln1, batch_size * seq_len * self.config.channels);
            let l_ln1_mean = next_tensor!(ctx, ln1_mean, batch_size * seq_len);
            let l_ln1_rstd = next_tensor!(ctx, ln1_rstd, batch_size * seq_len);
            let l_qkvr = next_tensor!(ctx, qkvr, batch_size * seq_len * 3 * self.config.channels);
            let l_atty = next_tensor!(ctx, atty, batch_size * seq_len * self.config.channels);
            let l_att =
                next_tensor!(ctx, att, batch_size * self.config.num_heads * seq_len * seq_len);
            let l_attproj = next_tensor!(ctx, attproj, batch_size * seq_len * self.config.channels);
            let l_residual2 =
                next_tensor!(ctx, residual2, batch_size * seq_len * self.config.channels);
            let l_ln2 = next_tensor!(ctx, ln2, batch_size * seq_len * self.config.channels);
            let l_ln2_mean = next_tensor!(ctx, ln2_mean, batch_size * seq_len);
            let l_ln2_rstd = next_tensor!(ctx, ln2_rstd, batch_size * seq_len);
            let l_fch = next_tensor!(ctx, fch, batch_size * seq_len * 4 * self.config.channels);
            let l_fch_gelu =
                next_tensor!(ctx, fch_gelu, batch_size * seq_len * 4 * self.config.channels);
            let l_fcproj = next_tensor!(ctx, fcproj, batch_size * seq_len * self.config.channels);
            /*
            matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
            attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B*T*C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
            */
            layernorm_forward(
                ctx,
                self.module,
                l_ln1,
                l_ln1_mean,
                l_ln1_rstd,
                residual,
                l_ln1w,
                l_ln1b,
                batch_size,
                seq_len,
                self.config.channels,
            );
            // matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
            matmul_forward(
                ctx,
                self.module,
                scratch,
                l_ln1,
                l_qkvw,
                l_qkvb,
                batch_size,
                seq_len,
                self.config.channels,
                3 * self.config.channels,
            );
            /*
            attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B*T*C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
            */
            attention_forward(
                ctx,
                self.module,
                cublas_handle,
                l_atty,
                l_qkvr,
                l_att,
                scratch,
                batch_size,
                seq_len,
                self.config.channels,
                self.config.num_heads,
            );
            matmul_forward(
                ctx,
                self.module,
                l_attproj,
                l_atty,
                l_attprojw,
                l_attprojb,
                batch_size,
                seq_len,
                self.config.channels,
                self.config.channels,
            );
            residual_forward(
                ctx,
                self.module,
                l_residual2,
                residual,
                l_attproj,
                batch_size * seq_len * self.config.channels,
            );
            layernorm_forward(
                ctx,
                self.module,
                l_ln2,
                l_ln2_mean,
                l_ln2_rstd,
                l_residual2,
                l_ln2w,
                l_ln2b,
                batch_size,
                seq_len,
                self.config.channels,
            );
            matmul_forward(
                ctx,
                self.module,
                l_fch,
                l_ln2,
                l_fcw,
                l_fcb,
                batch_size,
                seq_len,
                self.config.channels,
                4 * self.config.channels,
            );

            gelu_forward(
                ctx,
                self.module,
                l_fch_gelu,
                l_fch,
                batch_size * seq_len * 4 * self.config.channels,
            );
            matmul_forward(
                ctx,
                self.module,
                l_fcproj,
                l_fch_gelu,
                l_fcprojw,
                l_fcprojb,
                batch_size,
                seq_len,
                4 * self.config.channels,
                self.config.channels,
            );
            residual_forward(
                ctx,
                self.module,
                residual3,
                l_residual2,
                l_fcproj,
                batch_size * seq_len * self.config.channels,
            );
        }

        let residual = residual3;
        /*
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
        matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);
        */
        layernorm_forward(
            ctx,
            self.module,
            acts.lnf,
            acts.lnf_mean,
            acts.lnf_rstd,
            residual,
            params.lnfw,
            params.lnfb,
            batch_size,
            seq_len,
            self.config.channels,
        );
        let empty_tensor = ctx.new_tensor_slice(&[]).unwrap();
        matmul_forward(
            ctx,
            self.module,
            scratch,
            acts.lnf,
            params.wte,
            empty_tensor,
            batch_size,
            seq_len,
            self.config.channels,
            self.config.padded_vocab_size,
        );

        /*
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->cpu_losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
        */
        if cpu_targets.len() == 0 {
            self.mean_loss = -1.0;
            return;
        }
        let loss_len = acts.losses.len();
        fused_classifier3(
            ctx,
            self.module,
            scratch,
            acts.losses,
            empty_tensor,
            targets,
            batch_size,
            seq_len,
            self.config.vocab_size,
            self.config.padded_vocab_size,
        );
        if self.cpu_losses.is_none() {
            println!("acts.losses len = {}", acts.losses.len());
            self.cpu_losses = Some(PinnedHostBox::new_from_tensor(ctx, &acts.losses).unwrap());
        } else {
            acts.losses
                .copy_to_host(self.cpu_losses.as_mut().unwrap(), acts.losses.len(), ctx)
                .unwrap();
        }
        let mean_loss =
            self.cpu_losses.as_ref().unwrap()[0..(batch_size * seq_len)].iter().sum::<f32>()
                / (batch_size * seq_len) as f32;
        self.mean_loss = mean_loss;
        println!("mean loss: {}", mean_loss);
    }
}
