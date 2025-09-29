use core::slice;
use std::fs::File;
use std::path::Path;

use gpu_host::{GpuCtxGuard, GpuCtxSpace, PinnedHostBox, TensorSliceMut};
use memmap2::Mmap;

mod kernels;
mod params;

use kernels::*;

use params::{ActivationTensors, NUM_PARAMETER_TENSORS, ParameterTensors};

use crate::model::params::NUM_ACTIVATION_TENSORS;

#[derive(Debug, Clone, PartialEq)]
pub struct GPT2Config {
    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Padded vocabulary size.
    pub padded_vocab_size: usize,

    /// Number of layers.
    pub num_layers: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of channels.
    pub channels: usize,
}

impl GPT2Config {
    /// Creates a new GPT2Config instance.
    ///
    /// # Returns
    ///
    /// A new `GPT2Config` instance.
    fn new() -> Self {
        GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            padded_vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        }
    }

    fn get_params_sizes(&self) -> [usize; NUM_PARAMETER_TENSORS] {
        let mut param_sizes = [0; NUM_PARAMETER_TENSORS];
        let config = self;

        param_sizes[0] = config.padded_vocab_size * config.channels; // wte
        param_sizes[1] = config.max_seq_len * config.channels; // wpe
        param_sizes[2] = config.num_layers * config.channels; // ln1w
        param_sizes[3] = config.num_layers * config.channels; // ln1b
        param_sizes[4] = config.num_layers * (3 * config.channels) * config.channels; // qkvw
        param_sizes[5] = config.num_layers * (3 * config.channels); // qkvb
        param_sizes[6] = config.num_layers * config.channels * config.channels; // attprojw
        param_sizes[7] = config.num_layers * config.channels; // attprojb
        param_sizes[8] = config.num_layers * config.channels; // ln2w
        param_sizes[9] = config.num_layers * config.channels; // ln2b
        param_sizes[10] = config.num_layers * (4 * config.channels) * config.channels; // fcw
        param_sizes[11] = config.num_layers * (4 * config.channels); // fcb
        param_sizes[12] = config.num_layers * config.channels * (4 * config.channels); // fcprojw
        param_sizes[13] = config.num_layers * config.channels; // fcprojb
        param_sizes[14] = config.channels; // lnfw
        param_sizes[15] = config.channels; // lnfb

        param_sizes
    }

    fn get_act_sizes(&self, batch_size: usize, seq_len: usize) -> [usize; NUM_ACTIVATION_TENSORS] {
        let mut act_sizes = [0; NUM_ACTIVATION_TENSORS];
        let config = self;

        act_sizes[0] = batch_size * seq_len * config.channels; // encoded
        act_sizes[1] = config.num_layers * batch_size * seq_len * config.channels; // ln1
        act_sizes[2] = config.num_layers * batch_size * seq_len; // ln1_mean
        act_sizes[3] = config.num_layers * batch_size * seq_len; // ln1_rstd
        act_sizes[4] = config.num_layers * batch_size * seq_len * config.channels; // atty
        act_sizes[5] = config.num_layers * batch_size * config.num_heads * seq_len * seq_len; // att
        act_sizes[6] = config.num_layers * batch_size * seq_len * config.channels; // attproj
        act_sizes[7] = config.num_layers * batch_size * seq_len * config.channels; // residual2
        act_sizes[8] = config.num_layers * batch_size * seq_len * config.channels; // ln2
        act_sizes[9] = config.num_layers * batch_size * seq_len; // ln2_mean
        act_sizes[10] = config.num_layers * batch_size * seq_len; // ln2_rstd
        act_sizes[11] = config.num_layers * batch_size * seq_len * 4 * config.channels; // fch
        act_sizes[12] = config.num_layers * batch_size * seq_len * 4 * config.channels; // fch_gelu
        act_sizes[13] = config.num_layers * batch_size * seq_len * config.channels; // fcproj
        act_sizes[14] = config.num_layers * batch_size * seq_len * config.channels; // residual3
        act_sizes[15] = batch_size * seq_len * config.channels; // lnf
        act_sizes[16] = batch_size * seq_len; // lnf_mean
        act_sizes[17] = batch_size * seq_len; // lnf_rstd
        act_sizes[18] = batch_size * seq_len; // losses
        act_sizes[19] = config.num_layers * batch_size * seq_len * 3 * config.channels; // qkvr
        act_sizes[20] = batch_size
            * seq_len
            * std::cmp::max(
                3 * config.channels,
                std::cmp::max(config.num_heads * seq_len, config.padded_vocab_size),
            ); // output / scratch

        act_sizes
    }
}

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

        // Read model header
        let model_len = model_data.len();
        let model_header = model_data[0..256 * 4]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        assert!(model_header.len() == 256);
        // Check magic number and version
        if model_header[0] != 20240326 {
            panic!("Bad magic model file");
        }
        if model_header[1] != 3 {
            panic!("Bad version in model file\n---> HINT: try to re-run `python train_gpt2.py`");
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

        let bytes = &model_data[256 * 4..model_len];
        let param_sizes = config.get_params_sizes();
        // Count the number of parameters
        let num_parameters: usize = param_sizes.iter().sum();
        println!("num_parameters: {}", num_parameters);
        let cpu_params =
            unsafe { slice::from_raw_parts(bytes.as_ptr() as *const f32, num_parameters) };
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
        inputs: &[i32],
        targets: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) {
        assert!(inputs.len() >= batch_size * seq_len);
        assert!(targets.len() >= batch_size * seq_len);
        assert!(inputs.iter().all(|&x| 0 <= x && (x as usize) < self.config.vocab_size));
        assert!(targets.iter().all(|&x| 0 <= x && (x as usize) < self.config.vocab_size));

        // allocate space for all the activations if needed (done here, lazily)
        if self.acts.is_none() {
            let act_sizes = self.config.get_act_sizes(batch_size, seq_len);
            self.num_activations = act_sizes.iter().sum();
            self.acts =
                Some(ActivationTensors::new(ctx, act_sizes, &vec![0.0f32; self.num_activations]));
            println!(
                "allocated {} MiB for activations",
                (self.num_activations * std::mem::size_of::<f32>()) / (1024 * 1024)
            );

            // also create memory for caching inputs and targets
            assert!(self.inputs.is_none());
            self.inputs = Some(ctx.new_tensor_slice(&inputs[0..batch_size * seq_len]).unwrap());
            assert!(self.targets.is_none());
            self.targets = Some(ctx.new_tensor_slice(&targets[0..batch_size * seq_len]).unwrap());
        } else {
            // validate B,T is consistent with how we've allocated the memory before
            assert!(self.batch_size == batch_size);
            assert!(self.seq_len == seq_len);
        }

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

        if self.cpu_losses.is_none() {
            let cpu_losses = PinnedHostBox::new_from_tensor(ctx, &acts.losses).unwrap();
        } else {
            acts.losses
                .copy_to_host(self.cpu_losses.as_mut().unwrap(), acts.losses.len(), ctx)
                .unwrap();
        }

        unimplemented!()
    }
}
