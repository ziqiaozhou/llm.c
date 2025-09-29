use core::slice;
use std::fs::File;
use std::path::Path;

use gpu_host::{GpuCtxGuard, GpuCtxSpace, PinnedHostBox, TensorSliceMut};
use memmap2::Mmap;

pub(crate) mod dataloader;
mod kernels;
pub(crate) mod params;

use dataloader::parse_header_data;
use kernels::*;
use params::{ActivationTensors, GPT2Config, NUM_PARAMETER_TENSORS, ParameterTensors};

use crate::model::params::NUM_ACTIVATION_TENSORS;

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
        inputs: &[i32],
        targets: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) {
        assert!(inputs.len() >= batch_size * seq_len);
        assert!(targets.len() >= batch_size * seq_len);
        inputs
            .iter()
            .for_each(|&x| assert!(0 <= x && (x as usize) < self.config.vocab_size, "{}", x));
        targets
            .iter()
            .for_each(|&x| assert!(0 <= x && (x as usize) < self.config.vocab_size, "{}", x));

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
            println!("acts.losses len = {}", acts.losses.len());
            let cpu_losses = PinnedHostBox::new_from_tensor(ctx, &acts.losses).unwrap();
        } else {
            acts.losses
                .copy_to_host(self.cpu_losses.as_mut().unwrap(), acts.losses.len(), ctx)
                .unwrap();
        }

        unimplemented!()
    }
}
