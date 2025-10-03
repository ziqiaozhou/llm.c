use gpu_host::{GpuCtxSpace, TensorViewMut};

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
    pub(crate) fn new() -> Self {
        GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            padded_vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        }
    }

    pub(crate) fn get_params_sizes(&self) -> [usize; NUM_PARAMETER_TENSORS] {
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

    pub(crate) fn get_act_sizes(
        &self,
        batch_size: usize,
        seq_len: usize,
    ) -> [usize; NUM_ACTIVATION_TENSORS] {
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

    /*
        void fill_in_grad_act_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
        size_t NH = config.num_heads;
        size_t C = config.channels;
        act_sizes[0] = B * T * 4 * C; // bt4c
        act_sizes[1] = B * NH * T * T; // preatt
        act_sizes[2] = B * T * C; // residual3
    }
        */
    pub fn get_grad_act_sizes(&self, batch_size: usize, seq_len: usize) -> [usize; 3] {
        let mut grad_act_sizes = [0; 3];
        let config = self;

        grad_act_sizes[0] = batch_size * seq_len * 4 * config.channels; // bt4c
        grad_act_sizes[1] = batch_size * config.num_heads * seq_len * seq_len; // preatt
        grad_act_sizes[2] = batch_size * seq_len * config.channels; // residual3

        grad_act_sizes
    }
}

macro_rules! new_tensors {
    (
        pub const $param_len: ident: usize = $len: literal;
        pub struct $name_tensor:ident<'g> {
            pub tensor: TensorViewMut<'g, [$elem_ty: ty]>,
        }
        pub struct $name:ident<'g> {
            $(
                $(#[$doc:meta])*
                pub $field:ident : TensorViewMut<'g, [$_elem_ty: ty]>,
            )*
        }
    ) => {
        pub const $param_len: usize = $len;

        pub struct $name_tensor<'g> {
            pub tensor: TensorViewMut<'g, [$elem_ty]>,
            pub param_sizes: [usize; $param_len],
        }

        pub struct $name<'g> {
            $(
                $(#[$doc])*
                pub $field: TensorViewMut<'g, [$_elem_ty]>,
            )*
        }

        impl<'g> $name_tensor<'g> {
            pub fn new<'ctx, 'a: 'g, NS: GpuCtxSpace>(ctx: &'g gpu_host::GpuCtxGuard<'ctx, 'a, NS>, param_sizes: [usize; $param_len], init: &[$elem_ty]) -> Self {
                let tensor = ctx.new_tensor_view(init).unwrap();
                $name_tensor { tensor, param_sizes }
            }

            pub fn inner<'a>(&'a mut self) -> $name<'a>
            {
                let param_sizes = self.param_sizes;
                let mut params = self.tensor.index_mut(..);
                let mut i = 0;
                $(
                    let ($field, mut params) = params.split(param_sizes[i]);
                    i += 1;
                )*
                let _ = params;
                let _ = i;
                $name {
                    $(
                        $field,
                    )*
                }
            }
        }
    };
}

new_tensors! {
pub const NUM_PARAMETER_TENSORS: usize = 16;
pub struct ParameterTensors<'g> {
    pub tensor: TensorViewMut<'g, [f32]>,
}
pub struct ParameterTensorsInner<'g> {
    /// Token embeddings (V, C).
    pub wte: TensorViewMut<'g, [f32]>,

    /// Position embeddings (maxT, C).
    pub wpe: TensorViewMut<'g, [f32]>,

    /// Layer normalization weights for the first layer (L, C).
    pub ln1w: TensorViewMut<'g, [f32]>,

    /// Layer normalization biases for the first layer (L, C).
    pub ln1b: TensorViewMut<'g, [f32]>,

    /// Query, Key, Value weights (L, 3*C, C).
    pub qkvw: TensorViewMut<'g, [f32]>,

    /// Query, Key, Value biases (L, 3*C).
    pub qkvb: TensorViewMut<'g, [f32]>,

    /// Attention projection weights (L, C, C).
    pub attprojw: TensorViewMut<'g, [f32]>,

    /// Attention projection biases (L, C).
    pub attprojb: TensorViewMut<'g, [f32]>,

    /// Layer normalization weights for the second layer (L, C).
    pub ln2w: TensorViewMut<'g, [f32]>,

    /// Layer normalization biases for the second layer (L, C).
    pub ln2b: TensorViewMut<'g, [f32]>,

    /// Fully connected weights (L, 4*C, C).
    pub fcw: TensorViewMut<'g, [f32]>,

    /// Fully connected biases (L, 4*C).
    pub fcb: TensorViewMut<'g, [f32]>,

    /// Fully connected projection weights (L, C, 4*C).
    pub fcprojw: TensorViewMut<'g, [f32]>,

    /// Fully connected projection biases (L, C).
    pub fcprojb: TensorViewMut<'g, [f32]>,

    /// Final layer normalization weights (C).
    pub lnfw: TensorViewMut<'g, [f32]>,

    /// Final layer normalization biases (C).
    pub lnfb: TensorViewMut<'g, [f32]>,
}
}

new_tensors! {
pub const NUM_ACTIVATION_TENSORS: usize = 21;

pub struct ActivationTensors<'g> {
    pub tensor: TensorViewMut<'g, [f32]>,
}

/*
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* atty; // (L, B, T, C)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)

    float* losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    float* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    float* output;
} ActivationTensors;
 */
pub struct ActivationTensorsInner<'g> {
    /// Encoded (B, T, C)
    pub encoded: TensorViewMut<'g, [f32]>,

    /// Layer normalization 1 (L, B, T, C)
    pub ln1: TensorViewMut<'g, [f32]>,

    /// Layer normalization 1 mean (L, B, T)
    pub ln1_mean: TensorViewMut<'g, [f32]>,

    /// Layer normalization 1 reciprocal std (L, B, T)
    pub ln1_rstd: TensorViewMut<'g, [f32]>,

    /// Attention output (L, B, T, C)
    pub atty: TensorViewMut<'g, [f32]>,

    /// Attention scores (L, B, NH, T, T)
    pub att: TensorViewMut<'g, [f32]>,

    /// Attention projection (L, B, T, C)
    pub attproj: TensorViewMut<'g, [f32]>,

    /// Second residual connection (L, B, T, C)
    pub residual2: TensorViewMut<'g, [f32]>,

    /// Layer normalization 2 (L, B, T, C)
    pub ln2: TensorViewMut<'g, [f32]>,

    /// Layer normalization 2 mean (L, B, T)
    pub ln2_mean: TensorViewMut<'g, [f32]>,

    /// Layer normalization 2 reciprocal std (L, B, T)
    pub ln2_rstd: TensorViewMut<'g, [f32]>,

    /// Fully connected hidden (L, B, T, 4*C)
    pub fch: TensorViewMut<'g, [f32]>,

    /// Fully connected hidden GELU activation (L, B, T, 4*C)
    pub fch_gelu: TensorViewMut<'g, [f32]>,

    /// Fully connected projection (L, B, T, C)
    pub fcproj: TensorViewMut<'g, [f32]>,

    /// Third residual connection (L, B, T, C)
    pub residual3: TensorViewMut<'g, [f32]>,

    /// Final layer normalization (B, T, C)
    pub lnf: TensorViewMut<'g, [f32]>,

    /// Final layer normalization mean (B, T)
    pub lnf_mean: TensorViewMut<'g, [f32]>,

    /// Final layer normalization reciprocal std (B, T)
    pub lnf_rstd: TensorViewMut<'g, [f32]>,

    /// Losses (B, T)
    pub losses: TensorViewMut<'g, [f32]>,
    /// Query, Key, Value (L, B, T, 3*C)
    pub qkvr: TensorViewMut<'g, [f32]>,
    pub output: TensorViewMut<'g, [f32]>, // (B, T, max(3*C, NH*T, V))
}
}

new_tensors! {
pub const NUM_BATCH_TENSORS: usize = 2;
pub struct BatchTensor<'g> {
    pub tensor: TensorViewMut<'g, [i32]>,
}
pub struct BatchTensorInner<'g> {
    pub input: TensorViewMut<'g, [i32]>,
    pub target: TensorViewMut<'g, [i32]>,
}
}

/*
typedef struct {
    float* bt4c; // (B, T, 4*C)
    float* preatt; // (B, NH, T, T)
    float* residual3; // (B, T, C)
} GradActTensors;*/

new_tensors! {
pub const NUM_GRAD_ACT_TENSORS: usize = 3;
pub struct GradActTensors<'g> {
    pub tensor: TensorViewMut<'g, [f32]>,
}
pub struct GradActTensorsInner<'g> {
    pub bt4c: TensorViewMut<'g, [f32]>, // (B, T, 4*C)
    pub preatt: TensorViewMut<'g, [f32]>, // (B, NH, T, T)
    pub residual3: TensorViewMut<'g, [f32]>, // (B, T, C)
}
}
