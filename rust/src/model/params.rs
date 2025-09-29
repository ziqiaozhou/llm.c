use gpu_host::{GpuCtxSpace, TensorSlice};

macro_rules! new_tensors {
    (
        pub const $param_len: ident: usize = $len: literal;
        pub struct $name_tensor:ident<'ctx, NS: GpuCtxSpace>;
        pub struct $name:ident<'ctx, NS: GpuCtxSpace> {
            $(
                $(#[$doc:meta])*
                pub $field:ident : TensorSlice<'ctx, f32, NS>,
            )*
        }
    ) => {
        pub const $param_len: usize = $len;

        pub struct $name_tensor<'ctx, NS: GpuCtxSpace> {
            pub tensor: TensorSlice<'ctx, f32, NS>,
            pub param_sizes: [usize; $param_len],
        }

        pub struct $name<'ctx, NS: GpuCtxSpace> {
            $(
                $(#[$doc])*
                pub $field: TensorSlice<'ctx, f32, NS>,
            )*
        }

        impl<'ctx, NS: GpuCtxSpace> $name_tensor<'ctx, NS> {
            pub fn new(ctx: &'ctx gpu_host::GpuCtxGuard<'ctx, '_, NS>, param_sizes: [usize; $param_len], init: &[f32]) -> Self {
                let tensor = ctx.new_tensor_slice(init).unwrap();
                $name_tensor { tensor, param_sizes }
            }

            pub fn inner<'a>(
                &'a mut self,
                ctx: &'ctx gpu_host::GpuCtxGuard<'ctx, '_, NS>,
            ) -> $name<'a, NS>
            {
                let param_sizes = self.param_sizes;
                let params = &mut self.tensor;
                let mut len = 0;
                $(
                    len += 1;
                    let ($field, params) = ctx.split_tensor_slice(params, param_sizes[len-1]).unwrap();
                )*
                let _ = params;
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
pub struct ParameterTensors<'ctx, NS: GpuCtxSpace>;
pub struct ParameterTensorsInner<'ctx, NS: GpuCtxSpace> {
    /// Token embeddings (V, C).
    pub wte: TensorSlice<'ctx, f32, NS>,

    /// Position embeddings (maxT, C).
    pub wpe: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization weights for the first layer (L, C).
    pub ln1w: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization biases for the first layer (L, C).
    pub ln1b: TensorSlice<'ctx, f32, NS>,

    /// Query, Key, Value weights (L, 3*C, C).
    pub qkvw: TensorSlice<'ctx, f32, NS>,

    /// Query, Key, Value biases (L, 3*C).
    pub qkvb: TensorSlice<'ctx, f32, NS>,

    /// Attention projection weights (L, C, C).
    pub attprojw: TensorSlice<'ctx, f32, NS>,

    /// Attention projection biases (L, C).
    pub attprojb: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization weights for the second layer (L, C).
    pub ln2w: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization biases for the second layer (L, C).
    pub ln2b: TensorSlice<'ctx, f32, NS>,

    /// Fully connected weights (L, 4*C, C).
    pub fcw: TensorSlice<'ctx, f32, NS>,

    /// Fully connected biases (L, 4*C).
    pub fcb: TensorSlice<'ctx, f32, NS>,

    /// Fully connected projection weights (L, C, 4*C).
    pub fcprojw: TensorSlice<'ctx, f32, NS>,

    /// Fully connected projection biases (L, C).
    pub fcprojb: TensorSlice<'ctx, f32, NS>,

    /// Final layer normalization weights (C).
    pub lnfw: TensorSlice<'ctx, f32, NS>,

    /// Final layer normalization biases (C).
    pub lnfb: TensorSlice<'ctx, f32, NS>,
}
}

new_tensors! {
pub const NUM_ACTIVATION_TENSORS: usize = 23;

pub struct ActivationTensors<'ctx, NS: GpuCtxSpace>;

pub struct ActivationTensorsInner<'ctx, NS: GpuCtxSpace> {
    /// Encoded (B, T, C)
    pub encoded: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization 1 (L, B, T, C)
    pub ln1: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization 1 mean (L, B, T)
    pub ln1_mean: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization 1 reciprocal std (L, B, T)
    pub ln1_rstd: TensorSlice<'ctx, f32, NS>,

    /// Query, Key, Value (L, B, T, 3*C)
    pub qkv: TensorSlice<'ctx, f32, NS>,

    /// Attention output (L, B, T, C)
    pub atty: TensorSlice<'ctx, f32, NS>,

    /// Pre-attention scores (L, B, NH, T, T)
    pub preatt: TensorSlice<'ctx, f32, NS>,

    /// Attention scores (L, B, NH, T, T)
    pub att: TensorSlice<'ctx, f32, NS>,

    /// Attention projection (L, B, T, C)
    pub attproj: TensorSlice<'ctx, f32, NS>,

    /// Second residual connection (L, B, T, C)
    pub residual2: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization 2 (L, B, T, C)
    pub ln2: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization 2 mean (L, B, T)
    pub ln2_mean: TensorSlice<'ctx, f32, NS>,

    /// Layer normalization 2 reciprocal std (L, B, T)
    pub ln2_rstd: TensorSlice<'ctx, f32, NS>,

    /// Fully connected hidden (L, B, T, 4*C)
    pub fch: TensorSlice<'ctx, f32, NS>,

    /// Fully connected hidden GELU activation (L, B, T, 4*C)
    pub fch_gelu: TensorSlice<'ctx, f32, NS>,

    /// Fully connected projection (L, B, T, C)
    pub fcproj: TensorSlice<'ctx, f32, NS>,

    /// Third residual connection (L, B, T, C)
    pub residual3: TensorSlice<'ctx, f32, NS>,

    /// Final layer normalization (B, T, C)
    pub lnf: TensorSlice<'ctx, f32, NS>,

    /// Final layer normalization mean (B, T)
    pub lnf_mean: TensorSlice<'ctx, f32, NS>,

    /// Final layer normalization reciprocal std (B, T)
    pub lnf_rstd: TensorSlice<'ctx, f32, NS>,

    /// Logits (B, T, V)
    pub logits: TensorSlice<'ctx, f32, NS>,

    /// Probabilities (B, T, V)
    pub probs: TensorSlice<'ctx, f32, NS>,

    /// Losses (B, T)
    pub losses: TensorSlice<'ctx, f32, NS>,
}
}
