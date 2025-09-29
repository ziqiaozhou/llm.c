use gpu_host::{GpuCtxSpace, TensorSliceMut};

macro_rules! new_tensors {
    (
        pub const $param_len: ident: usize = $len: literal;
        pub struct $name_tensor:ident<'ctx, NS: GpuCtxSpace> {
            pub tensor: TensorSliceMut<'ctx, $elem_ty: ty, NS>,
        }
        pub struct $name:ident<'ctx, NS: GpuCtxSpace> {
            $(
                $(#[$doc:meta])*
                pub $field:ident : TensorSliceMut<'ctx, $_elem_ty: ty, NS>,
            )*
        }
    ) => {
        pub const $param_len: usize = $len;

        pub struct $name_tensor<'ctx, NS: GpuCtxSpace> {
            pub tensor: TensorSliceMut<'ctx, $elem_ty, NS>,
            pub param_sizes: [usize; $param_len],
        }

        pub struct $name<'ctx, NS: GpuCtxSpace> {
            $(
                $(#[$doc])*
                pub $field: TensorSliceMut<'ctx, $elem_ty, NS>,
            )*
        }

        impl<'ctx, NS: GpuCtxSpace> $name_tensor<'ctx, NS> {
            pub fn new(ctx: &'ctx gpu_host::GpuCtxGuard<'ctx, '_, NS>, param_sizes: [usize; $param_len], init: &[$elem_ty]) -> Self {
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
pub struct ParameterTensors<'ctx, NS: GpuCtxSpace> {
    pub tensor: TensorSliceMut<'ctx, f32, NS>,
}
pub struct ParameterTensorsInner<'ctx, NS: GpuCtxSpace> {
    /// Token embeddings (V, C).
    pub wte: TensorSliceMut<'ctx, f32, NS>,

    /// Position embeddings (maxT, C).
    pub wpe: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization weights for the first layer (L, C).
    pub ln1w: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization biases for the first layer (L, C).
    pub ln1b: TensorSliceMut<'ctx, f32, NS>,

    /// Query, Key, Value weights (L, 3*C, C).
    pub qkvw: TensorSliceMut<'ctx, f32, NS>,

    /// Query, Key, Value biases (L, 3*C).
    pub qkvb: TensorSliceMut<'ctx, f32, NS>,

    /// Attention projection weights (L, C, C).
    pub attprojw: TensorSliceMut<'ctx, f32, NS>,

    /// Attention projection biases (L, C).
    pub attprojb: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization weights for the second layer (L, C).
    pub ln2w: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization biases for the second layer (L, C).
    pub ln2b: TensorSliceMut<'ctx, f32, NS>,

    /// Fully connected weights (L, 4*C, C).
    pub fcw: TensorSliceMut<'ctx, f32, NS>,

    /// Fully connected biases (L, 4*C).
    pub fcb: TensorSliceMut<'ctx, f32, NS>,

    /// Fully connected projection weights (L, C, 4*C).
    pub fcprojw: TensorSliceMut<'ctx, f32, NS>,

    /// Fully connected projection biases (L, C).
    pub fcprojb: TensorSliceMut<'ctx, f32, NS>,

    /// Final layer normalization weights (C).
    pub lnfw: TensorSliceMut<'ctx, f32, NS>,

    /// Final layer normalization biases (C).
    pub lnfb: TensorSliceMut<'ctx, f32, NS>,
}
}

new_tensors! {
pub const NUM_ACTIVATION_TENSORS: usize = 23;

pub struct ActivationTensors<'ctx, NS: GpuCtxSpace> {
    pub tensor: TensorSliceMut<'ctx, f32, NS>,
}

pub struct ActivationTensorsInner<'ctx, NS: GpuCtxSpace> {
    /// Encoded (B, T, C)
    pub encoded: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization 1 (L, B, T, C)
    pub ln1: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization 1 mean (L, B, T)
    pub ln1_mean: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization 1 reciprocal std (L, B, T)
    pub ln1_rstd: TensorSliceMut<'ctx, f32, NS>,

    /// Query, Key, Value (L, B, T, 3*C)
    pub qkv: TensorSliceMut<'ctx, f32, NS>,

    /// Attention output (L, B, T, C)
    pub atty: TensorSliceMut<'ctx, f32, NS>,

    /// Pre-attention scores (L, B, NH, T, T)
    pub preatt: TensorSliceMut<'ctx, f32, NS>,

    /// Attention scores (L, B, NH, T, T)
    pub att: TensorSliceMut<'ctx, f32, NS>,

    /// Attention projection (L, B, T, C)
    pub attproj: TensorSliceMut<'ctx, f32, NS>,

    /// Second residual connection (L, B, T, C)
    pub residual2: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization 2 (L, B, T, C)
    pub ln2: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization 2 mean (L, B, T)
    pub ln2_mean: TensorSliceMut<'ctx, f32, NS>,

    /// Layer normalization 2 reciprocal std (L, B, T)
    pub ln2_rstd: TensorSliceMut<'ctx, f32, NS>,

    /// Fully connected hidden (L, B, T, 4*C)
    pub fch: TensorSliceMut<'ctx, f32, NS>,

    /// Fully connected hidden GELU activation (L, B, T, 4*C)
    pub fch_gelu: TensorSliceMut<'ctx, f32, NS>,

    /// Fully connected projection (L, B, T, C)
    pub fcproj: TensorSliceMut<'ctx, f32, NS>,

    /// Third residual connection (L, B, T, C)
    pub residual3: TensorSliceMut<'ctx, f32, NS>,

    /// Final layer normalization (B, T, C)
    pub lnf: TensorSliceMut<'ctx, f32, NS>,

    /// Final layer normalization mean (B, T)
    pub lnf_mean: TensorSliceMut<'ctx, f32, NS>,

    /// Final layer normalization reciprocal std (B, T)
    pub lnf_rstd: TensorSliceMut<'ctx, f32, NS>,

    /// Logits (B, T, V)
    pub logits: TensorSliceMut<'ctx, f32, NS>,

    /// Probabilities (B, T, V)
    pub probs: TensorSliceMut<'ctx, f32, NS>,

    /// Losses (B, T)
    pub losses: TensorSliceMut<'ctx, f32, NS>,
}
}

new_tensors! {
pub const NUM_BATCH_TENSORS: usize = 2;
pub struct BatchTensor<'ctx, NS: GpuCtxSpace> {
    pub tensor: TensorSliceMut<'ctx, i32, NS>,
}
pub struct BatchTensorInner<'ctx, NS: GpuCtxSpace> {
    pub input: TensorSliceMut<'ctx, i32, NS>,
    pub target: TensorSliceMut<'ctx, i32, NS>,
}
}
