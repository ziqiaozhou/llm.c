use cublas_sys::cublasSgemmStridedBatched;
use cudarc::cublas::sys as cublas_sys;
use gpu::float4;
use gpu_host::{CudaMemSlice, GpuCtxGuard, GpuCtxSpace, GpuModule};
use llm_rs_gpu::*;

macro_rules! next_tensor {
    ($ctx: ident, $params:ident, $size: expr) => {{
        let (left, right) = $ctx.split_tensor_slice($params, $size).unwrap();
        $params = right;
        assert!(left.len() == $size);
        left
    }};
}

struct GPUExecContext<'ctx, 'a, CN: GpuCtxSpace> {
    pub ctx: GpuCtxGuard<'ctx, 'a, CN>,
    pub m: GpuModule<CN>,
    pub cublas_handle: cublas_sys::cublasHandle_t,
}

pub fn encoder_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &'ctx mut CudaMemSlice<f32, CN>,
    inp: &'ctx CudaMemSlice<i32, CN>,
    wte: &'ctx CudaMemSlice<f32, CN>,
    wpe: &'ctx CudaMemSlice<f32, CN>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
) {
    assert!(channel % 4 == 0);
    let n = batch_size * seq_len * channel;
    const BSIZE: usize = 512;
    let grid_size = (n / 4).div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 0, 0, @const BSIZE as u32, 0, 0, 0);
    let out = unsafe { &mut *(out as *mut _ as *mut CudaMemSlice<float4, CN>) };
    let wte = unsafe { &*(wte as *const _ as *const CudaMemSlice<float4, CN>) };
    let wpe = unsafe { &*(wpe as *const _ as *const CudaMemSlice<float4, CN>) };
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    encoder_forward_kernel3::launch(
        config,
        ctx,
        m,
        out,
        inp,
        wte,
        wpe,
        batch_size as _,
        seq_len as _,
        channel as _,
    )
    .expect("Failed to run encoder_forward_kernel3");
}

/*
void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}
*/

pub fn encoder_backward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    dwte: &'ctx mut CudaMemSlice<f32, CN>,
    dwpe: &'ctx mut CudaMemSlice<f32, CN>,
    dout: &'ctx CudaMemSlice<f32, CN>,
    inp: &'ctx CudaMemSlice<i32, CN>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
) {
    let n = batch_size * seq_len * channel;
    const BSIZE: usize = 256;
    let grid_size = n.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 0, 0, @const BSIZE as u32, 0, 0, 0);
    encoder_backward_kernel::launch(
        config,
        ctx,
        m,
        dwte,
        dwpe,
        dout,
        inp,
        batch_size as _,
        seq_len as _,
        channel as _,
    )
    .expect("failed to launch encoder_backward_kernel");
}

/*
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn layernorm_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &'ctx mut CudaMemSlice<f32, CN>,
    mean: &'ctx mut CudaMemSlice<f32, CN>,
    rstd: &'ctx mut CudaMemSlice<f32, CN>,
    inp: &CudaMemSlice<f32, CN>,
    weight: &CudaMemSlice<f32, CN>,
    bias: &CudaMemSlice<f32, CN>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
) {
    let n = batch_size * seq_len;
    const BSIZE: usize = 512;
    let grid_size = (n * 32).div_ceil(BSIZE);
    let len = channel * n;
    assert!(inp.len() == len, "{} != {}", inp.len(), len);
    assert!(out.len() == len);
    assert!(mean.len() == n);
    assert!(rstd.len() == n);
    assert!(weight.len() == channel);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    layernorm_forward_kernel3::launch(
        config,
        ctx,
        m,
        out,
        mean,
        rstd,
        inp,
        weight,
        bias,
        n as _,
        channel as _,
    )
    .expect("failed to launch layernorm_forward_kernel3");
}

/*
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    int sqrt_block_size = 16;

    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}*/

pub(crate) fn matmul_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &'ctx mut CudaMemSlice<f32, CN>,
    inp: &CudaMemSlice<f32, CN>,
    weight: &CudaMemSlice<f32, CN>,
    bias: &CudaMemSlice<f32, CN>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    out_channel: usize,
) {
    let n = batch_size * seq_len;
    const SQRT_BLOCK_SIZE: usize = 16;
    let grid_x = (n).div_ceil(8 * SQRT_BLOCK_SIZE);
    let grid_y = (out_channel).div_ceil(8 * SQRT_BLOCK_SIZE);
    let config = gpu_host::gpu_config!(grid_x as u32, grid_y as u32, 1, @const SQRT_BLOCK_SIZE as u32, @const SQRT_BLOCK_SIZE as u32, 1, 0);
    matmul_forward_kernel4::launch(
        config,
        ctx,
        m,
        out,
        inp,
        weight,
        bias,
        channel as _,
        out_channel as _,
    )
    .expect("failed to launch matmul_forward_kernel4");
}

/*
void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // new approach: first cuBLAS another batched matmul
    float* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn attention_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    cublas_handle: cublas_sys::cublasHandle_t,
    out: &'ctx mut CudaMemSlice<f32, CN>,
    qkvr: &'ctx mut CudaMemSlice<f32, CN>,
    att: &'ctx mut CudaMemSlice<f32, CN>,
    inp: &CudaMemSlice<f32, CN>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    num_heads: usize,
) {
    const BSIZE: usize = 256;
    let head_size = channel / num_heads;
    assert!(channel % num_heads == 0);
    let total_threads = batch_size * num_heads * seq_len * head_size;
    let num_blocks = total_threads.div_ceil(BSIZE);
    let mut qkvr = qkvr;
    let q = next_tensor!(ctx, qkvr, batch_size * seq_len * channel);
    let k = next_tensor!(ctx, qkvr, batch_size * seq_len * channel);
    let v = next_tensor!(ctx, qkvr, batch_size * seq_len * channel);
    let _ = qkvr;
    let config = gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    permute_kernel::launch(
        config,
        ctx,
        m,
        q,
        k,
        v,
        inp,
        batch_size as _,
        seq_len as _,
        num_heads as _,
        head_size as _,
    )
    .expect("failed to launch permute_kernel");
    // batched matrix multiply with cuBLAS
    const CUBLAS_OP_T: cublas_sys::cublasOperation_t = cublas_sys::cublasOperation_t::CUBLAS_OP_T;
    const CUBLAS_OP_N: cublas_sys::cublasOperation_t = cublas_sys::cublasOperation_t::CUBLAS_OP_N;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let preatt = inp;
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len as i32,
            seq_len as i32,
            head_size as i32,
            &alpha,
            k.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            q.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            &beta,
            preatt.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // multiply all elements of preatt elementwise by scale
    let scale = 1.0f32 / (head_size as f32).sqrt();
    let grid_size = (batch_size * num_heads * seq_len * 32).div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    softmax_forward_kernel5::launch(
        config,
        ctx,
        m,
        att,
        scale,
        preatt,
        (batch_size * num_heads) as _,
        seq_len as _,
    )
    .expect("failed to launch softmax_forward_kernel5");

    let vaccum = inp;
    // new approach: first cuBLAS another batched matmul
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_size as i32,
            seq_len as i32,
            seq_len as i32,
            &alpha,
            v.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            att.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            &beta,
            vaccum.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    // now unpermute
    let total_threads = batch_size * seq_len * channel;
    let num_blocks = total_threads.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    unpermute_kernel::launch(
        config,
        ctx,
        m,
        vaccum,
        out,
        batch_size as _,
        seq_len as _,
        num_heads as _,
        head_size as _,
    )
    .expect("failed to launch unpermute_kernel");
}

/*
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn residual_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &'ctx mut CudaMemSlice<f32, CN>,
    inp1: &CudaMemSlice<f32, CN>,
    inp2: &CudaMemSlice<f32, CN>,
    n: usize,
) {
    const BSIZE: usize = 256;
    let grid_size = n.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    residual_forward_kernel::launch(config, ctx, m, out, inp1, inp2, n as _)
        .expect("failed to launch residual_forward_kernel");
}

/*
void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}*/

pub(crate) fn gelu_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &'ctx mut CudaMemSlice<f32, CN>,
    inp: &CudaMemSlice<f32, CN>,
    n: usize,
) {
    const BSIZE: usize = 128;
    let grid_size = n.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    gelu_forward_kernel::launch(config, ctx, m, out, inp, n as _)
        .expect("failed to launch gelu_forward_kernel");
}

/*
void fused_classifier3(float* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn fused_classifier3<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    logits: &'ctx mut CudaMemSlice<f32, CN>,
    losses: &'ctx mut CudaMemSlice<f32, CN>,
    dlosses: &'ctx CudaMemSlice<f32, CN>,
    targets: &'ctx CudaMemSlice<i32, CN>,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    pad_vocab_size: usize,
) {
    const BSIZE: usize = 1024;
    let grid_size = batch_size * seq_len;
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    let empty_tensor = ctx.new_tensor_slice(&[]).unwrap();
    fused_classifier_kernel3::launch(
        config,
        ctx,
        m,
        logits,
        losses,
        empty_tensor,
        dlosses,
        targets,
        batch_size as _,
        seq_len as _,
        vocab_size as _,
        pad_vocab_size as _,
    )
    .expect("failed to launch fused_classifier_kernel3");
}
