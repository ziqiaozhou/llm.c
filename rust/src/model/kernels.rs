use gpu::float4;
use gpu_host::{CudaMemSlice, GpuCtxGuard, GpuCtxSpace, GpuModule};
use llm_rs_gpu::*;

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
    println!("encoder_forward grid size: {}", grid_size);
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
    assert!(inp.len() == len);
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
