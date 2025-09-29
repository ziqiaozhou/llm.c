use gpu_host::{CudaMemSlice, GpuCtxGuard, GpuCtxSpace, GpuModule};
use llm_rs_gpu::*;

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
    let grid_size = (n + BSIZE - 1).div_ceil(BSIZE);
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
