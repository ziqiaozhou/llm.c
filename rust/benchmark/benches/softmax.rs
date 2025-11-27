mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};
use gpu::prelude::*;
use gpu_host::{GpuCtxGuard, GpuModule, TensorViewMut, cuda_ctx};

/*
fn test_softmax_forward_kernel5(N: u32, T: u32) {
    let LEN: usize = (N * T * T) as usize;
    let inv_temperature = random_f32_vec(1)[0];
    // input: (N, T, T) flattened
    let inp = random_float4_vec(LEN / 4);
    let inp_ref = inp.as_slice();
    let inp_f32 = inp_ref.flatten();

    let mut out = vec![0f32; LEN];

    // Reference CPU softmax
    let expected = softmax_cpu(inp_f32, N, T, inv_temperature);
    const BLOCK_SIZE: u32 = 256;
    let grid_size = (N * T * 32).div_ceil(BLOCK_SIZE);

    cuda_ctx(0, |ctx, m| {
        let d_inp = ctx
            .new_tensor_view::<[gpu::Float4]>(&inp)
            .expect("alloc failed");
        let mut d_out = ctx.new_tensor_view::<[f32]>(&out).expect("alloc failed");
        let config = gpu_host::gpu_config!(grid_size, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        softmax_forward_kernel5::launch(config, ctx, m, &mut d_out, inv_temperature, &d_inp, N, T)
            .expect("Failed to run softmax_forward_kernel5");
        d_out.copy_to_host(&mut out).expect("copy to host failed");
    });
    assert!(
        f32_eq(&out, &expected, 1e-5),
        "out not match:\n\n{:?}\n\n{:?}",
        &out[0..32],
        &expected[0..32],
    );
}

fn test_softmax_autoregressive_backward_kernel(B: u32, T: u32, C: u32, NH: u32) {
    let scale = 0.4;

    let len = (B * NH * T * T) as usize;
    // Example att and datt tensors (B, T, T) flattened
    let att = random_f32_vec(len);
    let datt = random_f32_vec(len);
    let mut dpreatt = vec![0f32; len];
    const BLOCK_SIZE: u32 = 256;
    const T_PER_BLOCK: u32 = 4;
    let gdim_x = T / T_PER_BLOCK;
    let gdim_y = B * NH;
    let expected = softmax_autoregressive_backward_kernel_cpu(&datt, &att, B, T, NH, scale);
    cuda_ctx(0, |ctx, m| {
        let d_att = ctx.new_tensor_view::<[f32]>(&att).expect("alloc failed");
        let d_datt = ctx.new_tensor_view::<[f32]>(&datt).expect("alloc failed");
        let mut d_dpreatt = ctx
            .new_tensor_view::<[f32]>(&dpreatt)
            .expect("alloc failed");
        let config = gpu_host::gpu_config!(gdim_x, gdim_y, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::softmax_autoregressive_backward_kernel::launch(
            config,
            ctx,
            m,
            &mut d_dpreatt,
            &d_datt,
            &d_att,
            B,
            T,
            C,
            scale,
        )
        .expect("Failed to run softmax_autoregressive_backward_kernel");
        d_dpreatt
            .copy_to_host(&mut dpreatt)
            .expect("copy to host failed");
    });
    assert!(
        f32_eq(&dpreatt, &expected, 1e-4),
        "dpreatt not match:\n\n{:?}\n\n{:?}",
        &dpreatt[500..520],
        &expected[500..520],
    );
}
*/
struct SoftMaxForward<'a> {
    config: Config,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
    out: gpu_host::TensorViewMut<'a, [f32]>,
    
}

impl<'a> KernelRunner<'a> for SoftMaxForward<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        m: &'a gpu_host::GpuModule<N>,
        config: Config,
    ) -> Self {
        let len = (config.batch_size * config.seq_len * config.channel) as usize;
        let inp = ctx
            .new_tensor_view(rand_f32_vec(len).as_slice())
            .expect("tensor alloc failed");
        let out = ctx
            .new_tensor_view(vec![0f32; len].as_slice())
            .expect("tensor alloc failed");
        Self { config, inp, out }
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llmrs::kernels::softmax_forward(
            ctx,
            m,
            &mut self.out,
            &self.inp,
            self.config.batch_size,
            self.config.seq_len,
            self.config.channel,
        );
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::softmax_forward_host(
                self.out.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
            );
        }
    }
}

struct SoftMaxBack<'a> {
    config: Config,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
    out: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a> KernelRunner<'a> for SoftMaxBack<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        m: &'a gpu_host::GpuModule<N>,
        config: Config,
    ) -> Self {
        let len = (config.batch_size * config.seq_len * config.channel) as usize;
        let inp = ctx
            .new_tensor_view(rand_f32_vec(len).as_slice())
            .expect("tensor alloc failed");
        let out = ctx
            .new_tensor_view(vec![0f32; len].as_slice())
            .expect("tensor alloc failed");
        Self { config, inp, out }
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llmrs::kernels::softmax_backward(
            ctx,
            m,
            &mut self.out,
            &self.inp,
            self.config.batch_size,
            self.config.seq_len,
            self.config.channel,
        );
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::softmax_backward_host(
                self.out.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
            );
        }
    }
}

fn softmax_bench(c: &mut Criterion) {
    gpu_host::cuda_ctx(0, |ctx, m| {
        bench_llm_rs::<_, SoftMaxForward>(c, "softmax_forward", ctx, m);
    });

    gpu_host::cuda_ctx(0, |ctx, m| {
        bench_llm_rs::<_, SoftMaxBack>(c, "softmax_backward", ctx, m);
    });
}

criterion_group! {
  name = softmax;
  config = Criterion::default().warm_up_time(Duration::from_secs(3));
  targets = softmax_bench
}

criterion_main!(softmax);
