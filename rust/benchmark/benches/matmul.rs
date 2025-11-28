mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

struct MatMulBack<'a> {
    bias: gpu_host::TensorViewMut<'a, [f32]>,
    dout: gpu_host::TensorViewMut<'a, [f32]>,
    config: Config,
}

impl<'a> KernelRunner<'a> for MatMulBack<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let bias = ctx
            .new_tensor_view(rand_f32_vec(config.out_channel).as_slice())
            .expect("tensor alloc failed");
        let dout = ctx
            .new_tensor_view(
                rand_f32_vec(config.batch_size * config.seq_len * config.out_channel).as_slice(),
            )
            .expect("tensor alloc failed");
        Some(Self { bias, dout, config })
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llmrs::kernels::matmul_backward_bias_kernel4(
            ctx,
            m,
            &mut self.bias,
            &self.dout,
            self.config.batch_size, // batch size
            self.config.seq_len,    // seq length
            self.config.out_channel,
        );
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::matmul_backward_bias_kernel4_host(
                self.bias.as_devptr() as _,
                self.dout.as_devptr() as _,
                self.config.batch_size as _, // batch size
                self.config.seq_len as _,    // seq length
                self.config.out_channel as _,
            );
        }
    }
}

struct MatMulForward<'a> {
    bias: gpu_host::TensorViewMut<'a, [f32]>,
    out: gpu_host::TensorViewMut<'a, [f32]>,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
    weight: gpu_host::TensorViewMut<'a, [f32]>,
    config: Config,
}

impl<'a> KernelRunner<'a> for MatMulForward<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let channel = config.out_channel / 4;
        let bias = ctx
            .new_tensor_view(rand_f32_vec(channel).as_slice())
            .expect("tensor alloc failed");
        let out = ctx
            .new_tensor_view(
                rand_f32_vec(config.batch_size * config.seq_len * config.out_channel)
                    .as_slice(),
            )
            .expect("tensor alloc failed");
        let inp = ctx
            .new_tensor_view(
                rand_f32_vec(config.batch_size * config.seq_len * channel).as_slice(),
            )
            .expect("tensor alloc failed");
        let weight = ctx
            .new_tensor_view(rand_f32_vec(channel * channel).as_slice())
            .expect("tensor alloc failed");
        Some(Self { bias, out, inp, weight, config })
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llmrs::kernels::matmul_forward(
            ctx,
            m,
            &mut self.out,
            &self.inp,
            &self.weight,
            &self.bias,
            self.config.batch_size,
            self.config.seq_len,
            self.config.channel,
            self.config.out_channel,
        );
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::matmul_forward_host(
                self.out.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.weight.as_devptr() as _,
                self.bias.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
                self.config.out_channel as _,
            );
        }
    }
}

fn matmul_bench(c: &mut Criterion) {
    gpu_host::cuda_ctx(0, |ctx, m| {
        bench_llm_rs::<_, MatMulForward>(c, "matmul_forward", ctx, m);
        bench_llm_rs::<_, MatMulBack>(c, "matmul_back", ctx, m);
    });
}

criterion_group! {
  name = matmul;
  config = Criterion::default().warm_up_time(Duration::from_secs(3));
  targets = matmul_bench
}

criterion_main!(matmul);
