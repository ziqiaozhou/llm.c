use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::Rng;

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn random_f32_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<f32>()).collect()
}

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn random_float4_vec(n: usize) -> Vec<gpu::Float4> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            gpu::Float4::new([
                rng.random::<f32>(),
                rng.random::<f32>(),
                rng.random::<f32>(),
                rng.random::<f32>(),
            ])
        })
        .collect()
}

#[allow(dead_code)]
pub fn random_i32_vec(n: usize) -> Vec<i32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<i32>()).collect()
}

fn matmul_back_benchmarks<N: gpu_host::GpuCtxSpace>(
    ctx: &gpu_host::GpuCtxGuard<N>,
    m: &gpu_host::GpuModule<N>,
    c: &mut Criterion,
) {
    let mut group = c.benchmark_group("matmul_backward");
    for batch_size in [1] {
        for seq_length_order in (10..20).step_by(4) {
            let seq_length = 1 << seq_length_order;
            for out_channel in [128, 1024] {
                let mut dbias = ctx
                    .new_tensor_view(random_f32_vec(out_channel as usize).as_slice())
                    .expect("tensor alloc failed");
                let dout = ctx
                    .new_tensor_view(
                        random_f32_vec((batch_size * seq_length * out_channel) as usize).as_slice(),
                    )
                    .expect("tensor alloc failed");
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "llmrs_{}_{}_{}",
                        batch_size, seq_length, out_channel
                    )),
                    &(batch_size, seq_length, out_channel),
                    |b, &(batch_size, seq_length, out_channel)| {
                        b.iter(|| {
                            llmrs::matmul_backward_bias_kernel4(
                                ctx,
                                m,
                                &mut dbias,
                                &dout,
                                batch_size,
                                seq_length,
                                out_channel,
                            );
                            let _ = ctx.sync();
                        })
                    },
                );

                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "llmc_{}_{}_{}",
                        batch_size, seq_length, out_channel
                    )),
                    &(batch_size, seq_length, out_channel),
                    |b, &(batch_size, seq_length, out_channel)| {
                        b.iter(|| unsafe {
                            llmc::matmul_backward_bias_kernel4_host(
                                dbias.as_devptr() as _,
                                dout.as_devptr() as _,
                                batch_size as _,
                                seq_length as _,
                                out_channel as _,
                            );
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

fn matmul_forward_benchmarks<N: gpu_host::GpuCtxSpace>(
    ctx: &gpu_host::GpuCtxGuard<N>,
    m: &gpu_host::GpuModule<N>,
    c: &mut Criterion,
) {
    let mut group = c.benchmark_group("matmul_forward");
    for batch_size in [1] {
        for seq_length_order in (10..20).step_by(4) {
            let seq_length = 1 << seq_length_order;
            for out_channel in [128, 1024] {
                let channel = out_channel / 4;
                let bias = ctx
                    .new_tensor_view(random_f32_vec(channel as usize).as_slice())
                    .expect("tensor alloc failed");
                let mut dout = ctx
                    .new_tensor_view(
                        random_f32_vec((batch_size * seq_length * out_channel) as usize).as_slice(),
                    )
                    .expect("tensor alloc failed");
                let inp = ctx
                    .new_tensor_view(
                        random_f32_vec((batch_size * seq_length * channel) as usize).as_slice(),
                    )
                    .expect("tensor alloc failed");
                let weight = ctx
                    .new_tensor_view(random_f32_vec((channel * channel) as usize).as_slice())
                    .expect("tensor alloc failed");
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "llmrs_{}_{}_{}",
                        batch_size, seq_length, out_channel
                    )),
                    &(batch_size, seq_length, out_channel),
                    |b, &(batch_size, seq_length, out_channel)| {
                        b.iter(|| {
                            llmrs::matmul_forward(
                                ctx,
                                m,
                                &mut dout,
                                &inp,
                                &weight,
                                &bias,
                                batch_size,
                                seq_length,
                                channel,
                                out_channel,
                            );
                            let _ = ctx.sync();
                        })
                    },
                );

                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "llmc_{}_{}_{}",
                        batch_size, seq_length, out_channel
                    )),
                    &(batch_size, seq_length, out_channel),
                    |b, &(batch_size, seq_length, out_channel)| {
                        b.iter(|| unsafe {
                            llmc::matmul_forward_host(
                                dout.as_devptr() as _,
                                inp.as_devptr() as _,
                                weight.as_devptr() as _,
                                bias.as_devptr() as _,
                                batch_size as _,
                                seq_length as _,
                                channel as _,
                                out_channel as _,
                            );
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

fn llmrs_benchmarks(c: &mut Criterion) {
    gpu_host::cuda_ctx(0, |ctx, m| {
        matmul_forward_benchmarks(ctx, m, c);
        matmul_back_benchmarks(ctx, m, c);
    });
}

criterion_group! {
  name = llm_rs;
  config = Criterion::default().warm_up_time(Duration::from_secs(3));
  targets = llmrs_benchmarks
}

criterion_main!(llm_rs);
