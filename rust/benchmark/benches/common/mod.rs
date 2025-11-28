use criterion::Criterion;
use rand::Rng;

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn rand_f32_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<f32>()).collect()
}

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn rand_float4_vec(n: usize) -> Vec<gpu::Float4> {
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
pub fn rand_i32_vec(n: usize) -> Vec<i32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<i32>()).collect()
}

pub trait KernelRunner<'a>: Sized {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self>;

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    );

    fn c_fn(&mut self);
}

#[allow(dead_code)]
pub struct Config {
    pub batch_size: usize,
    pub seq_len: usize,
    pub channel: usize,
    pub out_channel: usize,
    pub vocab_size: usize,
    pub padded_vocab_size: usize,
    pub head_size: usize,
    pub num_heads: usize,
}

impl Config {
    fn to_str(&self) -> String {
        format!(
            "{}_{}_{}_{}_{}_{}_scvpho",
            self.seq_len,
            self.channel,
            self.vocab_size,
            self.padded_vocab_size,
            self.head_size,
            self.out_channel
        )
    }
}

pub fn bench_llm_rs<'a, N: gpu_host::GpuCtxSpace, B: KernelRunner<'a>>(
    c: &mut Criterion,
    name: &str,
    ctx: &'a gpu_host::GpuCtxGuard<N>,
    m: &'a gpu_host::GpuModule<N>,
) {
    let mut group = c.benchmark_group(name);
    let batch_size = 1;
    for seq_length_order in (10..20).step_by(4) {
        let seq_len = 1 << seq_length_order;
        for out_channel in [128, 1024] {
            for vocab_size in [1024, 4096] {
                let num_heads = 8;
                let channel = out_channel / 4;
                let head_size = channel / num_heads;
                let config = Config {
                    batch_size,
                    seq_len,
                    vocab_size,
                    padded_vocab_size: vocab_size,
                    head_size,
                    channel,
                    out_channel,
                    num_heads,
                };
                let config_str = config.to_str();
                let Some(mut mybench) = B::new(ctx, config) else {
                    continue;
                };
                group.bench_function(format!("rs_{}", config_str).as_str(), |b| {
                    b.iter(|| {
                        mybench.rs_fn(ctx, m);
                        let _ = ctx.sync();
                    })
                });
                group.bench_function(format!("c_{}", config_str).as_str(), |b| {
                    b.iter(|| {
                        mybench.c_fn();
                        let _ = ctx.sync();
                    })
                });
            }
        }
    }
    group.finish();
}
