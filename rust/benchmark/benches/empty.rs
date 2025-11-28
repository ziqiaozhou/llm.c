mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

struct Empty {
    config: Config,
}

#[gpu::cuda_kernel]
fn empty() {
    if gpu::thread_id::<gpu::DimX>() >= 1 {
        return;
    }
    gpu::sync::sync_threads();
}

impl<'a> KernelRunner<'a> for Empty {
    fn new<N: gpu_host::GpuCtxSpace>(
        _ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        Some(Self { config })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BDIM: u32 = 256;
        let config = &self.config;
        let grid =
            (config.batch_size * config.seq_len * config.channel).div_ceil(BDIM as usize) as u32;
        gpu_host::gpu_config!(grid, 1, 1, @const BDIM, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        let launch_config = self.launch_config();
        empty::launch(launch_config, ctx, m).expect("kernel launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::empty_host(
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
            );
        }
    }
}

fn bench_function(c: &mut Criterion) {
    gpu_host::cuda_ctx(0, |ctx, m| {
        bench_llm_rs::<_, Empty>(c, "empty", ctx, m);
    });
}

criterion_group! {
  name = bench;
  config = Criterion::default().warm_up_time(Duration::from_secs(3));
  targets = bench_function
}

criterion_main!(bench);
