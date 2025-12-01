CRITERION_HOME=results/criterion-no-check DISABLE_GPU_BOUND_CHECK=true cargo bench;
cargo r --bin benchmark -- ./benchmark/results/criterion-no-check/
cargo clean
CRITERION_HOME=results/criterion DISABLE_GPU_BOUND_CHECK=false cargo bench
cargo r --bin benchmark -- ./benchmark/results/criterion/