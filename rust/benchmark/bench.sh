export USE_LLVM=1
rm ../../libtrain_gpt2fp32.a
rm ../../libtrain_gpt2fp32.o
VER=1.87
CRITERION_HOME=results/criterion-no-check-llvm-$VER DISABLE_GPU_BOUND_CHECK=true cargo bench;
cargo r --bin benchmark -- ./benchmark/results/criterion-no-check-llvm-$VER/
cargo clean
CRITERION_HOME=results/criterion-llvm-$VER DISABLE_GPU_BOUND_CHECK=false cargo bench
cargo r --bin benchmark -- ./benchmark/results/criterion-llvm-$VER/

rm ../../libtrain_gpt2fp32.so
rm ../../libtrain_gpt2fp32.o
export USE_LLVM=0
CRITERION_HOME=results/criterion-no-check-$VER DISABLE_GPU_BOUND_CHECK=true cargo bench;
cargo r --bin benchmark -- ./benchmark/results/criterion-no-check-$VER/
cargo clean
CRITERION_HOME=results/criterion-$VER DISABLE_GPU_BOUND_CHECK=false cargo bench
cargo r --bin benchmark -- ./benchmark/results/criterion-$VER/
