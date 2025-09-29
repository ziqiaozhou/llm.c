#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};

use clap::Parser;
use cudarc::cublas::sys as cublas_sys;
use gpu_host::{GpuCtxGuard, GpuCtxSpace, GpuModule, cuda_ctx};

mod dataloader;
mod model;
mod tokenizer;

use dataloader::DataLoader;
use model::GPT2;

macro_rules! top_path {
    ($p: literal) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../", $p)
    };
}
#[derive(Parser, Debug)]
struct Args {
    /// Name of the user
    #[arg(default_value = top_path!("dev/data/tinyshakespeare/tiny_shakespeare_train.bin"))]
    train_data_pattern: PathBuf,
    #[arg(default_value = top_path!("dev/data/tinyshakespeare/tiny_shakespeare_val.bin"))]
    val_data_pattern: PathBuf,
    #[arg(default_value = top_path!("gpt2_124M.bin"), value_parser)]
    model_path: PathBuf,
    #[arg(default_value = top_path!("gpt2_tokenizer.bin"), value_parser)]
    tokenizer_path: PathBuf,
    #[arg(default_value = "llm_rs.log")]
    output_log_file: String,
    #[arg(default_value_t = 4)]
    batch_size: i32,
    #[arg(default_value_t = 1024)]
    seq_length: i32,
    #[arg(default_value_t = 3e-4)]
    learning_rate: f32,
    #[arg(default_value_t = 20)]
    val_loss_every: usize,
    #[arg(default_value_t = 20)]
    val_max_steps: usize,
    #[arg(default_value_t = 1)]
    sample_every: usize,
    #[arg(default_value_t = 1)]
    gen_t: usize,
}

fn main() {
    let args = Args::parse();
    println!("params: {args:?}");
    eprintln!("WARNING: the code is currently broken, please ignore this run");
    cuda_ctx(0, |ctx, m| {
        llm_rs_run(ctx, m, &args);
    });
}

#[derive(Debug)]
pub struct UnsafeCudaContext {
    pub(crate) cu_device: cudarc::driver::sys::CUdevice,
    pub(crate) cu_ctx: *mut std::ffi::c_void,
    pub(crate) ordinal: usize,
    pub(crate) has_async_alloc: bool,
    pub(crate) num_streams: AtomicUsize,
    pub(crate) event_tracking: AtomicBool,
    pub(crate) error_state: AtomicU32,
}

fn llm_rs_run<'ctx, 'a, NS: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, 'a, NS>,
    m: &GpuModule<NS>,
    args: &Args,
) {
    let mut handle = MaybeUninit::uninit();
    let cublas_handle = unsafe {
        cublas_sys::cublasCreate_v2(handle.as_mut_ptr());
        handle.assume_init()
    };
    let (major, minor) = ctx.get_compute_capability();
    let enable_tf32 = major >= 8;
    unsafe {
        cublas_sys::cublasSetMathMode(
            cublas_handle,
            if enable_tf32 {
                cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
            } else {
                cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH
            },
        );
    }
    println!("cublas handle: {:?}, TF32: {}", cublas_handle, enable_tf32);
    let model = GPT2::new(ctx, m, &args.model_path).unwrap_or_else(|_| {
        panic!("Error initializing model from checkpoint");
    });
    println!("| max_sequence_length T | %{} |\n", model.config.max_seq_len);
    println!("| vocab_size V          | %{} |\n", model.config.vocab_size);
    println!("| padded_vocab_size Vp  | %{} |\n", model.config.padded_vocab_size);
    println!("| num_layers L          | %{} |\n", model.config.num_layers);
    println!("| num_heads NH          | %{} |\n", model.config.num_heads);
    println!("| channels C            | %{} |\n", model.config.channels);
    println!("| num_parameters        | %{} |\n", model.num_parameters);
    println!("+-----------------------+----------------------------------------------------+\n");
    let train_loader = DataLoader::new(
        &args.train_data_pattern,
        args.batch_size as usize,
        args.seq_length as usize,
    );
    let mut val_loader =
        DataLoader::new(&args.val_data_pattern, args.batch_size as usize, args.seq_length as usize);
    let val_num_batches = if val_loader.num_batches > args.val_max_steps {
        args.val_max_steps
    } else {
        val_loader.num_batches
    };
    println!("| train_num_batches     | %{} |\n", train_loader.num_batches);
    println!("val_num_batches: {}", val_num_batches);
    println!("+-----------------------+----------------------------------------------------+\n");
    println!(
        "allocated {} MiB for model parameters",
        (model.num_parameters * std::mem::size_of::<f32>()) / (1024 * 1024)
    );

    let tokenizer = tokenizer::Tokenizer::new(&args.tokenizer_path);

    // some memory for generating samples from the model
    let rng_state: u64 = 1337;
    let mut gen_tokens = vec![0; args.batch_size as usize * args.seq_length as usize];
    let cpu_logits = vec![0.0f32; model.config.vocab_size];

    // train
    for step in 0..=train_loader.num_batches {
        let last_step = step == train_loader.num_batches;

        // once in a while estimate the validation loss
        if step % args.val_loss_every == 0 || last_step {
            let mut val_loss = 0.0f32;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch();
                /*model.forward(
                    val_loader.inputs,
                    val_loader.targets,
                    args.batch_size,
                    args.seq_length,
                );*/
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("val loss {val_loss}");
            //logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % args.sample_every == 0) || last_step {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            gen_tokens.iter_mut().for_each(|t| *t = 50256); // GPT2_EOT
            // now sample from the model autoregressively
            println!("generating:\n---");
            for t in 1..args.gen_t {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                //model.forward(&gen_tokens, None, args.batch_size, args.seq_length);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                /*let logits = model
                    .acts
                    .logits
                    .ptr
                    .wrapping_add((t - 1) * model.config.padded_vocab_size);
                let logits = unsafe {
                    std::slice::from_raw_parts(logits, model.config.padded_vocab_size as usize)
                };*/
            }
        }
    }
}
