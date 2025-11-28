/// cargo r --bin benchmark -- ./results/criterion-no-check/
/// cargo r --bin benchmark -- ./results/criterion
use std::collections::HashMap;
use std::env::args;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct BenchmarkJson {
    mean: Mean,
}

#[derive(Debug, Deserialize)]
struct Mean {
    point_estimate: f64, // in nanoseconds
}

const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
fn main() {
    let mut results = HashMap::new();
    let cargo_dir = CARGO_MANIFEST_DIR;
    let criterion_dir =
        args().nth(1).unwrap_or_else(|| format!("{}/../target/criterion", cargo_dir));
    let criterion_dir = Path::new(&criterion_dir);
    if !criterion_dir.exists() {
        eprintln!("Criterion directory not found: {:?}", criterion_dir);
        return;
    }

    // Iterate over all benchmark folders
    for entry in fs::read_dir(criterion_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            let bench_name = entry.file_name().into_string().unwrap().trim().to_string();

            let mut sub_results = HashMap::new();
            let bench_dir = entry.path();
            for entry in fs::read_dir(&bench_dir).unwrap() {
                let entry = entry.unwrap();
                if !entry.file_type().unwrap().is_dir() {
                    continue;
                }
                let config = entry.file_name().into_string().unwrap();
                let new_json_path = entry.path().join("new/estimates.json");
                if new_json_path.exists() {
                    let content = fs::read_to_string(new_json_path).unwrap();
                    let json: BenchmarkJson = serde_json::from_str(&content).unwrap();
                    // Convert nanoseconds
                    let mean_ms = json.mean.point_estimate;
                    let config_parts: Vec<&str> = config.split("_").collect();
                    assert!(config_parts.len() >= 4);
                    let (t, o, v) = (config_parts[1], config_parts[2], config_parts[3]);
                    let rs_or_c = if config_parts[0] == "rs" { 1 } else { 0 };
                    let t_o_v = format!("{}_{}_{}", t, o, v);
                    if !sub_results.contains_key(&t_o_v) {
                        sub_results.insert(t_o_v.clone(), [0.0, 0.0]);
                    }
                    sub_results.get_mut(&t_o_v).unwrap()[rs_or_c] = mean_ms;
                }
            }
            results.insert(bench_name.clone(), sub_results);
        }
    }

    let mut file =
        BufWriter::new(std::fs::File::create("results.csv").expect("failed to create result.csv"));
    writeln!(file, "Benchmark\tT_O\trs\tc\trs/c\tdiff\trs-norm\trs-norm/c-norm").unwrap();
    println!(
        "{:<25} {:<12} {:<14} {:<14} {:<14} {:<14} {:<14} {:<14}",
        "Benchmark", "T_O", "rs", "c", "rs/c", "diff", "rs-norm", "rs-norm/c-norm"
    );
    let empty_time_diff = results["empty"]["1024_32_1024"][1] - results["empty"]["1024_32_1024"][0];
    for (bench_name, sub_results) in &results {
        for (t_o, times) in sub_results {
            let rs_time = times[1];
            let c_time = times[0];
            if c_time > 0.0 {
                let rs_to_c = rs_time / c_time;
                let rs_time_norm = rs_time - empty_time_diff;
                let rs_norm_to_c = rs_time_norm / c_time;
                println!(
                    "{:<25} {:<12} {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:>14.2}",
                    &bench_name[..std::cmp::min(20, bench_name.len())],
                    t_o,
                    rs_time,
                    c_time,
                    rs_to_c,
                    rs_time - c_time,
                    rs_time_norm,
                    rs_norm_to_c
                );
                writeln!(
                    file,
                    "{bench}\t{t_o}\t{rs:.2}\t{c:.2}\t{rs_c:.2}\t{diff:.2}\t{rs_norm:.2}\t{rs_norm_c:.2}",
                    bench = &bench_name[..std::cmp::min(20, bench_name.len())],
                    t_o = t_o,
                    rs = rs_time,
                    c = c_time,
                    rs_c = rs_to_c,
                    diff = rs_time - c_time,
                    rs_norm = rs_time_norm,
                    rs_norm_c = rs_norm_to_c
                ).unwrap();
            }
        }
    }
}
