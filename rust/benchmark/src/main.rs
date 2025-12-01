/// CRITERION_HOME=results/criterion-no-check DISABLE_GPU_BOUND_CHECK=true cargo bench;
/// cargo r --bin benchmark -- ./benchmark/results/criterion-no-check/ nocheck
/// cargo clean
/// CRITERION_HOME=results/criterion DISABLE_GPU_BOUND_CHECK=false cargo bench
/// cargo r --bin benchmark -- ./benchmark/results/criterion/ check
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
    let criterion_name = args().nth(2).unwrap_or_else(|| "check".to_string());
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
                    let rs_or_c = config_parts[0].to_string();
                    let thread_count = config_parts[8..12]
                        .iter()
                        .map(|s| s.parse::<usize>().unwrap())
                        .product::<usize>();
                    let t_o_v = format!("{}_{}_{}_{}", t, o, v, thread_count);
                    if !sub_results.contains_key(&t_o_v) {
                        sub_results.insert(t_o_v.clone(), HashMap::<String, f64>::new());
                    }
                    if !sub_results.get(&t_o_v).unwrap().contains_key(&rs_or_c) {
                        sub_results.get_mut(&t_o_v).unwrap().insert(rs_or_c.clone(), mean_ms);
                    } else {
                        panic!("Duplicate entry for {} {}", t_o_v, rs_or_c);
                    }
                }
            }
            // Clean up benchmark name
            let bench_name = bench_name
                .replace("_kernel", "")
                .replace("_", "-")
                .replace("backward", "bwd")
                .replace("forward", "fwd")
                .replace("back", "bwd")
                .replace("_bench", "");
            results.insert(bench_name.clone(), sub_results);
        }
    }

    let mut file = BufWriter::new(
        std::fs::File::create(format!("{}.csv", criterion_name))
            .expect("failed to create result.csv"),
    );
    writeln!(file, "Benchmark\tT\tO\tV\tThreads\trs\tc\trs/c\tempty-rs\tempty-c\trs-norm/c-norm")
        .unwrap();
    println!(
        "{:<20} {:<14} {:<14} {:<14} {:<14} {:<14} {:<14} {:<14}",
        "Benchmark", "tT_O_V_Th", "rs", "c", "rs/c", "empty-rs", "empty-c", "rs-norm/c-norm"
    );
    let mut keys: Vec<_> = results.keys().into_iter().map(|k| k.to_string()).collect();
    keys.sort(); // alphabetical order
    let mut latex_data: HashMap<String, Vec<f64>> = HashMap::new();
    for bench_name in &keys {
        let sub_results = &results[bench_name];
        let mut sub_keys = sub_results.keys().collect::<Vec<_>>();
        sub_keys.sort_by(|a, b| {
            let len_cmp = a.len().cmp(&b.len());
            if len_cmp == std::cmp::Ordering::Equal { a.cmp(b) } else { len_cmp }
        });

        for t_o in sub_keys {
            let times = &sub_results[t_o];
            let Some(&rs_time) = times.get("rs") else {
                continue;
            };
            let Some(&c_time) = times.get("c") else {
                continue;
            };
            let Some(&empty_c_time) = times.get("emptyc") else {
                continue;
            };
            let Some(&empty_rs_time) = times.get("emptyrs") else {
                continue;
            };
            let empty_time_diff = empty_rs_time - empty_c_time;
            // get usize value of t, o, v, threads
            let t_o_parts: Vec<usize> =
                t_o.split("_").map(|s| s.parse::<usize>().unwrap()).collect();
            let (t, o, v, threads) = (t_o_parts[0], t_o_parts[1], t_o_parts[2], t_o_parts[3]);
            if c_time > 0.0 {
                let rs_to_c = rs_time / c_time;
                let rs_time_norm = rs_time - empty_time_diff;
                let rs_norm_to_c = rs_time_norm / c_time;
                println!(
                    "{:<20} {:<14} {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:>14.2}",
                    &bench_name[..std::cmp::min(20, bench_name.len())],
                    t_o,
                    rs_time,
                    c_time,
                    rs_to_c,
                    empty_rs_time,
                    empty_c_time,
                    rs_norm_to_c
                );
                writeln!(
                    file,
                    "{bench}\t{t}\t{o}\t{v}\t{threads}\t{rs:.2}\t{c:.2}\t{rs_c:.2}\t{empty_rs_time:.2}\t{empty_c_time:.2}\t{rs_norm_c:.2}",
                    bench = &bench_name[..std::cmp::min(18, bench_name.len())],
                    t = t,
                    o = o,
                    v = v,
                    threads = threads,
                    rs = rs_time,
                    c = c_time,
                    rs_c = rs_to_c,
                    empty_rs_time = empty_rs_time,
                    empty_c_time = empty_c_time,
                    rs_norm_c = rs_norm_to_c
                ).unwrap();
                latex_data.entry(bench_name.to_string()).or_default().push(rs_norm_to_c);
            }
        }
    }

    drop(file);

    // Generate LaTeX data
    let mut latexfile = BufWriter::new(
        std::fs::File::create(format!("{}.tex", criterion_name))
            .expect("failed to create result.tex"),
    );
    let max_runs = latex_data.values().map(|v| v.len()).max().unwrap_or(0);
    let coords = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",");

    writeln!(latexfile, "\\newcommand{{\\xsymbols}}{{ symbolic x coords={{ {} }}}}", coords)
        .unwrap();

    for run_index in 0..max_runs {
        let seq = [1024, 16384, 1048576][run_index];
        let offset = run_index as i32 - (max_runs as i32 - 1) / 2;
        let data = latex_data
            .iter()
            .map(|(bench, values)| {
                if run_index >= values.len() {
                    "".to_string()
                } else {
                    format!("({},{})", bench, values[run_index])
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(latexfile, "\\pgfkeyssetvalue{{/ratio/{criterion_name}/{seq}}}{{\n{data}\n}}",)
            .unwrap();
    }
}
