//! Native embed CLI. One process can serve many embeds over JSONL.
//!
//!   milton-embed --text hello --prefix document
//!   milton-embed --jsonl          # stdin: {text, prefix}  stdout: {vector}|{error}
//!
//! Not shipped in the npm package. WASM packaging is issue #6.

use milton::{ForwardFault, Model, Prefix};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn default_gguf() -> PathBuf {
    if let Ok(p) = env::var("MILTON_REFERENCE_GGUF").or_else(|_| env::var("MILTON_GGUF")) {
        return PathBuf::from(p);
    }
    repo_root().join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf")
}

fn parse_fault(s: &str) -> Result<ForwardFault, String> {
    match s {
        "none" | "" => Ok(ForwardFault::None),
        "layernorm" | "wrong-layernorm" => Ok(ForwardFault::WrongLayernorm),
        "pooling" | "wrong-pooling" => Ok(ForwardFault::WrongPooling),
        "dropped-prefix" | "drop-prefix" => Ok(ForwardFault::DroppedPrefix),
        other => Err(format!("unknown fault {other:?}")),
    }
}

#[derive(Debug, Deserialize)]
struct Req {
    text: String,
    #[serde(default = "default_prefix")]
    prefix: String,
    #[serde(default)]
    fault: Option<String>,
}

fn default_prefix() -> String {
    "none".into()
}

#[derive(Debug, Serialize)]
struct OkOut {
    vector: Vec<f32>,
    dims: usize,
    prefix: String,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut gguf = default_gguf();
    let mut text: Option<String> = None;
    let mut prefix = "none".to_string();
    let mut jsonl = false;
    let mut fault = ForwardFault::None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf = PathBuf::from(&args[i]);
            }
            "--text" => {
                i += 1;
                text = Some(args[i].clone());
            }
            "--prefix" => {
                i += 1;
                prefix = args[i].clone();
            }
            "--jsonl" => jsonl = true,
            "--wrong" => {
                i += 1;
                match parse_fault(&args[i]) {
                    Ok(f) => fault = f,
                    Err(e) => {
                        eprintln!("{e}");
                        return ExitCode::from(2);
                    }
                }
            }
            "-h" | "--help" => {
                eprintln!("milton-embed [--gguf PATH] [--text T --prefix document|query|none] [--jsonl] [--wrong layernorm|pooling|dropped-prefix]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown arg {other}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    if !gguf.exists() {
        eprintln!(
            "fail-closed: GGUF not found at {} — set MILTON_GGUF or run npm run harness:setup",
            gguf.display()
        );
        return ExitCode::from(2);
    }

    let model = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("fail-closed: load: {e}");
            return ExitCode::from(2);
        }
    };

    if jsonl {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    let _ = writeln!(stdout, "{}", json!({"error": e.to_string()}));
                    continue;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let req: Req = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let _ = writeln!(stdout, "{}", json!({"error": e.to_string()}));
                    let _ = stdout.flush();
                    continue;
                }
            };
            let f = match req.fault.as_deref() {
                Some(s) => match parse_fault(s) {
                    Ok(f) => f,
                    Err(_) => fault,
                },
                None => fault,
            };
            match Prefix::parse(&req.prefix) {
                Ok(p) => match model.embed_with_fault(&req.text, p, f) {
                    Ok(v) => {
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::to_string(&OkOut {
                                dims: v.len(),
                                prefix: req.prefix,
                                vector: v,
                            })
                            .unwrap()
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(stdout, "{}", json!({"error": e.to_string()}));
                    }
                },
                Err(e) => {
                    let _ = writeln!(stdout, "{}", json!({"error": e.to_string()}));
                }
            }
            let _ = stdout.flush();
        }
        return ExitCode::SUCCESS;
    }

    let Some(text) = text else {
        eprintln!("fail-closed: --text required (or --jsonl)");
        return ExitCode::from(2);
    };
    let p = match Prefix::parse(&prefix) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match model.embed_with_fault(&text, p, fault) {
        Ok(v) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&OkOut {
                    dims: v.len(),
                    prefix,
                    vector: v,
                })
                .unwrap()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("fail-closed: {e}");
            ExitCode::from(1)
        }
    }
}
