//! Harness-only stage profile CLI. Requires `--features profile`.
//!
//!   milton-profile --text hello --prefix document
//!   milton-profile --jsonl   # stdin: {text, prefix}  stdout: profile JSON
//!
//! Not shipped. Separate target-dir so default milton-embed is untouched.

use milton::{Model, Prefix};
use serde::Deserialize;
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

fn next_arg<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("fail-closed: {flag} requires a value"))
}

#[derive(Debug, Deserialize)]
struct Req {
    text: String,
    #[serde(default = "default_prefix")]
    prefix: String,
}

fn default_prefix() -> String {
    "none".into()
}

fn emit(model: &Model, text: &str, prefix: &str) -> Result<String, String> {
    let p = Prefix::parse(prefix).map_err(|e| e.to_string())?;
    let (v, snap) = model.embed_profiled(text, p).map_err(|e| e.to_string())?;
    serde_json::to_string(&json!({
        "vector": v,
        "dims": v.len(),
        "prefix": prefix,
        "profile": snap,
    }))
    .map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut gguf = default_gguf();
    let mut text: Option<String> = None;
    let mut prefix = "none".to_string();
    let mut jsonl = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => match next_arg(&args, &mut i, "--gguf") {
                Ok(v) => gguf = PathBuf::from(v),
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            },
            "--text" => match next_arg(&args, &mut i, "--text") {
                Ok(v) => text = Some(v.to_string()),
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            },
            "--prefix" => match next_arg(&args, &mut i, "--prefix") {
                Ok(v) => prefix = v.to_string(),
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            },
            "--jsonl" => jsonl = true,
            "-h" | "--help" => {
                eprintln!(
                    "milton-profile [--gguf PATH] [--text T --prefix document|query|none] [--jsonl]"
                );
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
            match emit(&model, &req.text, &req.prefix) {
                Ok(s) => {
                    let _ = writeln!(stdout, "{s}");
                }
                Err(e) => {
                    let _ = writeln!(stdout, "{}", json!({"error": e}));
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
    match emit(&model, &text, &prefix) {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("fail-closed: {e}");
            ExitCode::from(1)
        }
    }
}
