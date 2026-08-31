//! Harness CLI: dump token IDs or check goldens / the dropped-prefix must-fail.
//!
//! Casing and normalization must-fail controls ship as named `cargo test`s
//! (`must_fail_wrong_casing_turns_red_and_is_named`,
//! `must_fail_wrong_normalization_turns_red_and_is_named`).
//!
//! Not shipped in the npm package. Feature `cli`.

use milton::{apply_prefix, tokenize, Prefix};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate/ is inside the repo")
        .to_path_buf()
}

fn load_json(path: &std::path::Path) -> Value {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn corpus_cases() -> Vec<Value> {
    let v = load_json(&repo_root().join("harness/corpus/corpus.json"));
    v["cases"].as_array().cloned().expect("corpus.cases")
}

fn goldens() -> Value {
    load_json(&repo_root().join("harness/goldens/tokens.json"))
}

fn gold_ids(gold: &Value, id: &str) -> Vec<u32> {
    gold["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|it| it["id"].as_str() == Some(id))
        .unwrap()["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn compare_exact() -> (Value, bool) {
    let gold = goldens();
    let cases = corpus_cases();
    let mut failures = Vec::new();
    let mut n = 0u32;
    for case in &cases {
        let id = case["id"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let kind = case["prefix"].as_str().unwrap();
        let prefix = Prefix::parse(kind).expect("corpus prefix");
        let got = tokenize(text, prefix);
        let expected = gold_ids(&gold, id);
        n += 1;
        if got != expected {
            failures.push(json!({
                "id": id,
                "prefix": kind,
                "expected": expected,
                "got": got,
                "expected_len": expected.len(),
                "got_len": got.len(),
            }));
        }
    }
    let pass = failures.is_empty();
    let receipt = json!({
        "schema": "milton.tokenizer-receipt/1",
        "n": n,
        "matched": n as usize - failures.len(),
        "failed": failures.len(),
        "result": if pass { "pass" } else { "fail" },
        "match": "exact-token-id",
        "corpus_digest": gold["corpus_digest"],
        "tokens_digest": gold["tokens_digest"],
        "failures": failures,
    });
    (receipt, pass)
}

fn must_fail_dropped_prefix() -> (Value, bool) {
    let gold = goldens();
    let cases = corpus_cases();
    let mut named = Vec::new();
    for case in &cases {
        let id = case["id"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let expected = gold_ids(&gold, id);
        let got = tokenize(text, Prefix::None);
        if got != expected {
            named.push(id.to_string());
        }
    }
    let red = named.iter().any(|id| id == "short-hello-document")
        && named.iter().any(|id| id == "short-hello-query")
        && !named.iter().any(|id| id == "short-hello-none");
    let receipt = json!({
        "control": "must_fail_dropped_prefix",
        "result": if red { "red-as-required" } else { "FAILED-TO-CATCH" },
        "named": named,
    });
    (receipt, red)
}

fn usage() {
    eprintln!(
        "milton-tokenize — nomic-embed-text-v1.5 tokenizer CLI (harness)\n\
         \n\
         Usage:\n\
           milton-tokenize --prefix <document|query|none> <text>\n\
           milton-tokenize --check-goldens\n\
           milton-tokenize --must-fail dropped-prefix\n\
         \n\
         Casing / normalization must-fail controls ship as named cargo tests:\n\
           cargo test --manifest-path crate/Cargo.toml must_fail\n"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        usage();
        return ExitCode::FAILURE;
    }
    match args[0].as_str() {
        "--check-goldens" => {
            let (receipt, pass) = compare_exact();
            println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            if pass {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        "--must-fail" => {
            let which = args.get(1).map(String::as_str).unwrap_or("");
            if which != "dropped-prefix" {
                usage();
                return ExitCode::FAILURE;
            }
            let (receipt, ok) = must_fail_dropped_prefix();
            println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        "--prefix" => {
            if args.len() < 3 {
                usage();
                return ExitCode::FAILURE;
            }
            let prefix = match Prefix::parse(&args[1]) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            let text = args[2..].join(" ");
            let ids = tokenize(&text, prefix);
            let prefixed = apply_prefix(&text, prefix);
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "prefix": prefix.as_kind(),
                    "prefixed": prefixed,
                    "n": ids.len(),
                    "ids": ids,
                }))
                .unwrap()
            );
            ExitCode::SUCCESS
        }
        _ => {
            usage();
            ExitCode::FAILURE
        }
    }
}
