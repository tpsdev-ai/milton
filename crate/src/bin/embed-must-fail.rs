//! Must-fail control: a deliberately-broken forward must turn RED and be named.
//!
//! Faults: wrong layernorm, wrong pooling (CLS vs mean), dropped prefix.

use milton::{compare_vectors, ForwardFault, Model, Prefix};
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn default_gguf() -> PathBuf {
    env::var("MILTON_REFERENCE_GGUF")
        .or_else(|_| env::var("MILTON_GGUF"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf"))
}

#[derive(Debug, Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    text: String,
    prefix: String,
}

#[derive(Debug, Deserialize)]
struct Goldens {
    items: Vec<GoldenItem>,
}

#[derive(Debug, Deserialize)]
struct GoldenItem {
    id: String,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EpsilonFile {
    epsilon: f32,
    epsilon_abs: f32,
}

fn main() -> ExitCode {
    let root = repo_root();
    let gguf = default_gguf();
    if !gguf.exists() {
        eprintln!("fail-closed: GGUF missing at {}", gguf.display());
        return ExitCode::from(2);
    }
    let corpus: Corpus = serde_json::from_str(
        &fs::read_to_string(root.join("harness/corpus/corpus.json")).unwrap(),
    )
    .unwrap();
    let goldens: Goldens = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/vectors.json")).unwrap(),
    )
    .unwrap();
    let eps: EpsilonFile = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/epsilon.json")).unwrap(),
    )
    .unwrap();

    let model = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("fail-closed: load: {e}");
            return ExitCode::from(2);
        }
    };

    let faults = [
        (ForwardFault::WrongLayernorm, "wrong-layernorm"),
        (ForwardFault::WrongPooling, "wrong-pooling"),
        (ForwardFault::DroppedPrefix, "dropped-prefix"),
    ];

    let mut named = Vec::new();
    let mut slipped = Vec::new();

    for (fault, label) in faults {
        let mut red = Vec::new();
        for c in &corpus.cases {
            let prefix = Prefix::parse(&c.prefix).unwrap();
            let expected = goldens
                .items
                .iter()
                .find(|it| it.id == c.id)
                .map(|it| it.vector.as_slice());
            let Some(expected) = expected else { continue };
            match model.embed_with_fault(&c.text, prefix, fault) {
                Ok(got) => {
                    let cmp = compare_vectors(&got, expected, eps.epsilon, eps.epsilon_abs);
                    if !cmp.pass {
                        red.push(format!(
                            "{} reason={}",
                            c.id,
                            cmp.reason.unwrap_or_default()
                        ));
                    }
                }
                Err(e) => red.push(format!("{} threw={e}", c.id)),
            }
        }
        if red.is_empty() {
            slipped.push(label.to_string());
        } else {
            named.push(format!("{label} RED {} cases ({})", red.len(), red.join("; ")));
        }
    }

    let pass = slipped.is_empty();
    let receipt = json!({
        "schema": "milton.embed.must-fail/1",
        "result": if pass { "pass" } else { "fail" },
        "n_controls": faults.len(),
        "slipped": slipped,
        "named": named,
    });
    println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
