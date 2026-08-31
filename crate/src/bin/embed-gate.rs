//! Golden-vector gate: Milton native forward vs pinned llama.cpp goldens.
//! Fail-closed. Never loosens epsilon.

use milton::{compare_vectors, Model, Prefix};
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
    corpus_digest: String,
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
        &fs::read_to_string(root.join("harness/corpus/corpus.json")).expect("corpus"),
    )
    .expect("parse corpus");
    let goldens: Goldens = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/vectors.json")).expect("vectors"),
    )
    .expect("parse goldens");
    let eps: EpsilonFile = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/epsilon.json")).expect("epsilon"),
    )
    .expect("parse epsilon");
    let pin: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/pin.json")).expect("pin"),
    )
    .unwrap_or(json!({}));

    let model = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("fail-closed: load: {e}");
            return ExitCode::from(2);
        }
    };

    let mut failures = Vec::new();
    let mut max_cos_dist = 0.0f32;
    let mut sum_cos_dist = 0.0f32;
    let mut max_abs = 0.0f32;
    let n = corpus.cases.len();

    for c in &corpus.cases {
        let prefix = match Prefix::parse(&c.prefix) {
            Ok(p) => p,
            Err(e) => {
                failures.push(json!({"id": c.id, "reason": e.to_string()}));
                continue;
            }
        };
        let expected = match goldens.items.iter().find(|it| it.id == c.id) {
            Some(it) => &it.vector,
            None => {
                failures.push(json!({"id": c.id, "reason": "missing_golden"}));
                continue;
            }
        };
        match model.embed(&c.text, prefix) {
            Ok(got) => {
                let cmp = compare_vectors(&got, expected, eps.epsilon, eps.epsilon_abs);
                let cd = cmp.cos_dist.unwrap_or(1.0);
                let ab = cmp.max_abs.unwrap_or(f32::INFINITY);
                if cd > max_cos_dist {
                    max_cos_dist = cd;
                }
                if ab.is_finite() && ab > max_abs {
                    max_abs = ab;
                }
                sum_cos_dist += if cd.is_finite() { cd } else { 1.0 };
                if !cmp.pass {
                    failures.push(json!({
                        "id": c.id,
                        "prefix": c.prefix,
                        "reason": cmp.reason,
                        "cos_dist": cmp.cos_dist,
                        "max_abs": cmp.max_abs,
                        "got_head": got.iter().take(4).copied().collect::<Vec<_>>(),
                        "expected_head": expected.iter().take(4).copied().collect::<Vec<_>>(),
                    }));
                }
            }
            Err(e) => {
                failures.push(json!({"id": c.id, "reason": format!("embed_threw:{e}")}));
            }
        }
    }

    let pass = failures.is_empty();
    let receipt = json!({
        "schema": "milton.embed.receipt/1",
        "result": if pass { "pass" } else { "fail" },
        "n": n,
        "failed": failures.len(),
        "max_cos_dist": max_cos_dist,
        "mean_cos_dist": if n > 0 { sum_cos_dist / n as f32 } else { 0.0 },
        "max_abs": max_abs,
        "epsilon": eps.epsilon,
        "epsilon_abs": eps.epsilon_abs,
        "corpus_digest": goldens.corpus_digest,
        "pooling": model.meta.pooling,
        "pooling_type": model.meta.pooling_type,
        "rope_freq_base": model.meta.rope_freq_base,
        "layer_norm_epsilon": model.meta.layer_norm_epsilon,
        "block_count": model.meta.block_count,
        "embedding_length": model.meta.embedding_length,
        "gguf_sha256": pin.get("gguf_sha256"),
        "failures": failures,
    });
    println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
