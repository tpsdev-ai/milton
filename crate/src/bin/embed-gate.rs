//! Golden-vector gate: Milton Q4_K_M vs F16/F32 llama-embedding oracle.
//!
//! Pass = every gated case is within the derived quant budget of `ref_f32`
//! AND `ratio = cos_dist / quant_budget <= ratio_max` (1.5, pinned in
//! `quant-budget.json` next to `safety_factor`). The budget is llama.cpp's
//! own Q4_K_M vs F16 error (not a hand-picked epsilon). `epsilon.json` is
//! the Q4-vs-Q4 run-to-run floor and is not rewritten here. empty-none and
//! short-hello-none stay locked to that floor against `vectors.json`.
//! All 18 corpus cases are gated (#15: unicode-nfd / newlines-tabs goldens
//! are llama.cpp GGUF-forward on HF token IDs).

use milton::{compare_vectors, f32_gate_pass, Model, Prefix};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
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
    #[serde(default)]
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

#[derive(Debug, Deserialize)]
struct QuantBudget {
    safety_factor: f32,
    ratio_max: f32,
    gate_cos_dist: f32,
    max_quant_budget_cos_dist_gated: f32,
    pending_excluded: Vec<String>,
    per_case: Vec<BudgetCase>,
    ref_f32: BudgetRef,
}

#[derive(Debug, Deserialize)]
struct BudgetCase {
    id: String,
    quant_budget_cos_dist: f32,
}

#[derive(Debug, Deserialize)]
struct BudgetRef {
    gguf_sha256: String,
    gguf_file_type: String,
    llamacpp_commit: String,
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
    let f32_goldens: Goldens = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/vectors-f16.json")).expect("vectors-f16"),
    )
    .expect("parse F16 goldens");
    let q4_goldens: Goldens = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/vectors.json")).expect("vectors"),
    )
    .expect("parse Q4 goldens");
    let eps: EpsilonFile = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/epsilon.json")).expect("epsilon"),
    )
    .expect("parse epsilon");
    let budget: QuantBudget = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/quant-budget.json")).expect("quant-budget"),
    )
    .expect("parse quant-budget");
    let pin: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/pin.json")).expect("pin"),
    )
    .unwrap_or(json!({}));
    let pin_f16: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(root.join("harness/goldens/pin-f16.json")).expect("pin-f16"),
    )
    .unwrap_or(json!({}));

    if !(budget.gate_cos_dist > 0.0) {
        eprintln!("fail-closed: quant-budget gate_cos_dist must be positive");
        return ExitCode::from(2);
    }
    if !(budget.ratio_max > 0.0) {
        eprintln!("fail-closed: quant-budget ratio_max must be positive");
        return ExitCode::from(2);
    }

    let pending: HashSet<String> = budget.pending_excluded.iter().cloned().collect();
    // Landed serial-path lock: these two must stay within the Q4 run-to-run floor.
    let q4_lock: HashSet<&str> = ["empty-none", "short-hello-none"].into_iter().collect();

    let model = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("fail-closed: load: {e}");
            return ExitCode::from(2);
        }
    };

    let mut failures = Vec::new();
    let mut pending_rows = Vec::new();
    let mut gated_rows = Vec::new();
    let mut max_cos_f32 = 0.0f32;
    let mut sum_cos_f32 = 0.0f32;
    let mut max_abs_f32 = 0.0f32;
    let mut max_cos_q4 = 0.0f32;
    let mut min_ratio = f32::INFINITY;
    let mut max_ratio = 0.0f32;
    let mut n_gated = 0usize;
    let n = corpus.cases.len();

    for c in &corpus.cases {
        let prefix = match Prefix::parse(&c.prefix) {
            Ok(p) => p,
            Err(e) => {
                failures.push(json!({"id": c.id, "reason": e.to_string()}));
                continue;
            }
        };
        let ref_f32 = match f32_goldens.items.iter().find(|it| it.id == c.id) {
            Some(it) => &it.vector,
            None => {
                failures.push(json!({"id": c.id, "reason": "missing_f32_golden"}));
                continue;
            }
        };
        let q_llama = q4_goldens.items.iter().find(|it| it.id == c.id);
        let case_budget = budget
            .per_case
            .iter()
            .find(|it| it.id == c.id)
            .map(|it| it.quant_budget_cos_dist);
        match model.embed(&c.text, prefix) {
            Ok(got) => {
                let vs_f32 = compare_vectors(&got, ref_f32, budget.gate_cos_dist, f32::MAX);
                let cd = vs_f32.cos_dist.unwrap_or(1.0);
                let ab = vs_f32.max_abs.unwrap_or(f32::INFINITY);
                let vs_q4 = q_llama.map(|q| compare_vectors(&got, &q.vector, eps.epsilon, eps.epsilon_abs));
                let cd_q4 = vs_q4.as_ref().and_then(|c| c.cos_dist).unwrap_or(1.0);
                let is_pending = pending.contains(&c.id);
                let qb = case_budget.unwrap_or(0.0);
                let decision = f32_gate_pass(cd, qb, budget.gate_cos_dist, budget.ratio_max);
                if !is_pending {
                    n_gated += 1;
                    if cd > max_cos_f32 {
                        max_cos_f32 = cd;
                    }
                    if ab.is_finite() && ab > max_abs_f32 {
                        max_abs_f32 = ab;
                    }
                    sum_cos_f32 += if cd.is_finite() { cd } else { 1.0 };
                    if cd_q4.is_finite() && cd_q4 > max_cos_q4 {
                        max_cos_q4 = cd_q4;
                    }
                    if decision.ratio.is_finite() && decision.ratio < min_ratio {
                        min_ratio = decision.ratio;
                    }
                    if decision.ratio.is_finite() && decision.ratio > max_ratio {
                        max_ratio = decision.ratio;
                    }
                    gated_rows.push(json!({
                        "id": c.id,
                        "quant_budget_cos_dist": case_budget,
                        "milton_vs_f32_cos_dist": vs_f32.cos_dist,
                        "milton_vs_q_llama_cos_dist": vs_q4.as_ref().and_then(|c| c.cos_dist),
                        "ratio_vs_quant_budget": decision.ratio,
                        "within_absolute": decision.within_absolute,
                        "within_ratio": decision.within_ratio,
                    }));
                }
                let lock_fail = q4_lock.contains(c.id.as_str())
                    && vs_q4.as_ref().is_some_and(|c| !c.pass);
                let row = json!({
                    "id": c.id,
                    "prefix": c.prefix,
                    "pending_issue": if is_pending { Some(15) } else { None },
                    "quant_budget_cos_dist": case_budget,
                    "milton_vs_f32_cos_dist": vs_f32.cos_dist,
                    "milton_vs_f32_max_abs": vs_f32.max_abs,
                    "milton_vs_q_llama_cos_dist": vs_q4.as_ref().and_then(|c| c.cos_dist),
                    "milton_vs_q_llama_max_abs": vs_q4.as_ref().and_then(|c| c.max_abs),
                    "ratio_vs_quant_budget": decision.ratio,
                    "within_absolute": decision.within_absolute,
                    "within_ratio": decision.within_ratio,
                    "got_head": got.iter().take(4).copied().collect::<Vec<_>>(),
                    "f32_head": ref_f32.iter().take(4).copied().collect::<Vec<_>>(),
                });
                if is_pending {
                    pending_rows.push(row);
                } else if !decision.pass || lock_fail {
                    let mut reason = Vec::new();
                    if !decision.within_absolute {
                        reason.push(format!(
                            "cos_dist={cd} > gate_cos_dist={}",
                            budget.gate_cos_dist
                        ));
                    }
                    if !decision.within_ratio {
                        reason.push(format!(
                            "ratio={} > ratio_max={}",
                            decision.ratio, budget.ratio_max
                        ));
                    }
                    if lock_fail {
                        reason.push(format!(
                            "q4_lock:{}",
                            vs_q4.as_ref().and_then(|c| c.reason.clone()).unwrap_or_default()
                        ));
                    }
                    let mut fail = row;
                    fail["reason"] = json!(reason.join(","));
                    failures.push(fail);
                }
            }
            Err(e) => {
                failures.push(json!({"id": c.id, "reason": format!("embed_threw:{e}")}));
            }
        }
    }

    let pass = failures.is_empty();
    let receipt = json!({
        "schema": "milton.embed.receipt/2",
        "oracle": "ref_f32",
        "result": if pass { "pass" } else { "fail" },
        "n": n,
        "n_gated": n_gated,
        "failed": failures.len(),
        "max_cos_dist": max_cos_f32,
        "mean_cos_dist": if n_gated > 0 { sum_cos_f32 / n_gated as f32 } else { 0.0 },
        "max_abs": max_abs_f32,
        "max_milton_vs_q_llama_cos_dist": max_cos_q4,
        "gate_cos_dist": budget.gate_cos_dist,
        "safety_factor": budget.safety_factor,
        "ratio_max": budget.ratio_max,
        "min_ratio_gated": if min_ratio.is_finite() { min_ratio } else { 0.0 },
        "max_ratio_gated": max_ratio,
        "max_quant_budget_cos_dist_gated": budget.max_quant_budget_cos_dist_gated,
        "q4_epsilon_unchanged": {
            "epsilon": eps.epsilon,
            "epsilon_abs": eps.epsilon_abs,
        },
        "q4_lock": ["empty-none", "short-hello-none"],
        "pending_excluded": budget.pending_excluded,
        "corpus_digest": q4_goldens.corpus_digest,
        "pooling": model.meta.pooling,
        "pooling_type": model.meta.pooling_type,
        "rope_freq_base": model.meta.rope_freq_base,
        "layer_norm_epsilon": model.meta.layer_norm_epsilon,
        "block_count": model.meta.block_count,
        "embedding_length": model.meta.embedding_length,
        "gguf_sha256": pin.get("gguf_sha256"),
        "ref_f32_gguf_sha256": budget.ref_f32.gguf_sha256,
        "ref_f32_file_type": budget.ref_f32.gguf_file_type,
        "ref_f32_llamacpp_commit": budget.ref_f32.llamacpp_commit,
        "pin_f16": pin_f16,
        "failures": failures,
        "gated": gated_rows,
        "pending": pending_rows,
    });
    println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
