//! Must-fail control: a deliberately-wrong dequant (wrong block scale, wrong
//! type) must turn the run RED and name the failure. A gate that has only
//! ever passed is a decoration.
//!
//! Usage: dequant-must-fail [--goldens PATH] [--epsilon PATH]
//! Exit 0 = wrong dequants were caught (the control itself PASSES).
//! Exit 1 = a wrong dequant slipped through (control FAILED — do not trust the gate).

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use milton::{
    compare_vectors, dequantize_wrong_block_scale, dequantize_wrong_type, TensorType,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Goldens {
    schema: String,
    kernel_blocks: Vec<KernelBlock>,
}

#[derive(Debug, Deserialize)]
struct KernelBlock {
    id: String,
    #[serde(rename = "type")]
    type_name: String,
    n_elements: usize,
    wire_hex: String,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EpsilonFile {
    epsilon: f32,
    epsilon_abs: f32,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn hex_decode(s: &str) -> Vec<u8> {
    let s = s.trim();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn type_from_name(name: &str) -> TensorType {
    match name {
        "F32" => TensorType::F32,
        "F16" => TensorType::F16,
        "Q8_0" => TensorType::Q8_0,
        "Q4_K" | "Q4K" => TensorType::Q4K,
        "Q5_K" | "Q5K" => TensorType::Q5K,
        "Q6_K" | "Q6K" => TensorType::Q6K,
        other => panic!("unverified type {other}"),
    }
}

fn main() -> ExitCode {
    let root = repo_root();
    let mut goldens_path = root.join("harness/goldens/dequant.json");
    let mut epsilon_path = root.join("harness/goldens/dequant-epsilon.json");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--goldens" => goldens_path = PathBuf::from(args.next().expect("path")),
            "--epsilon" => epsilon_path = PathBuf::from(args.next().expect("path")),
            other => panic!("unknown arg {other}"),
        }
    }
    let goldens: Goldens =
        serde_json::from_str(&fs::read_to_string(&goldens_path).expect("read goldens")).unwrap();
    assert_eq!(goldens.schema, "milton.dequant/1");
    let eps: EpsilonFile =
        serde_json::from_str(&fs::read_to_string(&epsilon_path).expect("read epsilon")).unwrap();

    let mut named = Vec::new();
    let mut slipped = Vec::new();

    for kb in &goldens.kernel_blocks {
        let ty = type_from_name(&kb.type_name);
        let wire = hex_decode(&kb.wire_hex);

        // Control 1: wrong block scale
        match dequantize_wrong_block_scale(ty, &wire, kb.n_elements, &kb.id) {
            Ok(got) => {
                let cmp = compare_vectors(&got, &kb.values, eps.epsilon, eps.epsilon_abs);
                if cmp.pass {
                    slipped.push(format!(
                        "wrong-block-scale slipped GREEN on {} ({})",
                        kb.id, kb.type_name
                    ));
                } else {
                    named.push(format!(
                        "wrong-block-scale RED {} reason={}",
                        kb.id,
                        cmp.reason.unwrap_or_default()
                    ));
                }
            }
            Err(e) => named.push(format!("wrong-block-scale RED {} threw={e}", kb.id)),
        }

        // Control 2: wrong type
        match dequantize_wrong_type(ty, &wire, kb.n_elements, &kb.id) {
            Ok(got) => {
                let cmp = if got.len() == kb.values.len() {
                    compare_vectors(&got, &kb.values, eps.epsilon, eps.epsilon_abs)
                } else {
                    milton::Compare {
                        pass: false,
                        reason: Some(format!("dim_mismatch:{}->{}", kb.values.len(), got.len())),
                        cosine: None,
                        cos_dist: None,
                        max_abs: None,
                    }
                };
                if cmp.pass {
                    slipped.push(format!(
                        "wrong-type slipped GREEN on {} ({})",
                        kb.id, kb.type_name
                    ));
                } else {
                    named.push(format!(
                        "wrong-type RED {} reason={}",
                        kb.id,
                        cmp.reason.unwrap_or_default()
                    ));
                }
            }
            Err(e) => named.push(format!("wrong-type RED {} threw={e}", kb.id)),
        }
    }

    let receipt = serde_json::json!({
        "schema": "milton.dequant.mustfail/1",
        "result": if slipped.is_empty() && !named.is_empty() { "pass" } else { "fail" },
        "named": named,
        "slipped": slipped,
        "n_controls": named.len() + slipped.len(),
    });
    println!("{}", serde_json::to_string_pretty(&receipt).unwrap());

    if slipped.is_empty() && !named.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
