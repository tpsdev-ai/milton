//! Compare Milton dequant against the pinned llama.cpp fixture.
//!
//! Fail-closed: any tensor over tolerance fails the run and is named.
//! Usage: dequant-gate [--gguf PATH] [--goldens PATH] [--epsilon PATH]

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use milton::{
    compare_vectors, dequantize, GgufFile, TensorType,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Goldens {
    schema: String,
    gguf_sha256: String,
    tensors: Vec<GoldenTensor>,
    #[serde(default)]
    kernel_blocks: Vec<KernelBlock>,
}

#[derive(Debug, Deserialize)]
struct GoldenTensor {
    name: String,
    #[serde(rename = "type")]
    type_name: String,
    n_elements: usize,
    head: Vec<f32>,
    tail: Vec<f32>,
    #[serde(default)]
    mid: Vec<f32>,
    mid_offset: Option<usize>,
    sha256_f32_le: String,
    #[serde(default)]
    wire_hex: Option<String>,
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

#[derive(Debug, Serialize)]
struct Receipt {
    schema: &'static str,
    result: &'static str,
    n: usize,
    failed: usize,
    max_abs: f32,
    mean_abs: f32,
    max_cos_dist: f32,
    mean_cos_dist: f32,
    epsilon: f32,
    epsilon_abs: f32,
    gguf_sha256: String,
    failures: Vec<Failure>,
}

#[derive(Debug, Serialize)]
struct Failure {
    id: String,
    reason: String,
    max_abs: Option<f32>,
    cos_dist: Option<f32>,
}

fn repo_root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent().unwrap().to_path_buf()
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err("odd hex length".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn type_from_name(name: &str) -> Result<TensorType, String> {
    match name {
        "F32" => Ok(TensorType::F32),
        "F16" => Ok(TensorType::F16),
        "Q8_0" => Ok(TensorType::Q8_0),
        "Q4_K" | "Q4K" => Ok(TensorType::Q4K),
        "Q5_K" | "Q5K" => Ok(TensorType::Q5K),
        "Q6_K" | "Q6K" => Ok(TensorType::Q6K),
        other => Err(format!("unverified type {other}")),
    }
}

fn sha256_f32_le(vals: &[f32]) -> String {
    sha256_bytes(&vals.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}

fn sha256_bytes(data: &[u8]) -> String {
    sha256::hash(data)
}

mod sha256 {
    /// Compact SHA-256 (FIPS 180-4). Harness already pins the GGUF by sha256;
    /// fixtures do the same for dequantized f32 bytes.
    pub fn hash(data: &[u8]) -> String {
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
            0x1f83d9ab, 0x5be0cd19,
        ];
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let bit_len = (data.len() as u64).saturating_mul(8);
        let mut buf = data.to_vec();
        buf.push(0x80);
        while (buf.len() % 64) != 56 {
            buf.push(0);
        }
        buf.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in buf.chunks_exact(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut a = h;
            for i in 0..64 {
                let s1 = a[4].rotate_right(6) ^ a[4].rotate_right(11) ^ a[4].rotate_right(25);
                let ch = (a[4] & a[5]) ^ ((!a[4]) & a[6]);
                let t1 = a[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a[0].rotate_right(2) ^ a[0].rotate_right(13) ^ a[0].rotate_right(22);
                let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
                let t2 = s0.wrapping_add(maj);
                a[7] = a[6];
                a[6] = a[5];
                a[5] = a[4];
                a[4] = a[3].wrapping_add(t1);
                a[3] = a[2];
                a[2] = a[1];
                a[1] = a[0];
                a[0] = t1.wrapping_add(t2);
            }
            for i in 0..8 {
                h[i] = h[i].wrapping_add(a[i]);
            }
        }
        h.iter().map(|x| format!("{x:08x}")).collect()
    }
}

fn fail(failures: &mut Vec<Failure>, n: &mut usize, id: String, reason: String) {
    *n += 1;
    failures.push(Failure {
        id,
        reason,
        max_abs: None,
        cos_dist: None,
    });
}

fn record(
    failures: &mut Vec<Failure>,
    n: &mut usize,
    max_abs: &mut f32,
    sum_abs: &mut f32,
    max_cos: &mut f32,
    sum_cos: &mut f32,
    id: String,
    cmp: milton::Compare,
) {
    *n += 1;
    if let Some(a) = cmp.max_abs {
        if a > *max_abs {
            *max_abs = a;
        }
        *sum_abs += a;
    }
    if let Some(c) = cmp.cos_dist {
        if c > *max_cos {
            *max_cos = c;
        }
        *sum_cos += c;
    }
    if !cmp.pass {
        failures.push(Failure {
            id,
            reason: cmp.reason.unwrap_or_else(|| "fail".into()),
            max_abs: cmp.max_abs,
            cos_dist: cmp.cos_dist,
        });
    }
}

fn parse_args() -> (Option<PathBuf>, PathBuf, PathBuf) {
    let root = repo_root();
    let mut gguf = None;
    let mut goldens = root.join("harness/goldens/dequant.json");
    let mut epsilon = root.join("harness/goldens/dequant-epsilon.json");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = args.next().map(PathBuf::from),
            "--goldens" => goldens = PathBuf::from(args.next().expect("--goldens PATH")),
            "--epsilon" => epsilon = PathBuf::from(args.next().expect("--epsilon PATH")),
            other => panic!("unknown arg {other}"),
        }
    }
    if gguf.is_none() {
        let default = root.join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if default.exists() {
            gguf = Some(default);
        }
    }
    (gguf, goldens, epsilon)
}

fn slice_eq<'a>(got: &'a [f32], off: usize, want: &[f32]) -> Option<&'a [f32]> {
    let end = off.checked_add(want.len())?;
    got.get(off..end)
}

fn main() -> ExitCode {
    let (gguf_path, goldens_path, epsilon_path) = parse_args();
    let goldens: Goldens = serde_json::from_str(&fs::read_to_string(&goldens_path).expect("read goldens"))
        .expect("parse goldens");
    if goldens.schema != "milton.dequant/1" {
        eprintln!("fail-closed: unexpected goldens schema {}", goldens.schema);
        return ExitCode::from(2);
    }
    let eps: EpsilonFile =
        serde_json::from_str(&fs::read_to_string(&epsilon_path).expect("read epsilon")).expect("parse epsilon");
    if !(eps.epsilon > 0.0) || !(eps.epsilon_abs > 0.0) {
        eprintln!("fail-closed: epsilon must be positive");
        return ExitCode::from(2);
    }

    let file = gguf_path.as_ref().and_then(|p| {
        if p.exists() {
            Some(GgufFile::open(p).unwrap_or_else(|e| panic!("open GGUF: {e}")))
        } else {
            None
        }
    });

    if let Some(ref f) = file {
        let sha = sha256_bytes(&fs::read(gguf_path.as_ref().unwrap()).expect("read gguf"));
        if sha != goldens.gguf_sha256 {
            eprintln!(
                "fail-closed: GGUF sha256 {sha} != pinned {}",
                goldens.gguf_sha256
            );
            return ExitCode::from(2);
        }
        let _ = f;
    }

    let mut failures = Vec::new();
    let mut n = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut max_cos = 0.0f32;
    let mut sum_cos = 0.0f32;

    for kb in &goldens.kernel_blocks {
        let ty = match type_from_name(&kb.type_name) {
            Ok(t) => t,
            Err(e) => {
                fail(&mut failures, &mut n, kb.id.clone(), e);
                continue;
            }
        };
        let wire = match hex_decode(&kb.wire_hex) {
            Ok(w) => w,
            Err(e) => {
                fail(&mut failures, &mut n, kb.id.clone(), format!("wire_hex: {e}"));
                continue;
            }
        };
        match dequantize(ty, &wire, kb.n_elements, &kb.id) {
            Ok(got) => record(
                &mut failures,
                &mut n,
                &mut max_abs,
                &mut sum_abs,
                &mut max_cos,
                &mut sum_cos,
                kb.id.clone(),
                compare_vectors(&got, &kb.values, eps.epsilon, eps.epsilon_abs),
            ),
            Err(e) => fail(&mut failures, &mut n, kb.id.clone(), format!("dequant_threw:{e}")),
        }
    }

    for t in &goldens.tensors {
        let ty = match type_from_name(&t.type_name) {
            Ok(x) => x,
            Err(e) => {
                fail(&mut failures, &mut n, t.name.clone(), e);
                continue;
            }
        };

        let got = if let Some(ref f) = file {
            match f.dequantize_tensor(&t.name) {
                Ok(v) => v,
                Err(e) => {
                    fail(&mut failures, &mut n, t.name.clone(), format!("dequant_threw:{e}"));
                    continue;
                }
            }
        } else if let Some(hx) = &t.wire_hex {
            let wire = match hex_decode(hx) {
                Ok(w) => w,
                Err(e) => {
                    fail(&mut failures, &mut n, t.name.clone(), format!("wire_hex: {e}"));
                    continue;
                }
            };
            match dequantize(ty, &wire, t.n_elements, &t.name) {
                Ok(v) => v,
                Err(e) => {
                    fail(&mut failures, &mut n, t.name.clone(), format!("dequant_threw:{e}"));
                    continue;
                }
            }
        } else {
            eprintln!(
                "skip {} (GGUF absent, no full wire_hex — unverified, not counted)",
                t.name
            );
            continue;
        };

        if got.len() != t.n_elements {
            fail(
                &mut failures,
                &mut n,
                t.name.clone(),
                format!("dim_mismatch:{}->{}", t.n_elements, got.len()),
            );
            continue;
        }

        let digest = sha256_f32_le(&got);
        if digest != t.sha256_f32_le {
            fail(
                &mut failures,
                &mut n,
                t.name.clone(),
                format!("sha256_mismatch:got {digest} want {}", t.sha256_f32_le),
            );
        }

        if !t.head.is_empty() {
            if let Some(slice) = slice_eq(&got, 0, &t.head) {
                record(
                    &mut failures, &mut n, &mut max_abs, &mut sum_abs, &mut max_cos, &mut sum_cos,
                    format!("{}#head", t.name),
                    compare_vectors(slice, &t.head, eps.epsilon, eps.epsilon_abs),
                );
            } else {
                fail(&mut failures, &mut n, format!("{}#head", t.name), "head_oob".into());
            }
        }
        if !t.tail.is_empty() {
            let off = got.len() - t.tail.len();
            if let Some(slice) = slice_eq(&got, off, &t.tail) {
                record(
                    &mut failures, &mut n, &mut max_abs, &mut sum_abs, &mut max_cos, &mut sum_cos,
                    format!("{}#tail", t.name),
                    compare_vectors(slice, &t.tail, eps.epsilon, eps.epsilon_abs),
                );
            } else {
                fail(&mut failures, &mut n, format!("{}#tail", t.name), "tail_oob".into());
            }
        }
        if !t.mid.is_empty() {
            let off = t.mid_offset.unwrap_or(got.len() / 2);
            if let Some(slice) = slice_eq(&got, off, &t.mid) {
                record(
                    &mut failures, &mut n, &mut max_abs, &mut sum_abs, &mut max_cos, &mut sum_cos,
                    format!("{}#mid", t.name),
                    compare_vectors(slice, &t.mid, eps.epsilon, eps.epsilon_abs),
                );
            } else {
                fail(&mut failures, &mut n, format!("{}#mid", t.name), "mid_oob".into());
            }
        }
    }

    let pass = failures.is_empty();
    let receipt = Receipt {
        schema: "milton.dequant.receipt/1",
        result: if pass { "pass" } else { "fail" },
        n,
        failed: failures.len(),
        max_abs,
        mean_abs: if n == 0 { 0.0 } else { sum_abs / n as f32 },
        max_cos_dist: max_cos,
        mean_cos_dist: if n == 0 { 0.0 } else { sum_cos / n as f32 },
        epsilon: eps.epsilon,
        epsilon_abs: eps.epsilon_abs,
        gguf_sha256: goldens.gguf_sha256,
        failures,
    };
    println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
