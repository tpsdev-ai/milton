//! One-shot GGUF inspector for the forward-pass slice. Not shipped.

use milton::GgufFile;
use std::env;
use std::path::PathBuf;

fn main() {
    let path = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf")
    });
    let gguf = GgufFile::open(&path).expect("open GGUF");
    println!("path {}", path.display());
    println!("=== metadata ===");
    for (k, v) in &gguf.metadata {
        let s = v.as_display();
        if s.len() > 200 {
            println!("{k} = {}…", &s[..200]);
        } else {
            println!("{k} = {s}");
        }
    }
    println!("=== tensors {} ===", gguf.tensors.len());
    for t in &gguf.tensors {
        println!(
            "{:<40} {:<8} {:?} n={}",
            t.name,
            t.tensor_type.name(),
            t.dimensions,
            t.n_elements()
        );
    }
}
