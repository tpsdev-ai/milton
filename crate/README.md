# Milton crate — tokenizer slice (issue #3)

Pure Rust nomic-embed-text-v1.5 WordPiece + Flair prefix convention.

```rust
use milton::{tokenize, Prefix};

let ids = tokenize("hello", Prefix::Document);
// [CLS] search _ document : hello [SEP]
```

- **Core is pure:** `tokenize` / `apply_prefix` do no I/O.
- **Vocab is embedded** from the pinned `vocab/vocab.txt` (HF nomic-embed-text-v1.5 @ `e9b6763…`).
- **Prefixes** are byte-identical to `harness/lib/prefix.js`: `search_document: ` / `search_query: ` / passthrough. The space after the colon is load-bearing.
- **Not in this slice:** GGUF load (#4), forward/mean-pool/L2 (#5), WASM packaging (#6).

Goldens: `harness/goldens/tokens.json`. Pin + digest: `harness/goldens/tokenizer-pin.json`.

```sh
cargo test --manifest-path crate/Cargo.toml
cargo run --manifest-path crate/Cargo.toml --features cli --bin milton-tokenize -- --check-goldens
```
