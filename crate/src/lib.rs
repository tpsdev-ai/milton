//! Milton tokenizer slice — nomic-embed-text-v1.5 WordPiece + Flair prefixes.
//!
//! The core (`tokenize`, `apply_prefix`) is pure: no I/O, no network, no filesystem.
//! Vocab is embedded at compile time from the pinned `vocab.txt`.
//! GGUF load, forward, and WASM packaging are later issues — not this crate slice.

mod prefix;
mod tokenizer;
mod vocab;

#[cfg(test)]
mod conformance;

pub use prefix::{apply_prefix, Prefix, PrefixError};
pub use tokenizer::{tokenize, tokenize_kind, tokenize_prefixed, TokenizeError};
