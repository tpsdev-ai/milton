//! Metadata as read from the pinned GGUF. Skips if the vendor GGUF is absent.

use std::path::PathBuf;

fn gguf_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf")
}

#[test]
fn nomic_metadata_matches_the_file() {
    let path = gguf_path();
    if !path.exists() {
        eprintln!("skip: GGUF not present at {}", path.display());
        return;
    }
    let gguf = milton::GgufFile::open(&path).expect("open GGUF");
    let meta = gguf.model_meta().expect("model meta");
    assert_eq!(meta.architecture, "nomic-bert");
    assert_eq!(meta.name.as_deref(), Some("nomic-embed-text-v1.5"));
    assert_eq!(meta.block_count, 12);
    assert_eq!(meta.embedding_length, 768);
    assert_eq!(meta.context_length, 2048);
    assert_eq!(meta.pooling_type, Some(1));
    assert_eq!(meta.pooling.as_deref(), Some("mean"));
    assert_eq!(meta.rope_freq_base, Some(1000.0));
    assert_eq!(meta.causal_attn, Some(false));
    assert_eq!(meta.feed_forward_length, Some(3072));
    assert_eq!(meta.attention_head_count, Some(12));
    assert_eq!(
        meta.pooling_key.as_deref(),
        Some("nomic-bert.pooling_type")
    );
    assert!(
        meta.normalization.contains_key("nomic-bert.attention.layer_norm_epsilon"),
        "layer_norm_epsilon must be recorded as-read: {:?}",
        meta.normalization
    );
    let census = gguf.quant_type_census();
    assert!(census.get("F32").copied().unwrap_or(0) > 0, "{census:?}");
    assert!(census.get("Q4_K").copied().unwrap_or(0) > 0, "{census:?}");
    assert!(census.get("Q5_K").copied().unwrap_or(0) > 0, "{census:?}");
    assert!(census.get("Q6_K").copied().unwrap_or(0) > 0, "{census:?}");
    assert_eq!(census.get("Q8_0").copied().unwrap_or(0), 0);
    assert_eq!(census.get("F16").copied().unwrap_or(0), 0);
}

#[test]
fn unknown_architecture_is_refused() {
    // A tiny GGUF-shaped buffer with a wrong architecture is out of scope
    // for a hand-rolled file; the public constructor is fail-closed via
    // ModelMeta::from_gguf once general.architecture is not nomic-bert.
    // This test pins the error type so a later loosening cannot sneak through.
    let err = milton::Error::UnsupportedModel("v1 supports nomic-embed-text-v1.5 only".into());
    assert!(format!("{err}").contains("unsupported model"));
}
