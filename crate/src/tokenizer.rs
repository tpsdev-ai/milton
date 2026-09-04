//! Pure nomic-embed-text-v1.5 tokenizer: BERT WordPiece + Flair prefixes.
//!
//! Pipeline matches the pinned HF `tokenizer.json`:
//! BertNormalizer (clean_text, chinese, strip_accents via lowercase default,
//! lowercase) → BertPreTokenizer → WordPiece (`##`) → `[CLS] $A [SEP]`.
//!
//! No I/O in this module.

use crate::prefix::{apply_prefix, Prefix};
use crate::vocab::Vocab;
use std::fmt;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

const ID_CLS: u32 = 101;
const ID_SEP: u32 = 102;

/// Public error type. `tokenize` itself is infallible for valid `Prefix`;
/// this exists so an unverified / invalid path can refuse rather than guess.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizeError {
    InvalidPrefix(String),
}

impl fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrefix(k) => write!(
                f,
                "tokenize: invalid prefix kind {k:?} (expected 'document' | 'query' | 'none')"
            ),
        }
    }
}

impl std::error::Error for TokenizeError {}

/// BERT-normalizer + WordPiece switches. `NOMIC` is the only verified path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pipeline {
    pub clean_text: bool,
    pub handle_chinese_chars: bool,
    pub strip_accents: bool,
    pub lowercase: bool,
    pub add_special: bool,
}

impl Pipeline {
    /// tokenizer.json: BertNormalizer(clean_text, chinese, strip_accents=null, lowercase)
    /// → strip_accents defaults to lowercase = true.
    pub const NOMIC: Self = Self {
        clean_text: true,
        handle_chinese_chars: true,
        strip_accents: true,
        lowercase: true,
        add_special: true,
    };
}

/// `tokenize(text, prefix) -> token IDs`.
///
/// Applies the Flair prefix template, then the pinned nomic WordPiece pipeline.
/// The only verified path; there is no "best effort" fallback.
pub fn tokenize(text: &str, prefix: Prefix) -> Vec<u32> {
    tokenize_with(text, prefix, Pipeline::NOMIC)
}

/// Same as `tokenize`, but parse the harness kind string (`document` | `query` | `none`).
/// Refuses the silent inversion (`"search_document"` as a kind) instead of guessing.
pub fn tokenize_kind(text: &str, kind: &str) -> Result<Vec<u32>, TokenizeError> {
    let prefix = Prefix::parse(kind).map_err(|e| match e {
        crate::prefix::PrefixError::InvalidKind(k) => TokenizeError::InvalidPrefix(k),
    })?;
    Ok(tokenize(text, prefix))
}

/// Tokenize a string that already has the prefix applied (or is passthrough).
pub fn tokenize_prefixed(prefixed: &str) -> Vec<u32> {
    tokenize_prefixed_with(prefixed, Pipeline::NOMIC)
}

pub(crate) fn tokenize_with(text: &str, prefix: Prefix, pipeline: Pipeline) -> Vec<u32> {
    tokenize_prefixed_with(&apply_prefix(text, prefix), pipeline)
}

/// Deliberately-wrong tokenizations used only by the must-fail control.
/// These are not public API and must never be used as a fallback.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum WrongKind {
    DroppedPrefix,
    NoLowercase,
    NoAccentStrip,
}

#[cfg(test)]
pub(crate) fn tokenize_wrong(text: &str, prefix: Prefix, kind: WrongKind) -> Vec<u32> {
    match kind {
        WrongKind::DroppedPrefix => tokenize_with(text, Prefix::None, Pipeline::NOMIC),
        WrongKind::NoLowercase => tokenize_with(
            text,
            prefix,
            Pipeline {
                lowercase: false,
                ..Pipeline::NOMIC
            },
        ),
        WrongKind::NoAccentStrip => tokenize_with(
            text,
            prefix,
            Pipeline {
                strip_accents: false,
                ..Pipeline::NOMIC
            },
        ),
    }
}

pub(crate) fn tokenize_prefixed_with(prefixed: &str, pipeline: Pipeline) -> Vec<u32> {
    let normalized = normalize(prefixed, pipeline);
    let pieces = pretokenize(&normalized);
    let vocab = Vocab::global();
    let mut ids = Vec::new();
    if pipeline.add_special {
        ids.push(ID_CLS);
    }
    for piece in pieces {
        ids.extend(vocab.wordpiece(&piece));
    }
    if pipeline.add_special {
        ids.push(ID_SEP);
    }
    ids
}

fn normalize(text: &str, pipeline: Pipeline) -> String {
    let mut s = text.to_string();
    if pipeline.clean_text {
        s = clean_text(&s);
    }
    if pipeline.handle_chinese_chars {
        s = handle_chinese_chars(&s);
    }
    if pipeline.strip_accents {
        s = strip_accents(&s);
    }
    if pipeline.lowercase {
        s = lowercase(&s);
    }
    s
}

fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_other(c: char) -> bool {
    // tokenizers UnicodeCategories::is_other — Cc/Cf/Cs/Co/Cn
    matches!(
        get_general_category(c),
        GeneralCategory::Control
            | GeneralCategory::Format
            | GeneralCategory::Surrogate
            | GeneralCategory::PrivateUse
            | GeneralCategory::Unassigned
    )
}

fn is_punctuation_cat(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

fn is_mark_nonspacing(c: char) -> bool {
    get_general_category(c) == GeneralCategory::NonspacingMark
}

fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => is_other(c),
    }
}

fn is_chinese_char(c: char) -> bool {
    // Same ranges as huggingface/tokenizers BertNormalizer, including the
    // 0x2B920 start (documented there; llama.cpp copies it).
    matches!(
        c as usize,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

fn is_bert_punc(c: char) -> bool {
    c.is_ascii_punctuation() || is_punctuation_cat(c)
}

fn clean_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c == '\0' || c == '\u{FFFD}' || is_control(c) {
            continue;
        }
        if is_whitespace(c) {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn handle_chinese_chars(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if is_chinese_char(c) {
            out.push(' ');
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn strip_accents(text: &str) -> String {
    text.nfd().filter(|c| !is_mark_nonspacing(*c)).collect()
}

fn lowercase(text: &str) -> String {
    text.chars().flat_map(|c| c.to_lowercase()).collect()
}

fn pretokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    for word in text.split(is_whitespace) {
        if word.is_empty() {
            continue;
        }
        split_punct(word, &mut words);
    }
    words
}

fn split_punct(word: &str, out: &mut Vec<String>) {
    let mut current = String::new();
    for c in word.chars() {
        if is_bert_punc(c) {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.push(c.to_string());
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn hello_document_matches_pin() {
        assert_eq!(
            tokenize("hello", Prefix::Document),
            vec![101, 3945, 1035, 6254, 1024, 7592, 102]
        );
    }

    #[test]
    fn hello_query_is_asymmetric() {
        assert_eq!(
            tokenize("hello", Prefix::Query),
            vec![101, 3945, 1035, 23032, 1024, 7592, 102]
        );
        assert_ne!(
            tokenize("hello", Prefix::Query),
            tokenize("hello", Prefix::Document)
        );
    }

    #[test]
    fn empty_none_is_cls_sep() {
        assert_eq!(tokenize("", Prefix::None), vec![101, 102]);
    }

    #[test]
    fn empty_document_embeds_the_prefix() {
        assert_eq!(
            tokenize("", Prefix::Document),
            vec![101, 3945, 1035, 6254, 1024, 102]
        );
    }

    /// Compare-only fixtures (not the 18-case goldens corpus) pin n=31/32/33
    /// so the ATTN_PARALLEL_MIN_TOKENS=32 serial→parallel gate is itself a
    /// bit-exact check. Token counts must match `ids.len()` after prefix.
    #[test]
    fn attn_crossover_fixtures_pin_min_tokens_boundary() {
        let raw = include_str!("../../harness/corpus/compare-crossover.json");
        let doc: serde_json::Value =
            serde_json::from_str(raw).expect("compare-crossover.json");
        assert_eq!(doc["gate"], 32);
        assert_eq!(
            crate::model::ATTN_PARALLEL_MIN_TOKENS,
            doc["gate"].as_u64().unwrap() as usize,
            "Rust default must match compare-crossover.json gate",
        );
        assert_eq!(
            crate::model::attn_parallel_min_tokens(),
            doc["gate"].as_u64().unwrap() as usize,
            "effective gate (no override) must match compare-crossover.json gate",
        );
        let want = [
            ("attn-crossover-31", 31usize),
            ("attn-crossover-32", 32),
            ("attn-crossover-33", 33),
        ];
        let cases = doc["cases"].as_array().expect("cases");
        assert_eq!(cases.len(), want.len());
        for (id, n) in want {
            let case = cases
                .iter()
                .find(|c| c["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("missing {id}"));
            let text = case["text"].as_str().expect("text");
            let prefix = Prefix::parse(case["prefix"].as_str().expect("prefix")).unwrap();
            let ids = tokenize(text, prefix);
            assert_eq!(
                ids.len(),
                n,
                "{id} tokenize len {} != pinned n_tokens {n}",
                ids.len()
            );
            assert_eq!(case["n_tokens"].as_u64().unwrap() as usize, n);
        }
    }
}
