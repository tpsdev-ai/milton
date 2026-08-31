//! Exact-match goldens + named must-fail controls.
//! Compiled only under `cfg(test)`.

use crate::prefix::Prefix;
use crate::tokenizer::{tokenize, tokenize_wrong, WrongKind};
use serde_json::Value;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate/ sits in the repo root")
        .to_path_buf()
}

fn load(path: &str) -> Value {
    let p = repo_root().join(path);
    let raw = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

struct Case {
    id: String,
    text: String,
    prefix: Prefix,
    expected: Vec<u32>,
}

fn load_cases() -> (Vec<Case>, Value) {
    let gold = load("harness/goldens/tokens.json");
    let corpus = load("harness/corpus/corpus.json");
    assert_eq!(gold["schema"], "milton.token-goldens/1");
    assert_eq!(
        gold["n"].as_u64().unwrap() as usize,
        corpus["cases"].as_array().unwrap().len()
    );
    let mut cases = Vec::new();
    for c in corpus["cases"].as_array().unwrap() {
        let id = c["id"].as_str().unwrap().to_string();
        let item = gold["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|it| it["id"].as_str() == Some(id.as_str()))
            .unwrap_or_else(|| panic!("golden missing {id}"));
        let expected = item["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        cases.push(Case {
            id,
            text: c["text"].as_str().unwrap().to_string(),
            prefix: Prefix::parse(c["prefix"].as_str().unwrap()).unwrap(),
            expected,
        });
    }
    (cases, gold)
}

struct Diff {
    id: String,
    expected: Vec<u32>,
    got: Vec<u32>,
}

fn diffs(cases: &[Case], got: impl Fn(&Case) -> Vec<u32>) -> Vec<Diff> {
    let mut out = Vec::new();
    for c in cases {
        let g = got(c);
        if g != c.expected {
            out.push(Diff {
                id: c.id.clone(),
                expected: c.expected.clone(),
                got: g,
            });
        }
    }
    out
}

#[test]
fn exact_match_all_corpus_cases() {
    let (cases, gold) = load_cases();
    let n = cases.len();
    assert_eq!(n, 18, "conformance corpus is 18 cases");
    let misses = diffs(&cases, |c| tokenize(&c.text, c.prefix));
    if !misses.is_empty() {
        let mut msg = format!(
            "EXACT-MATCH FAIL: {}/{} corpus cases mismatch (tokens_digest={})\n",
            misses.len(),
            n,
            gold["tokens_digest"]
        );
        for m in &misses {
            msg.push_str(&format!(
                "  {id}: expected {exp:?} got {got:?}\n",
                id = m.id,
                exp = m.expected,
                got = m.got
            ));
        }
        panic!("{msg}");
    }
}

#[test]
fn must_fail_dropped_prefix_turns_red_and_names_prefixed_items() {
    let (cases, _) = load_cases();
    let misses = diffs(&cases, |c| {
        tokenize_wrong(&c.text, c.prefix, WrongKind::DroppedPrefix)
    });
    let named: Vec<&str> = misses.iter().map(|m| m.id.as_str()).collect();
    assert!(
        !misses.is_empty(),
        "must_fail_dropped_prefix: expected RED, got a clean match — control is broken"
    );
    assert!(
        named.contains(&"short-hello-document"),
        "must_fail_dropped_prefix must name short-hello-document; named {named:?}"
    );
    assert!(
        named.contains(&"short-hello-query"),
        "must_fail_dropped_prefix must name short-hello-query; named {named:?}"
    );
    assert!(
        !named.contains(&"short-hello-none"),
        "unprefixed sibling must still match; named {named:?}"
    );
}

#[test]
fn must_fail_wrong_casing_turns_red_and_is_named() {
    let (cases, _) = load_cases();
    let misses = diffs(&cases, |c| {
        tokenize_wrong(&c.text, c.prefix, WrongKind::NoLowercase)
    });
    let named: Vec<&str> = misses.iter().map(|m| m.id.as_str()).collect();
    assert!(
        !misses.is_empty(),
        "must_fail_wrong_casing: expected RED (skipping lowercase must disagree with goldens)"
    );
    // Mixed-case Flair / long bodies are the intended victims.
    assert!(
        named.iter().any(|id| {
            matches!(
                *id,
                "long-repeated"
                    | "flair-memory-preference"
                    | "flair-memory-decision"
                    | "flair-query-recall"
                    | "prefix-already-present"
                    | "numeric-ids"
            )
        }),
        "must_fail_wrong_casing must name a mixed-case corpus item; named {named:?}"
    );
}

#[test]
fn must_fail_wrong_normalization_turns_red_and_is_named() {
    let (cases, _) = load_cases();
    let misses = diffs(&cases, |c| {
        tokenize_wrong(&c.text, c.prefix, WrongKind::NoAccentStrip)
    });
    let named: Vec<&str> = misses.iter().map(|m| m.id.as_str()).collect();
    assert!(
        !misses.is_empty(),
        "must_fail_wrong_normalization: expected RED (skipping accent-strip must disagree)"
    );
    assert!(
        named.contains(&"unicode-nfc") || named.contains(&"unicode-nfd"),
        "must_fail_wrong_normalization must name a unicode accent case; named {named:?}"
    );
}

#[test]
fn nfc_and_nfd_goldens_match_the_pinned_reference() {
    let (cases, _) = load_cases();
    let nfc = cases.iter().find(|c| c.id == "unicode-nfc").unwrap();
    let nfd = cases.iter().find(|c| c.id == "unicode-nfd").unwrap();
    assert_eq!(
        nfc.expected, nfd.expected,
        "pinned HF nomic tokenizer collapses NFC/NFD via strip_accents; goldens must record that"
    );
    assert_eq!(tokenize(&nfc.text, nfc.prefix), nfc.expected);
    assert_eq!(tokenize(&nfd.text, nfd.prefix), nfd.expected);
}
