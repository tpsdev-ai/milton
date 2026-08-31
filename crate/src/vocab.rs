//! Embedded nomic-embed-text-v1.5 WordPiece vocab (bert-base-uncased IDs).
//!
//! Pinned file: `crate/vocab/vocab.txt`
//! sha256 `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
//! Source: huggingface.co/nomic-ai/nomic-embed-text-v1.5 @ e9b6763023c676ca8431644204f50c2b100d9aab
//!
//! Loaded from `include_str!` — no runtime I/O.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Pinned vocab.txt contents (one token per line; line index is the id).
pub const VOCAB_TXT: &str = include_str!("../vocab/vocab.txt");

pub const UNK: u32 = 100;
pub const CLS: u32 = 101;
pub const SEP: u32 = 102;

pub const UNK_TOKEN: &str = "[UNK]";
pub const CLS_TOKEN: &str = "[CLS]";
pub const SEP_TOKEN: &str = "[SEP]";

const CONTINUING: &str = "##";
const MAX_CHARS: usize = 100;

pub struct Vocab {
    token_to_id: HashMap<String, u32>,
}

impl Vocab {
    pub fn global() -> &'static Self {
        static CELL: OnceLock<Vocab> = OnceLock::new();
        CELL.get_or_init(Self::from_embedded)
    }

    fn from_embedded() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id: u32 = 0;
        for line in VOCAB_TXT.split('\n') {
            if line.is_empty() && id > 0 {
                // trailing newline after the last token
                continue;
            }
            let token = line.trim_end();
            token_to_id.insert(token.to_string(), id);
            id += 1;
        }
        let v = Self { token_to_id };
        // Fail closed: a vocab that does not contain the BERT specials is not nomic.
        assert_eq!(v.id(UNK_TOKEN), Some(UNK), "embedded vocab UNK id mismatch");
        assert_eq!(v.id(CLS_TOKEN), Some(CLS), "embedded vocab CLS id mismatch");
        assert_eq!(v.id(SEP_TOKEN), Some(SEP), "embedded vocab SEP id mismatch");
        assert_eq!(
            v.token_to_id.len(),
            30522,
            "embedded vocab size mismatch — pin is 30522"
        );
        v
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn unk(&self) -> u32 {
        UNK
    }

    /// Greedy longest-match WordPiece. `##` continuation prefix. Whole-word [UNK]
    /// if any slice is unknown or the word exceeds `max_input_chars_per_word`.
    pub fn wordpiece(&self, word: &str) -> Vec<u32> {
        let char_len = word.chars().count();
        if char_len > MAX_CHARS {
            return vec![self.unk()];
        }
        if word.is_empty() {
            return Vec::new();
        }

        let mut ids = Vec::new();
        let mut start = 0usize;
        let bytes = word.as_bytes();
        let n = bytes.len();

        while start < n {
            let mut end = n;
            let mut found: Option<u32> = None;
            while start < end {
                let slice = &word[start..end];
                let id = if start > 0 {
                    let mut key = String::with_capacity(CONTINUING.len() + slice.len());
                    key.push_str(CONTINUING);
                    key.push_str(slice);
                    self.id(&key)
                } else {
                    self.id(slice)
                };
                if let Some(id) = id {
                    found = Some(id);
                    break;
                }
                // Step back one UTF-8 character (HF tokenizers WordPiece).
                end = match word[..end].chars().next_back() {
                    Some(c) => end - c.len_utf8(),
                    None => start,
                };
            }
            match found {
                Some(id) => {
                    ids.push(id);
                    start = end;
                }
                None => return vec![self.unk()],
            }
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_and_hello() {
        let v = Vocab::global();
        assert_eq!(v.id("hello"), Some(7592));
        assert_eq!(v.id("[CLS]"), Some(101));
        assert_eq!(v.id("[SEP]"), Some(102));
        assert_eq!(v.id("[UNK]"), Some(100));
        assert_eq!(v.id("[PAD]"), Some(0));
        assert_eq!(v.id("[MASK]"), Some(103)); // present in the pin; not used by tokenize
    }

    #[test]
    fn wordpiece_hello_and_unk() {
        let v = Vocab::global();
        assert_eq!(v.wordpiece("hello"), vec![7592]);
        assert_eq!(
            v.wordpiece("stapler"),
            vec![v.id("staple").unwrap(), v.id("##r").unwrap()]
        );
    }
}
