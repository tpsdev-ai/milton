//! Flair nomic prefix convention — byte-identical to `harness/lib/prefix.js`.
//!
//! Source of truth (documented by the harness README):
//! - Flair `resources/embeddings-provider.ts`
//! - harper-fabric-embeddings `src/engine.ts` `NOMIC_TEMPLATES`
//!
//! ```text
//! document: "search_document: {text}"
//! query:    "search_query: {text}"
//! none:     "{text}"   // omitted inputType = passthrough
//! ```
//!
//! The space after the colon is load-bearing. Passing the prefix STRING
//! (`"search_document"`) as the kind VALUE is the known silent inversion
//! bug — this module only accepts the closed union `document` | `query` | `none`.

use std::fmt;

/// Closed prefix union. Matches `PREFIX_KINDS` in the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefix {
    Document,
    Query,
    None,
}

impl Prefix {
    /// `search_document: ` — space after the colon is load-bearing.
    pub const DOCUMENT: &'static str = "search_document: ";
    /// `search_query: ` — space after the colon is load-bearing.
    pub const QUERY: &'static str = "search_query: ";

    /// Parse a harness/Flair kind. Rejects the prefix *string* used as a kind.
    pub fn parse(kind: &str) -> Result<Self, PrefixError> {
        match kind {
            "document" => Ok(Self::Document),
            "query" => Ok(Self::Query),
            "none" => Ok(Self::None),
            other => Err(PrefixError::InvalidKind(other.to_string())),
        }
    }

    pub fn as_kind(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
            Self::None => "none",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Document => Self::DOCUMENT,
            Self::Query => Self::QUERY,
            Self::None => "",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrefixError {
    InvalidKind(String),
}

impl fmt::Display for PrefixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKind(k) => write!(
                f,
                "applyPrefix: invalid kind {k:?} (expected 'document' | 'query' | 'none')"
            ),
        }
    }
}

impl std::error::Error for PrefixError {}

/// Prefix templates as **config**, not architecture code.
///
/// v1's default is Flair's nomic convention (`search_document: ` /
/// `search_query: ` / passthrough). The space after the colon is load-bearing.
/// A second BERT-family file is a new config, not a rewrite. No model registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixConfig {
    pub document: String,
    pub query: String,
    pub none: String,
}

impl Default for PrefixConfig {
    fn default() -> Self {
        Self::flair_nomic()
    }
}

impl PrefixConfig {
    /// v1 verified config — byte-identical to `harness/lib/prefix.js`.
    pub fn flair_nomic() -> Self {
        Self {
            document: Prefix::DOCUMENT.to_string(),
            query: Prefix::QUERY.to_string(),
            none: String::new(),
        }
    }

    pub fn template(&self, prefix: Prefix) -> &str {
        match prefix {
            Prefix::Document => &self.document,
            Prefix::Query => &self.query,
            Prefix::None => &self.none,
        }
    }

    pub fn apply(&self, text: &str, prefix: Prefix) -> String {
        let p = self.template(prefix);
        if p.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(p.len() + text.len());
        out.push_str(p);
        out.push_str(text);
        out
    }
}

/// Concatenate the configured template and `text`. Unconditional — already-prefixed
/// bodies are double-prefixed, matching Flair. Uses the v1 Flair config.
pub fn apply_prefix(text: &str, prefix: Prefix) -> String {
    PrefixConfig::default().apply(text, prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_after_colon_is_load_bearing() {
        assert_eq!(Prefix::DOCUMENT, "search_document: ");
        assert_eq!(Prefix::QUERY, "search_query: ");
        assert_eq!(
            apply_prefix("hello", Prefix::Document),
            "search_document: hello"
        );
        assert_eq!(apply_prefix("hello", Prefix::Query), "search_query: hello");
        assert_eq!(apply_prefix("hello", Prefix::None), "hello");
    }

    #[test]
    fn concatenates_unconditionally() {
        assert_eq!(
            apply_prefix("search_document: already prefixed", Prefix::Document),
            "search_document: search_document: already prefixed"
        );
    }

    #[test]
    fn preserves_empty_and_whitespace() {
        assert_eq!(apply_prefix("", Prefix::Document), "search_document: ");
        assert_eq!(apply_prefix("  x  ", Prefix::Query), "search_query:   x  ");
        assert_eq!(apply_prefix("", Prefix::None), "");
    }

    #[test]
    fn rejects_prefix_string_used_as_kind() {
        assert!(matches!(
            Prefix::parse("search_document"),
            Err(PrefixError::InvalidKind(_))
        ));
        assert!(matches!(
            Prefix::parse("search_query"),
            Err(PrefixError::InvalidKind(_))
        ));
        assert!(Prefix::parse("document").is_ok());
        assert!(Prefix::parse("query").is_ok());
        assert!(Prefix::parse("none").is_ok());
    }
}
