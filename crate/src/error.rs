//! Fail-closed errors. An unverified path must refuse, never guess.

use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub enum Error {
    Io { path: PathBuf, source: io::Error },
    InvalidGguf(String),
    UnsupportedGguf(String),
    UnsupportedTensorType { name: String, type_id: i32, type_name: String },
    InvalidTensorData(String),
    MissingTensor(String),
    UnsupportedModel(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "io error at {}: {source}", path.display()),
            Self::InvalidGguf(msg) => write!(f, "invalid GGUF: {msg}"),
            Self::UnsupportedGguf(msg) => write!(f, "unsupported GGUF: {msg}"),
            Self::UnsupportedTensorType {
                name,
                type_id,
                type_name,
            } => write!(
                f,
                "fail-closed: tensor {name} has unverified type {type_name} (id {type_id})"
            ),
            Self::InvalidTensorData(msg) => write!(f, "invalid tensor data: {msg}"),
            Self::MissingTensor(name) => write!(f, "missing tensor {name}"),
            Self::UnsupportedModel(msg) => write!(f, "unsupported model: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
