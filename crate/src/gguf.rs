//! GGUF v2/v3 reader for nomic-embed-text-v1.5.
//!
//! Layout and type-id table follow llama.cpp `gguf.h` / `ggml-common.h` and
//! camelid's `src/gguf/reader.rs` (MIT — github.com/timtoole02/Camelid).
//! Patterns are mirrored, not vendored. Unknown tensor types fail closed.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const GGML_MAX_DIMS: u32 = 4;
const GGML_MAX_NAME: usize = 64;

/// GGML tensor type ids we are willing to name. Anything else is `Unknown`
/// and dequant refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorType {
    F32,
    F16,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Unknown(i32),
}

impl TensorType {
    pub fn from_id(id: i32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            other => Self::Unknown(other),
        }
    }

    pub fn id(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Unknown(id) => id,
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::F32 => "F32".to_string(),
            Self::F16 => "F16".to_string(),
            Self::Q8_0 => "Q8_0".to_string(),
            Self::Q4K => "Q4_K".to_string(),
            Self::Q5K => "Q5_K".to_string(),
            Self::Q6K => "Q6_K".to_string(),
            Self::Unknown(id) => format!("Unknown({id})"),
        }
    }

    /// `(block_elements, block_bytes)` from llama.cpp `ggml-common.h` struct sizes.
    /// Q4_K_M is a *recipe*, not a GGML type: the file mixes Q4_K / Q6_K / F32
    /// (and possibly Q8_0 / F16). Confirm against the file, do not assume.
    pub fn layout(self) -> Option<(u64, u64)> {
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q4K => Some((256, 144)),
            Self::Q5K => Some((256, 176)),
            Self::Q6K => Some((256, 210)),
            Self::Unknown(_) => None,
        }
    }

    pub fn is_covered(self) -> bool {
        !matches!(self, Self::Unknown(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    pub fn as_display(&self) -> String {
        match self {
            Self::U8(v) => v.to_string(),
            Self::I8(v) => v.to_string(),
            Self::U16(v) => v.to_string(),
            Self::I16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::F32(v) => format!("{v}"),
            Self::Bool(v) => v.to_string(),
            Self::String(v) => v.clone(),
            Self::Array(vs) => {
                let inner: Vec<String> = vs.iter().map(Self::as_display).collect();
                format!("[{}]", inner.join(","))
            }
            Self::U64(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::F64(v) => format!("{v}"),
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::U8(v) => Some(i64::from(v)),
            Self::I8(v) => Some(i64::from(v)),
            Self::U16(v) => Some(i64::from(v)),
            Self::I16(v) => Some(i64::from(v)),
            Self::U32(v) => Some(i64::from(v)),
            Self::I32(v) => Some(i64::from(v)),
            Self::U64(v) => i64::try_from(v).ok(),
            Self::I64(v) => Some(v),
            Self::Bool(v) => Some(i64::from(v)),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            _ => self.as_i64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: TensorType,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub n_bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dimensions.iter().copied().product()
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub data_start_offset: u64,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    bytes: Vec<u8>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(path.to_path_buf(), bytes)
    }

    /// Parse a GGUF already in memory. WASM uses this so the runtime
    /// does not need filesystem access at `npm i` or embed time.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(PathBuf::from("<memory>"), bytes)
    }

    pub fn parse(path: PathBuf, bytes: Vec<u8>) -> Result<Self> {
        let file_len = bytes.len() as u64;
        let mut cur = Cursor::new(&bytes, path.clone());

        let magic = cur.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(Error::InvalidGguf("bad magic; expected GGUF".into()));
        }
        let version = cur.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::UnsupportedGguf(format!(
                "version {version}; expected v2 or v3"
            )));
        }
        let tensor_count = cur.read_i64()?;
        let metadata_count = cur.read_i64()?;
        if tensor_count < 0 || metadata_count < 0 {
            return Err(Error::InvalidGguf(
                "negative tensor or metadata count".into(),
            ));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = cur.read_string()?;
            let value = read_value(&mut cur)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(Error::InvalidGguf(format!("duplicate metadata key {key}")));
            }
        }

        let alignment = match metadata.get("general.alignment") {
            Some(MetadataValue::U32(v)) => u64::from(*v),
            Some(MetadataValue::U64(v)) => *v,
            Some(_) => {
                return Err(Error::InvalidGguf(
                    "general.alignment has non-integer type".into(),
                ))
            }
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::InvalidGguf(format!("invalid alignment {alignment}")));
        }

        let mut raw = Vec::new();
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            if name.len() >= GGML_MAX_NAME {
                return Err(Error::InvalidGguf(format!(
                    "tensor name {name} exceeds GGML_MAX_NAME"
                )));
            }
            let n_dims = cur.read_u32()?;
            if n_dims == 0 || n_dims > GGML_MAX_DIMS {
                return Err(Error::InvalidGguf(format!(
                    "tensor {name} has invalid dimension count {n_dims}"
                )));
            }
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let dim = cur.read_i64()?;
                if dim < 0 {
                    return Err(Error::InvalidGguf(format!(
                        "tensor {name} has negative dimension {dim}"
                    )));
                }
                dimensions.push(dim as u64);
            }
            let tensor_type = TensorType::from_id(cur.read_i32()?);
            let relative_offset = cur.read_u64()?;
            raw.push((name, dimensions, tensor_type, relative_offset));
        }

        let data_start_offset = align_to(cur.pos, alignment)?;
        if data_start_offset > file_len {
            return Err(Error::InvalidGguf(
                "aligned tensor data start is beyond end of file".into(),
            ));
        }

        let mut tensors = Vec::with_capacity(raw.len());
        let mut seen = BTreeMap::new();
        let mut expected_offset = 0u64;
        for (name, dimensions, tensor_type, relative_offset) in raw {
            if seen.insert(name.clone(), ()).is_some() {
                return Err(Error::InvalidGguf(format!("duplicate tensor name {name}")));
            }
            if relative_offset != expected_offset {
                return Err(Error::InvalidGguf(format!(
                    "tensor {name} offset {relative_offset} is not contiguous; expected {expected_offset}"
                )));
            }
            let n_bytes = tensor_nbytes(&name, &dimensions, tensor_type)?;
            let absolute_offset = data_start_offset.checked_add(relative_offset).ok_or_else(|| {
                Error::InvalidGguf(format!("tensor {name} absolute offset overflow"))
            })?;
            let end = absolute_offset
                .checked_add(n_bytes)
                .ok_or_else(|| Error::InvalidGguf(format!("tensor {name} byte range overflow")))?;
            if end > file_len {
                return Err(Error::InvalidGguf(format!(
                    "tensor {name} data extends beyond end of file"
                )));
            }
            tensors.push(TensorInfo {
                name,
                dimensions,
                tensor_type,
                relative_offset,
                absolute_offset,
                n_bytes,
            });
            expected_offset = align_to(
                relative_offset
                    .checked_add(n_bytes)
                    .ok_or_else(|| Error::InvalidGguf("tensor offset overflow".into()))?,
                alignment,
            )?;
        }

        Ok(Self {
            path,
            version,
            alignment,
            data_start_offset,
            metadata,
            tensors,
            bytes,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = info.absolute_offset as usize;
        let end = start
            .checked_add(info.n_bytes as usize)
            .ok_or_else(|| Error::InvalidTensorData(format!("tensor {} range overflow", info.name)))?;
        self.bytes.get(start..end).ok_or_else(|| {
            Error::InvalidTensorData(format!("tensor {} bytes out of range", info.name))
        })
    }

    pub fn quant_type_census(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for t in &self.tensors {
            *counts.entry(t.tensor_type.name()).or_insert(0) += 1;
        }
        counts
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(MetadataValue::as_str)
    }

    pub fn metadata_i64(&self, key: &str) -> Option<i64> {
        self.metadata.get(key).and_then(MetadataValue::as_i64)
    }

    pub fn metadata_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(MetadataValue::as_f64)
    }
}

fn tensor_nbytes(name: &str, dimensions: &[u64], tensor_type: TensorType) -> Result<u64> {
    let (block_size, type_size) = tensor_type.layout().ok_or_else(|| {
        Error::UnsupportedTensorType {
            name: name.to_string(),
            type_id: tensor_type.id(),
            type_name: tensor_type.name(),
        }
    })?;
    let first_dim = *dimensions.first().unwrap_or(&1);
    if first_dim % block_size != 0 {
        return Err(Error::InvalidGguf(format!(
            "tensor {name} first dimension {first_dim} is not divisible by block size {block_size}"
        )));
    }
    let mut elements = 1u64;
    for dim in dimensions {
        elements = elements.checked_mul(*dim).ok_or_else(|| {
            Error::InvalidGguf(format!("tensor {name} element count overflow"))
        })?;
    }
    elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| Error::InvalidGguf(format!("tensor {name} byte size overflow")))
}

fn align_to(value: u64, alignment: u64) -> Result<u64> {
    let add = alignment - 1;
    value
        .checked_add(add)
        .map(|v| v & !add)
        .ok_or_else(|| Error::InvalidGguf("alignment overflow".into()))
}

fn read_value(cur: &mut Cursor<'_>) -> Result<MetadataValue> {
    let ty = cur.read_i32()?;
    read_value_of_type(cur, ty)
}

fn read_value_of_type(cur: &mut Cursor<'_>, ty: i32) -> Result<MetadataValue> {
    Ok(match ty {
        0 => MetadataValue::U8(cur.read_u8()?),
        1 => MetadataValue::I8(cur.read_i8()?),
        2 => MetadataValue::U16(cur.read_u16()?),
        3 => MetadataValue::I16(cur.read_i16()?),
        4 => MetadataValue::U32(cur.read_u32()?),
        5 => MetadataValue::I32(cur.read_i32()?),
        6 => MetadataValue::F32(cur.read_f32()?),
        7 => MetadataValue::Bool(cur.read_bool()?),
        8 => MetadataValue::String(cur.read_string()?),
        9 => {
            let element_ty = cur.read_i32()?;
            if element_ty == 9 {
                return Err(Error::UnsupportedGguf("nested metadata arrays".into()));
            }
            let len = cur.read_u64()?;
            if len > 1_000_000 {
                return Err(Error::InvalidGguf(format!("metadata array too large: {len}")));
            }
            let mut values = Vec::with_capacity(len as usize);
            for _ in 0..len {
                values.push(read_value_of_type(cur, element_ty)?);
            }
            MetadataValue::Array(values)
        }
        10 => MetadataValue::U64(cur.read_u64()?),
        11 => MetadataValue::I64(cur.read_i64()?),
        12 => MetadataValue::F64(cur.read_f64()?),
        other => return Err(Error::UnsupportedGguf(format!("metadata value type {other}"))),
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    path: PathBuf,
    pos: u64,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], path: PathBuf) -> Self {
        Self {
            bytes,
            path,
            pos: 0,
        }
    }

    fn read_exact_into(&mut self, out: &mut [u8]) -> Result<()> {
        let start = self.pos as usize;
        let end = start.checked_add(out.len()).ok_or_else(|| {
            Error::InvalidGguf("cursor overflow".into())
        })?;
        let src = self.bytes.get(start..end).ok_or_else(|| {
            if end > self.bytes.len() {
                Error::InvalidGguf("unexpected end of file".into())
            } else {
                Error::Io {
                    path: self.path.clone(),
                    source: std::io::Error::from(ErrorKind::UnexpectedEof),
                }
            }
        })?;
        out.copy_from_slice(src);
        self.pos = end as u64;
        Ok(())
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0; n];
        self.read_exact_into(&mut out)?;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut b = [0; 1];
        self.read_exact_into(&mut b)?;
        Ok(b[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.read_exact_into(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_i16(&mut self) -> Result<i16> {
        let mut b = [0; 2];
        self.read_exact_into(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.read_exact_into(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn read_i32(&mut self) -> Result<i32> {
        let mut b = [0; 4];
        self.read_exact_into(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }
    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.read_exact_into(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn read_i64(&mut self) -> Result<i64> {
        let mut b = [0; 8];
        self.read_exact_into(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }
    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > 16 * 1024 * 1024 {
            return Err(Error::InvalidGguf(format!("string too large: {len}")));
        }
        let bytes = self.read_bytes(len as usize)?;
        String::from_utf8(bytes)
            .map_err(|_| Error::InvalidGguf("invalid UTF-8 string".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_ids_match_ggml() {
        assert_eq!(TensorType::from_id(0), TensorType::F32);
        assert_eq!(TensorType::from_id(1), TensorType::F16);
        assert_eq!(TensorType::from_id(8), TensorType::Q8_0);
        assert_eq!(TensorType::from_id(12), TensorType::Q4K);
        assert_eq!(TensorType::from_id(13), TensorType::Q5K);
        assert_eq!(TensorType::from_id(14), TensorType::Q6K);
        assert_eq!(TensorType::from_id(99), TensorType::Unknown(99));
    }

    #[test]
    fn block_layouts_match_ggml_common_h() {
        assert_eq!(TensorType::F32.layout(), Some((1, 4)));
        assert_eq!(TensorType::F16.layout(), Some((1, 2)));
        assert_eq!(TensorType::Q8_0.layout(), Some((32, 34)));
        assert_eq!(TensorType::Q4K.layout(), Some((256, 144)));
        assert_eq!(TensorType::Q5K.layout(), Some((256, 176)));
        assert_eq!(TensorType::Q6K.layout(), Some((256, 210)));
    }
}
