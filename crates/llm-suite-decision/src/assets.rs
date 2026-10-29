//! Jev/TypeSafe image assets, resolution, and multipart encoding.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use regex::Regex;
use uuid::Uuid;

use crate::error::DecisionError;

pub const SIGNATURE_LENGTH: usize = 8;
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const JPEG_SIGNATURE: &[u8] = b"\xff\xd8\xff";

pub fn content_type_from_bytes(header: &[u8]) -> String {
    if header.starts_with(PNG_SIGNATURE) {
        "image/png".to_string()
    } else if header.starts_with(JPEG_SIGNATURE) {
        "image/jpeg".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedImage {
    pub content_type: String,
    pub data: Vec<u8>,
}

#[derive(Clone)]
enum ImageSource {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

#[derive(Clone)]
pub struct ImageAsset {
    source: ImageSource,
    reference: String,
    resolved: Arc<Mutex<Option<ResolvedImage>>>,
}

impl PartialEq for ImageAsset {
    fn eq(&self, other: &Self) -> bool {
        self.reference == other.reference
    }
}

impl std::fmt::Debug for ImageAsset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageAsset")
            .field("reference", &self.reference)
            .finish()
    }
}

impl ImageAsset {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::new(ImageSource::Path(path.into()))
    }

    pub fn from_bytes(data: impl Into<Vec<u8>>) -> Self {
        Self::new(ImageSource::Bytes(data.into()))
    }

    pub fn from_base64(data: impl AsRef<str>) -> Result<Self, DecisionError> {
        let text = data.as_ref();
        let cleaned: String = text.split_whitespace().collect();
        if cleaned.is_empty() || !cleaned.is_ascii() {
            return Err(DecisionError::InvalidRequest(
                "Image data must contain non-empty ASCII base64 text.".into(),
            ));
        }
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&cleaned)
            .map_err(|err| {
                DecisionError::InvalidRequest(format!(
                    "Image data must contain valid base64 text: {err}"
                ))
            })?;
        Ok(Self::new(ImageSource::Bytes(decoded)))
    }

    fn new(source: ImageSource) -> Self {
        Self {
            source,
            reference: format!(
                "<image:{}>",
                Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)).simple()
            ),
            resolved: Arc::new(Mutex::new(None)),
        }
    }

    pub fn reference(&self) -> &str {
        &self.reference
    }

    pub fn resolve(&self) -> Result<ResolvedImage, DecisionError> {
        {
            let guard = self.resolved.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(resolved) = guard.as_ref() {
                return Ok(resolved.clone());
            }
        }
        let data = match &self.source {
            ImageSource::Path(path) => std::fs::read(&path).map_err(|err| {
                DecisionError::InvalidRequest(format!(
                    "Image file {:?} could not be read: {}.",
                    path, err
                ))
            })?,
            ImageSource::Bytes(data) => data.clone(),
        };
        if data.is_empty() {
            return Err(DecisionError::InvalidRequest("Image data is empty.".into()));
        }
        let ct = content_type_from_bytes(&data[..data.len().min(SIGNATURE_LENGTH)]);
        let resolved = ResolvedImage {
            content_type: ct,
            data,
        };
        let mut guard = self.resolved.lock().unwrap_or_else(|p| p.into_inner());
        *guard = Some(resolved.clone());
        Ok(resolved)
    }

    pub async fn resolve_async(&self) -> Result<ResolvedImage, DecisionError> {
        {
            let guard = self.resolved.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(resolved) = guard.as_ref() {
                return Ok(resolved.clone());
            }
        }
        let source = self.source.clone();
        let data = tokio::task::spawn_blocking(move || match source {
            ImageSource::Path(path) => std::fs::read(&path).map_err(|err| {
                DecisionError::InvalidRequest(format!(
                    "Image file {:?} could not be read: {}.",
                    path, err
                ))
            }),
            ImageSource::Bytes(data) => Ok(data),
        })
        .await
        .map_err(|err| {
            DecisionError::InvalidRequest(format!("image resolution task failed: {err}"))
        })??;
        if data.is_empty() {
            return Err(DecisionError::InvalidRequest("Image data is empty.".into()));
        }
        let ct = content_type_from_bytes(&data[..data.len().min(SIGNATURE_LENGTH)]);
        let resolved = ResolvedImage {
            content_type: ct,
            data,
        };
        let mut guard = self.resolved.lock().unwrap_or_else(|p| p.into_inner());
        *guard = Some(resolved.clone());
        Ok(resolved)
    }
}

pub type DecisionAsset = ImageAsset;

pub fn resolve_assets(
    assets: &BTreeMap<String, ImageAsset>,
) -> Result<BTreeMap<String, ResolvedImage>, DecisionError> {
    for name in assets.keys() {
        if name.trim().is_empty() {
            return Err(DecisionError::InvalidRequest(
                "Asset aliases must be non-empty strings.".into(),
            ));
        }
    }
    let mut cache: HashMap<String, ResolvedImage> = HashMap::new();
    for (name, asset) in assets {
        if cache.contains_key(&asset.reference) {
            continue;
        }
        let resolved = asset
            .resolve()
            .map_err(|err| DecisionError::InvalidRequest(format!("Asset {name:?}: {err}")))?;
        cache.insert(asset.reference.clone(), resolved);
    }
    let mut out = BTreeMap::new();
    for (name, asset) in assets {
        out.insert(name.clone(), cache.get(&asset.reference).cloned().unwrap());
    }
    Ok(out)
}

pub async fn resolve_assets_async(
    assets: &BTreeMap<String, ImageAsset>,
) -> Result<BTreeMap<String, ResolvedImage>, DecisionError> {
    for name in assets.keys() {
        if name.trim().is_empty() {
            return Err(DecisionError::InvalidRequest(
                "Asset aliases must be non-empty strings.".into(),
            ));
        }
    }
    let mut distinct: HashMap<String, ImageAsset> = HashMap::new();
    for asset in assets.values() {
        distinct
            .entry(asset.reference.clone())
            .or_insert_with(|| asset.clone());
    }
    let mut cache: HashMap<String, ResolvedImage> = HashMap::new();
    for (reference, asset) in &distinct {
        let name_for_error = assets
            .iter()
            .find(|(_, a)| a.reference == *reference)
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| reference.clone());
        let resolved = asset.resolve_async().await.map_err(|err| {
            DecisionError::InvalidRequest(format!("Asset {name_for_error:?}: {err}"))
        })?;
        cache.insert(reference.clone(), resolved);
    }
    let mut out = BTreeMap::new();
    for (name, asset) in assets {
        out.insert(name.clone(), cache.get(&asset.reference).cloned().unwrap());
    }
    Ok(out)
}

pub fn reference_pattern() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<image:[0-9a-f]{32}>").unwrap())
}

pub fn scan_references(content: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(content);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in reference_pattern().find_iter(&text) {
        let s = m.as_str().to_string();
        if seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Part {
    pub name: String,
    pub content_type: String,
    pub content: Vec<u8>,
    pub filename: Option<String>,
}

pub fn encode_multipart(parts: &[Part]) -> (String, Vec<u8>) {
    let boundary = Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext))
        .simple()
        .to_string();
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    for part in parts {
        let mut header = format!(
            "--{}\r\nContent-Disposition: form-data; name=\"{}\"",
            boundary, part.name
        );
        if let Some(filename) = &part.filename {
            header.push_str(&format!("; filename=\"{}\"", filename));
        }
        header.push_str(&format!("\r\nContent-Type: {}\r\n\r\n", part.content_type));
        chunks.push(header.into_bytes());
        chunks.push(part.content.clone());
        chunks.push(b"\r\n".to_vec());
    }
    chunks.push(format!("--{}--\r\n", boundary).into_bytes());
    let mut body = Vec::new();
    for c in chunks {
        body.extend_from_slice(&c);
    }
    (format!("multipart/form-data; boundary={}", boundary), body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_detects_png_and_jpeg() {
        assert_eq!(
            content_type_from_bytes(b"\x89PNG\r\n\x1a\nrest"),
            "image/png"
        );
        assert_eq!(content_type_from_bytes(b"\xff\xd8\xffrest"), "image/jpeg");
        assert_eq!(
            content_type_from_bytes(b"hello"),
            "application/octet-stream"
        );
    }

    #[test]
    fn from_base64_decodes() {
        let data = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\n");
        let asset = ImageAsset::from_base64(&data).unwrap();
        let resolved = asset.resolve().unwrap();
        assert_eq!(resolved.content_type, "image/png");
    }

    #[test]
    fn from_base64_rejects_empty() {
        assert!(ImageAsset::from_base64("   ").is_err());
    }

    #[test]
    fn resolve_empty_bytes_errors() {
        let asset = ImageAsset::from_bytes(Vec::<u8>::new());
        assert!(asset.resolve().is_err());
    }

    #[test]
    fn resolve_file_error_contains_path() {
        let asset = ImageAsset::from_path("/nonexistent/path/img.png");
        let err = asset.resolve().unwrap_err().to_string();
        assert!(err.contains("could not be read"));
    }

    #[test]
    fn scan_references_finds_unique() {
        let content = b"text <image:0123456789abcdef0123456789abcdef> and <image:0123456789abcdef0123456789abcdef> <image:ffffffffffffffffffffffffffffffff>";
        let refs = scan_references(content);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn scan_references_ignores_invalid() {
        let content = b"<image:short> <image:gggggggggggggggggggggggggggggggg>";
        let refs = scan_references(content);
        assert!(refs.is_empty());
    }

    #[test]
    fn encode_multipart_structure() {
        let parts = vec![
            Part {
                name: "request".into(),
                content_type: "application/json".into(),
                content: b"{}".to_vec(),
                filename: None,
            },
            Part {
                name: "asset-0".into(),
                content_type: "image/png".into(),
                content: vec![1, 2, 3],
                filename: Some("asset-0".into()),
            },
        ];
        let (ct, body) = encode_multipart(&parts);
        assert!(ct.starts_with("multipart/form-data; boundary="));
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"request\""));
        assert!(text.contains("name=\"asset-0\""));
        assert!(text.contains("filename=\"asset-0\""));
    }

    #[tokio::test]
    async fn resolve_assets_async_propagates_alias_error() {
        let mut map = BTreeMap::new();
        map.insert("a".to_string(), ImageAsset::from_bytes(Vec::<u8>::new()));
        let err = resolve_assets_async(&map).await.unwrap_err().to_string();
        assert!(err.contains("Asset"));
    }

    #[test]
    fn resolve_caches() {
        let asset = ImageAsset::from_bytes(b"\x89PNG\r\n\x1a\n".to_vec());
        let r1 = asset.resolve().unwrap();
        let r2 = asset.resolve().unwrap();
        assert_eq!(r1.data, r2.data);
    }
}
