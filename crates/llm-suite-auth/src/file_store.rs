//! JSON file credential store.
//!
//! One file holds a map of provider id to credential. Writes are atomic
//! (write to a temporary file, then rename) and the file is created with
//! owner-only permissions where the platform supports it. A process-level
//! lock serializes readers and writers inside one process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::credential::{Credential, CredentialInfo};
use crate::error::{AuthError, AuthResult};
use crate::store::{CredentialStore, ModifyFn};

/// File-backed credential store.
#[derive(Debug)]
pub struct FileCredentialStore {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
}

impl FileCredentialStore {
    /// Open (or lazily create) the store at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: Arc::new(Mutex::new(())),
        }
    }

    /// The backing file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    async fn load(&self) -> AuthResult<HashMap<String, Credential>> {
        match tokio::fs::read_to_string(&self.path).await {
            Ok(text) => {
                if text.trim().is_empty() {
                    return Ok(HashMap::new());
                }
                serde_json::from_str(&text)
                    .map_err(|e| AuthError::Store(format!("{}: {e}", self.path.display())))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(HashMap::new()),
            Err(e) => Err(AuthError::Store(format!(
                "reading {}: {e}",
                self.path.display()
            ))),
        }
    }

    async fn save(&self, map: &HashMap<String, Credential>) -> AuthResult<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| AuthError::Store(format!("creating directory: {e}")))?;
            }
        }
        let text = serde_json::to_string_pretty(map)
            .map_err(|e| AuthError::Store(format!("serializing credentials: {e}")))?;
        let tmp = self.path.with_extension("json.tmp");
        tokio::fs::write(&tmp, text)
            .await
            .map_err(|e| AuthError::Store(format!("writing credentials: {e}")))?;
        restrict_permissions(&tmp);
        tokio::fs::rename(&tmp, &self.path)
            .await
            .map_err(|e| AuthError::Store(format!("replacing credentials: {e}")))?;
        restrict_permissions(&self.path);
        Ok(())
    }
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(metadata) = std::fs::metadata(path) {
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o600);
        let _ = std::fs::set_permissions(path, permissions);
    }
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) {}

#[async_trait]
impl CredentialStore for FileCredentialStore {
    async fn read(&self, provider_id: &str) -> AuthResult<Option<Credential>> {
        let _guard = self.lock.lock().await;
        Ok(self.load().await?.remove(provider_id))
    }

    async fn list(&self) -> AuthResult<Vec<CredentialInfo>> {
        let _guard = self.lock.lock().await;
        let mut infos: Vec<CredentialInfo> = self
            .load()
            .await?
            .into_iter()
            .map(|(id, cred)| cred.info(id))
            .collect();
        infos.sort_by(|a, b| a.provider_id.cmp(&b.provider_id));
        Ok(infos)
    }

    async fn modify(&self, provider_id: &str, f: ModifyFn<'_>) -> AuthResult<Option<Credential>> {
        let _guard = self.lock.lock().await;
        let mut map = self.load().await?;
        let current = map.get(provider_id).cloned();
        let next = f(current.as_ref()).await?;
        let result = match next {
            Some(cred) => {
                map.insert(provider_id.to_string(), cred.clone());
                Some(cred)
            }
            None => map.get(provider_id).cloned(),
        };
        self.save(&map).await?;
        Ok(result)
    }

    async fn delete(&self, provider_id: &str) -> AuthResult<()> {
        let _guard = self.lock.lock().await;
        let mut map = self.load().await?;
        if map.remove(provider_id).is_some() {
            self.save(&map).await?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "file-store"))]
mod tests {
    use super::*;
    use crate::credential::{ApiKeyCredential, AuthMethod};

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llm-suite-auth-{}-{}.json", tag, std::process::id()))
    }

    #[tokio::test]
    async fn roundtrip_and_atomic_replace() {
        let path = temp_path("roundtrip");
        let _ = std::fs::remove_file(&path);
        let store = FileCredentialStore::new(&path);
        assert!(store.read("acme").await.expect("read").is_none());

        let written = store
            .modify(
                "acme",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("sk-file".to_string()),
                            env: Default::default(),
                        })))
                    })
                }),
            )
            .await
            .expect("write");
        assert_eq!(
            written.expect("post-write").as_api_key().expect("key").key.as_deref(),
            Some("sk-file")
        );

        let reread = store.read("acme").await.expect("read").expect("stored");
        assert_eq!(reread.method(), AuthMethod::ApiKey);
        assert!(!path.with_extension("json.tmp").exists(), "temp file must be renamed away");

        store.delete("acme").await.expect("delete");
        assert!(store.read("acme").await.expect("read").is_none());
        let _ = std::fs::remove_file(&path);
    }
}
