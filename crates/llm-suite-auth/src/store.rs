//! Credential storage contract.
//!
//! `modify` is the only write path: every mutation is a serialized
//! read-modify-write, and OAuth token refresh runs inside it so concurrent
//! requests cannot double-refresh a rotated token.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::credential::{Credential, CredentialInfo};
use crate::error::AuthResult;

/// Boxed future returned by a [`ModifyFn`] closure.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Type-erased read-modify-write closure. Sees the current credential because
/// correct writes (refresh, re-login) depend on it; returning `None` leaves
/// the entry unchanged. The read view is only valid inside the closure body,
/// so implementations that need it across awaits must clone first.
pub type ModifyFn<'a> = Box<
    dyn FnOnce(Option<&Credential>) -> BoxFuture<'a, AuthResult<Option<Credential>>>
        + Send
        + 'a,
>;

/// Application-owned credential storage, keyed by provider id.
///
/// Implementations must not resolve secrets while listing, and must reject
/// only on storage failure: `read` resolves `None` for missing entries.
#[async_trait]
pub trait CredentialStore: Send + Sync + fmt::Debug {
    /// Read the stored credential, possibly expired. For display and status;
    /// resolved request auth comes from the auth layer.
    async fn read(&self, provider_id: &str) -> AuthResult<Option<Credential>>;

    /// List stored credential metadata without resolving or exposing secrets.
    async fn list(&self) -> AuthResult<Vec<CredentialInfo>>;

    /// Serialized write; the only mutation path. Resolves with the post-write
    /// credential (`None` when the entry was removed or left unchanged).
    async fn modify(&self, provider_id: &str, f: ModifyFn<'_>) -> AuthResult<Option<Credential>>;

    /// Remove a credential (logout). Serialized against `modify`.
    async fn delete(&self, provider_id: &str) -> AuthResult<()>;
}

/// In-memory store for tests and hosts that inject their own persistence.
#[derive(Debug, Default, Clone)]
pub struct MemoryCredentialStore {
    inner: Arc<Mutex<HashMap<String, Credential>>>,
}

impl MemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl CredentialStore for MemoryCredentialStore {
    async fn read(&self, provider_id: &str) -> AuthResult<Option<Credential>> {
        Ok(self.inner.lock().await.get(provider_id).cloned())
    }

    async fn list(&self) -> AuthResult<Vec<CredentialInfo>> {
        let guard = self.inner.lock().await;
        let mut infos: Vec<CredentialInfo> = guard
            .iter()
            .map(|(id, cred)| cred.info(id.clone()))
            .collect();
        infos.sort_by(|a, b| a.provider_id.cmp(&b.provider_id));
        Ok(infos)
    }

    async fn modify(&self, provider_id: &str, f: ModifyFn<'_>) -> AuthResult<Option<Credential>> {
        let mut guard = self.inner.lock().await;
        let current = guard.get(provider_id).cloned();
        let next = f(current.as_ref()).await?;
        match next {
            Some(cred) => {
                guard.insert(provider_id.to_string(), cred.clone());
                Ok(Some(cred))
            }
            None => Ok(guard.get(provider_id).cloned()),
        }
    }

    async fn delete(&self, provider_id: &str) -> AuthResult<()> {
        self.inner.lock().await.remove(provider_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::ApiKeyCredential;

    fn api_key(key: &str) -> Credential {
        Credential::ApiKey(ApiKeyCredential {
            key: Some(key.to_string()),
            env: Default::default(),
        })
    }

    #[tokio::test]
    async fn modify_writes_reads_and_preserves_on_none() {
        let store = MemoryCredentialStore::new();
        assert!(store.read("p").await.expect("read").is_none());

        let written = store
            .modify("p", Box::new(|_| Box::pin(async { Ok(Some(api_key("k1"))) })))
            .await
            .expect("write");
        assert_eq!(written.expect("post-write").as_api_key().expect("key").key.as_deref(), Some("k1"));

        let unchanged = store
            .modify("p", Box::new(|_| Box::pin(async { Ok(None) })))
            .await
            .expect("no-op write");
        assert_eq!(unchanged.expect("kept").as_api_key().expect("key").key.as_deref(), Some("k1"));

        store.delete("p").await.expect("delete");
        assert!(store.read("p").await.expect("read").is_none());
    }

    #[tokio::test]
    async fn modify_closure_sees_current_credential() {
        let store = MemoryCredentialStore::new();
        store
            .modify("p", Box::new(|_| Box::pin(async { Ok(Some(api_key("k1"))) })))
            .await
            .expect("seed");

        let rotated = store
            .modify("p", Box::new(|current| {
                let old = current.and_then(|c| match c {
                    Credential::ApiKey(a) => a.key.clone(),
                    _ => None,
                });
                Box::pin(async move {
                    assert_eq!(old.as_deref(), Some("k1"));
                    Ok(Some(api_key("k2")))
                })
            }))
            .await
            .expect("rotate");
        assert_eq!(rotated.expect("post-write").as_api_key().expect("key").key.as_deref(), Some("k2"));
    }

    #[tokio::test]
    async fn list_reports_metadata_only() {
        let store = MemoryCredentialStore::new();
        store
            .modify("p", Box::new(|_| Box::pin(async { Ok(Some(api_key("k1"))) })))
            .await
            .expect("seed");
        let infos = store.list().await.expect("list");
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].provider_id, "p");
        assert_eq!(infos[0].method, AuthMethod::ApiKey);
        let dumped = format!("{:?}", infos);
        assert!(!dumped.contains("k1"), "listing must not expose secrets");
    }

    use crate::credential::AuthMethod;
}
