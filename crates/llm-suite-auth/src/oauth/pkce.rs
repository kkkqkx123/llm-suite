//! OAuth 2.0 authorization code flow with PKCE (RFC 7636).
//!
//! Supports both the loopback-redirect mode (a local listener receives the
//! callback) and the paste mode (the user copies the callback URL). The
//! provider supplies endpoints, client id and redirect URI.

use std::sync::Arc;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::Rng;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::credential::OAuthCredential;
use crate::error::{AuthError, AuthResult};
use crate::oauth::{credential_from_token, form_body, TokenResponse};

/// A PKCE verifier/challenge pair.
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
}

/// Generate a fresh PKCE verifier and its S256 challenge.
pub fn new_challenge() -> PkceChallenge {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let verifier = URL_SAFE_NO_PAD.encode(bytes);
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(digest);
    PkceChallenge {
        verifier,
        challenge,
    }
}

/// Configuration for one PKCE provider.
#[derive(Debug, Clone)]
pub struct PkceConfig {
    pub authorize_url: String,
    pub token_url: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: Option<String>,
    /// Bind a loopback listener on the redirect port and wait for the
    /// callback instead of asking the user to paste the URL.
    pub loopback: bool,
    /// Whether access through this flow is backed by a subscription.
    pub is_subscription: bool,
}

/// A started PKCE flow awaiting the authorization code.
#[derive(Debug)]
pub struct PkcePending {
    config: PkceConfig,
    verifier: String,
    state: String,
    listener: Option<LoopbackListener>,
}

/// A bound loopback listener serving exactly one callback request.
#[derive(Debug)]
pub struct LoopbackListener {
    listener: TcpListener,
    state: Arc<String>,
}

impl LoopbackListener {
    /// The port the listener is bound to.
    pub fn port(&self) -> u16 {
        self.listener.local_addr().map(|a| a.port()).unwrap_or(0)
    }

    /// Wait for the callback request, validate the state parameter and
    /// return the authorization code.
    pub async fn wait_for_code(self) -> AuthResult<String> {
        let (mut stream, _peer) = self
            .listener
            .accept()
            .await
            .map_err(|e| AuthError::Loopback(e.to_string()))?;
        let mut buffer = [0u8; 8192];
        let read = stream
            .read(&mut buffer)
            .await
            .map_err(|e| AuthError::Loopback(e.to_string()))?;
        let request = String::from_utf8_lossy(&buffer[..read]).to_string();
        let request_line = request.lines().next().unwrap_or_default().to_string();
        let path = request_line
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_string();
        let query = path.split_once('?').map(|(_, q)| q).unwrap_or_default();
        let params = parse_query(query);
        let body = if let Some(error) = params.get("error") {
            format!("Authorization failed: {error}")
        } else if params.get("state").map(String::as_str) == Some(self.state.as_str()) {
            "Authorization complete. You can close this window.".to_string()
        } else {
            "Authorization state mismatch.".to_string()
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .await
            .map_err(|e| AuthError::Loopback(e.to_string()))?;
        let _ = stream.flush().await;
        if let Some(error) = params.get("error") {
            return Err(AuthError::Login(format!(
                "authorization denied: {error}"
            )));
        }
        if params.get("state").map(String::as_str) != Some(self.state.as_str()) {
            return Err(AuthError::Login("authorization state mismatch".to_string()));
        }
        params
            .get("code")
            .cloned()
            .ok_or_else(|| AuthError::Login("callback carried no authorization code".to_string()))
    }
}

/// Parse a URL query string into pairs.
fn parse_query(query: &str) -> std::collections::HashMap<String, String> {
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((percent_decode(k), percent_decode(v)))
        })
        .collect()
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                if let Some(value) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    out.push(value);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Start a PKCE flow. When `loopback` is set, a local listener is bound and
/// the redirect URI's port is rewritten to the bound port.
pub async fn start(config: PkceConfig) -> AuthResult<PkcePending> {
    let challenge = new_challenge();
    let mut state_bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut state_bytes);
    let state = URL_SAFE_NO_PAD.encode(state_bytes);
    let listener = if config.loopback {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| AuthError::Loopback(e.to_string()))?;
        Some(LoopbackListener {
            listener,
            state: Arc::new(state.clone()),
        })
    } else {
        None
    };
    Ok(PkcePending {
        config,
        verifier: challenge.verifier,
        state,
        listener,
    })
}

impl PkcePending {
    /// The authorization URL the user must open.
    pub fn authorize_url(&self) -> String {
        let mut url = url::Url::parse(&self.config.authorize_url)
            .map_err(|e| AuthError::Transport(e.to_string()))
            .expect("authorize url parses");
        {
            let mut pairs = url.query_pairs_mut();
            pairs.append_pair("response_type", "code");
            pairs.append_pair("client_id", &self.config.client_id);
            pairs.append_pair("redirect_uri", &self.redirect_uri());
            pairs.append_pair("state", &self.state);
            if let Some(scope) = self.config.scope.as_deref() {
                pairs.append_pair("scope", scope);
            }
        }
        url.query_pairs_mut().append_pair("code_challenge", &self.challenge());
        url.query_pairs_mut()
            .append_pair("code_challenge_method", "S256");
        url.to_string()
    }

    fn challenge(&self) -> String {
        let digest = Sha256::digest(self.verifier.as_bytes());
        URL_SAFE_NO_PAD.encode(digest)
    }

    /// The effective redirect URI: the configured one, or the bound loopback
    /// port when running in loopback mode.
    pub fn redirect_uri(&self) -> String {
        match &self.listener {
            Some(listener) => rewrite_port(&self.config.redirect_uri, listener.port()),
            None => self.config.redirect_uri.clone(),
        }
    }

    /// Take the loopback listener, if any.
    pub fn take_listener(&mut self) -> Option<LoopbackListener> {
        self.listener.take()
    }

    /// Exchange an authorization code for a credential.
    pub async fn exchange_code(
        &self,
        code: &str,
        client: &reqwest::Client,
        now_ms: i64,
    ) -> AuthResult<OAuthCredential> {
        let redirect_uri = self.redirect_uri();
        let form = [
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri.as_str()),
            ("client_id", self.config.client_id.as_str()),
            ("code_verifier", self.verifier.as_str()),
        ];
        let response = client
            .post(&self.config.token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(form_body(&form))
            .send()
            .await
            .map_err(|e| AuthError::Transport(e.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| AuthError::Transport(e.to_string()))?;
        let token: TokenResponse = serde_json::from_str(&body).map_err(|e| {
            AuthError::Transport(format!("token response: {e} (HTTP {})", status.as_u16()))
        })?;
        credential_from_token(&token, self.config.is_subscription, now_ms)
    }

    /// Extract the authorization code from a pasted callback URL.
    pub fn extract_code_from_paste(input: &str) -> Option<String> {
        let query = input.split_once('?').map(|(_, q)| q)?;
        parse_query(query).get("code").cloned()
    }
}

/// Replace the port component of a URL, keeping scheme, host and path.
fn rewrite_port(redirect_uri: &str, port: u16) -> String {
    match url::Url::parse(redirect_uri) {
        Ok(mut url) => {
            let _ = url.set_port(Some(port));
            url.to_string()
        }
        Err(_) => redirect_uri.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_is_base64url_and_matches_verifier() {
        let challenge = new_challenge();
        assert!(!challenge.verifier.is_empty());
        assert!(challenge
            .verifier
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        let digest = Sha256::digest(challenge.verifier.as_bytes());
        assert_eq!(challenge.challenge, URL_SAFE_NO_PAD.encode(digest));
    }

    #[test]
    fn extract_code_from_pasted_url() {
        let pasted = "http://127.0.0.1:8080/callback?code=abc123&state=xyz";
        assert_eq!(
            PkcePending::extract_code_from_paste(pasted).as_deref(),
            Some("abc123")
        );
        assert_eq!(PkcePending::extract_code_from_paste("http://127.0.0.1:8080/callback"), None);
    }

    #[test]
    fn query_decoding_handles_percent_and_plus() {
        let params = parse_query("code=a%20b&state=x+y");
        assert_eq!(params.get("code").map(String::as_str), Some("a b"));
        assert_eq!(params.get("state").map(String::as_str), Some("x y"));
    }

    #[tokio::test]
    async fn authorize_url_carries_pkce_params() {
        let pending = start(PkceConfig {
            authorize_url: "https://example.test/oauth/authorize".to_string(),
            token_url: "https://example.test/oauth/token".to_string(),
            client_id: "cid".to_string(),
            redirect_uri: "http://127.0.0.1:8080/callback".to_string(),
            scope: Some("api".to_string()),
            loopback: false,
            is_subscription: true,
        })
        .await
        .expect("start");
        let url = pending.authorize_url();
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=cid"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("scope=api"));
        assert!(url.contains(&format!("state={}", pending.state)));
    }

    #[tokio::test]
    async fn loopback_listener_serves_callback_and_validates_state() {
        let mut pending = start(PkceConfig {
            authorize_url: "https://example.test/oauth/authorize".to_string(),
            token_url: "https://example.test/oauth/token".to_string(),
            client_id: "cid".to_string(),
            redirect_uri: "http://127.0.0.1:8080/callback".to_string(),
            scope: None,
            loopback: true,
            is_subscription: false,
        })
        .await
        .expect("start");
        let port = pending.listener.as_ref().expect("listener").port();
        assert!(port > 0);
        assert!(pending.redirect_uri().contains(&format!(":{port}/callback")));

        let listener = pending.take_listener().expect("listener");
        let state = listener.state.clone();
        let client = reqwest::Client::new();
        let handle = tokio::spawn(async move { listener.wait_for_code().await });
        let _ = client
            .get(format!(
                "http://127.0.0.1:{port}/callback?code=the-code&state={state}"
            ))
            .send()
            .await
            .expect("callback request");
        let code = handle.await.expect("join").expect("code");
        assert_eq!(code, "the-code");
    }

    #[tokio::test]
    async fn loopback_rejects_state_mismatch() {
        let mut pending = start(PkceConfig {
            authorize_url: "https://example.test/oauth/authorize".to_string(),
            token_url: "https://example.test/oauth/token".to_string(),
            client_id: "cid".to_string(),
            redirect_uri: "http://127.0.0.1:8080/callback".to_string(),
            scope: None,
            loopback: true,
            is_subscription: false,
        })
        .await
        .expect("start");
        let listener = pending.take_listener().expect("listener");
        let port = listener.port();
        let handle = tokio::spawn(async move { listener.wait_for_code().await });
        let _ = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/callback?code=c&state=wrong"))
            .send()
            .await
            .expect("callback request");
        let err = handle.await.expect("join").expect_err("must fail");
        assert!(matches!(err, AuthError::Login(_)));
    }
}
