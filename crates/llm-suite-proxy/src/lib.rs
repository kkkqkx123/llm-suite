//! Shared HTTP plumbing for the llm-suite clients.
//!
//! Every client that accepts a proxy URL funnels it through [`build_proxy`],
//! and every reqwest client is assembled through [`build_http_client`], so
//! the accepted schemes, the bypass list, timeouts and the failure messages
//! behave identically across the suite instead of drifting per crate.
//!
//! Three rules are deliberate:
//!
//! - Error messages never contain the configured URL: a proxy URL commonly
//!   carries credentials in its userinfo part, which must not reach logs or
//!   operator-facing error strings.
//! - A bypass list lives next to the proxy instead of being inferred, so a
//!   host decides locally which of its traffic is proxied.
//! - Proxy selection is explicit configuration first, environment second,
//!   direct last: a configured URL wins; without one the client keeps the
//!   reqwest default and honors the standard `ALL_PROXY` / `HTTPS_PROXY` /
//!   `HTTP_PROXY` / `NO_PROXY` environment variables; a configured URL
//!   disables the environment for that client. This module never re-reads
//!   the environment itself, so the precedence stays the reqwest default
//!   instead of a second hand-rolled copy.

use thiserror::Error;

/// Proxy schemes reqwest can actually honor.
const SUPPORTED_SCHEMES: [&str; 6] = ["http", "https", "socks4", "socks4a", "socks5", "socks5h"];

/// A configured proxy URL that cannot be turned into a usable proxy.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProxyError {
    #[error(
        "unsupported proxy scheme '{scheme}'; expected one of http, https, socks4, socks4a, socks5, socks5h"
    )]
    UnsupportedScheme { scheme: String },
    #[error("proxy URL has no host")]
    MissingHost,
    #[error("proxy URL could not be parsed")]
    Unparseable,
    #[error("proxy URL must not embed credentials; store them as a proxy credential instead")]
    UserinfoForbidden,
    #[error("{0}")]
    ClientBuild(String),
}

/// True when the configured route address embeds `userinfo` credentials.
///
/// Persisted profiles and definitions must be route-only; secrets live in
/// the credential store under a key derived from the route. The composed
/// in-memory address is the only place userinfo may appear.
pub fn has_userinfo(url: &str) -> bool {
    let trimmed = url.trim();
    let Some((_, rest)) = trimmed.split_once("://") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    authority.contains('@')
}

/// Reject route addresses that embed credentials, for persisted configs.
///
/// The error carries no address material so credentials never reach logs.
pub fn reject_userinfo(url: &str) -> Result<(), ProxyError> {
    if has_userinfo(url) {
        return Err(ProxyError::UserinfoForbidden);
    }
    Ok(())
}

/// Credential store key for the proxy at `route`.
///
/// The key is routing material only, safe to list and compare. One route
/// shares one credential across every profile pointing at it.
pub fn credential_key(route: &str) -> String {
    format!("proxy:{}", route.trim())
}

/// Compose the in-memory proxy address from a route plus stored credentials.
///
/// Percent-encodes the username and password so reserved characters cannot
/// break the authority section. The result is transient: callers pass it
/// to [`build_proxy`] and never persist it.
pub fn compose_proxy_url(
    route: &str,
    username: &str,
    password: &str,
) -> Result<String, ProxyError> {
    let trimmed = route.trim();
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err(ProxyError::Unparseable);
    };
    if rest.is_empty() {
        return Err(ProxyError::MissingHost);
    }
    if has_userinfo(trimmed) {
        return Err(ProxyError::UserinfoForbidden);
    }
    Ok(format!(
        "{}://{}:{}@{}",
        scheme,
        encode_userinfo(username),
        encode_userinfo(password),
        rest
    ))
}

fn encode_userinfo(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Builds a reqwest proxy from a URL plus an optional bypass list.
///
/// `no_proxy` entries are domain names (with or without a leading dot), CIDR
/// blocks, or `*`, and are handed to reqwest for matching, so their
/// semantics match the standard `NO_PROXY` spelling.
pub fn build_proxy(url: &str, no_proxy: &[String]) -> Result<reqwest::Proxy, ProxyError> {
    let trimmed = url.trim();
    let (scheme, rest) = trimmed.split_once("://").ok_or(ProxyError::Unparseable)?;
    let scheme = scheme.to_ascii_lowercase();
    if !SUPPORTED_SCHEMES.contains(&scheme.as_str()) {
        return Err(ProxyError::UnsupportedScheme { scheme });
    }
    if host_of(rest).is_none() {
        return Err(ProxyError::MissingHost);
    }

    let mut proxy = reqwest::Proxy::all(trimmed).map_err(|_| ProxyError::Unparseable)?;
    let entries: Vec<&str> = no_proxy
        .iter()
        .map(String::as_str)
        .filter(|entry| !entry.trim().is_empty())
        .collect();
    if !entries.is_empty() {
        proxy = proxy.no_proxy(reqwest::NoProxy::from_string(&entries.join(",")));
    }
    Ok(proxy)
}

/// Connection-level transport tuning consumed by [`build_http_client_with_options`].
///
/// Kept as primitives (not the `llm-suite-types` struct) so this leaf crate
/// stays dependency-free; the gateway maps `LlmHttpTransport` onto it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HttpTransportOptions {
    /// TCP/TLS connect timeout in seconds; `None` keeps the reqwest default.
    pub connect_timeout_secs: Option<u64>,
    /// Accept invalid TLS certificates (self-signed local gateways). An
    /// explicit opt-in; the gateway logs a warning when it builds such a
    /// client.
    pub danger_accept_invalid_certs: bool,
}

/// Shared reqwest client construction for the suite's HTTP clients: applies
/// the timeout, wires the proxy (with its bypass list) when configured, and
/// builds the client. Proxy failures keep [`ProxyError`] semantics so the
/// credential-hiding rules above hold everywhere.
pub fn build_http_client(
    timeout_secs: u64,
    proxy: Option<&str>,
    no_proxy: &[String],
) -> Result<reqwest::Client, ProxyError> {
    build_http_client_with_options(
        timeout_secs,
        proxy,
        no_proxy,
        &HttpTransportOptions::default(),
    )
}

/// [`build_http_client`] plus the Phase 4 transport knobs: an optional
/// connect timeout and the invalid-certificate opt-in. Proxy handling and
/// error semantics are identical; only connection-affecting fields live
/// here so the transport cache key can cover them exactly.
pub fn build_http_client_with_options(
    timeout_secs: u64,
    proxy: Option<&str>,
    no_proxy: &[String],
    options: &HttpTransportOptions,
) -> Result<reqwest::Client, ProxyError> {
    let mut builder =
        reqwest::Client::builder().timeout(std::time::Duration::from_secs(timeout_secs.max(1)));
    if let Some(connect_secs) = options.connect_timeout_secs {
        builder = builder.connect_timeout(std::time::Duration::from_secs(connect_secs.max(1)));
    }
    if options.danger_accept_invalid_certs {
        builder = builder.danger_accept_invalid_certs(true);
    }
    if let Some(proxy_url) = proxy {
        builder = builder.proxy(build_proxy(proxy_url, no_proxy)?);
    }
    builder.build().map_err(|err| {
        // reqwest build errors never carry the proxy URL, so rendering them
        // through Unparseable-style text stays credential-free.
        ProxyError::ClientBuild(format!("failed to build HTTP client: {err}"))
    })
}

/// Host of the authority part, ignoring userinfo, port and path.
fn host_of(after_scheme: &str) -> Option<&str> {
    let authority = after_scheme.split(['/', '?', '#']).next()?;
    let host_port = authority.rsplit('@').next()?;
    let host = match host_port.strip_prefix('[') {
        Some(rest) => rest.split(']').next()?,
        None => host_port.split(':').next()?,
    };
    (!host.is_empty()).then_some(host)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_every_scheme_reqwest_supports() {
        for url in [
            "http://127.0.0.1:7890",
            "https://proxy.example.com",
            "socks4://127.0.0.1:1080",
            "socks4a://127.0.0.1:1080",
            "socks5://127.0.0.1:1080",
            "socks5h://127.0.0.1:1080",
            "http://user:secret@proxy.example.com:3128",
            "  http://127.0.0.1:7890  ",
        ] {
            assert!(build_proxy(url, &[]).is_ok(), "{url} must be accepted");
        }
    }

    #[test]
    fn rejects_urls_without_a_usable_host_or_scheme() {
        assert_eq!(
            build_proxy("127.0.0.1:7890", &[]).err(),
            Some(ProxyError::Unparseable)
        );
        assert_eq!(
            build_proxy("ftp://127.0.0.1:7890", &[]).err(),
            Some(ProxyError::UnsupportedScheme {
                scheme: "ftp".to_string()
            })
        );
        assert_eq!(
            build_proxy("HTTP://proxy.example.com", &[]).err(),
            None,
            "the scheme check is case-insensitive"
        );
        for url in ["http://", "http:///v1", "http://user@:3128"] {
            assert_eq!(
                build_proxy(url, &[]).err(),
                Some(ProxyError::MissingHost),
                "{url} has no host"
            );
        }
    }

    #[test]
    fn errors_never_leak_the_configured_url() {
        let error = build_proxy("ftp://user:secret@proxy.example.com:21", &[])
            .expect_err("unsupported scheme must fail");
        let rendered = format!("{error}");
        assert!(rendered.contains("ftp"));
        assert!(
            !rendered.contains("secret") && !rendered.contains("proxy.example.com"),
            "credentials and host must stay out of the message: {rendered}"
        );
    }

    #[test]
    fn transport_options_build_a_client() {
        let options = HttpTransportOptions {
            connect_timeout_secs: Some(2),
            danger_accept_invalid_certs: false,
        };
        assert!(
            build_http_client_with_options(60, None, &[], &options).is_ok(),
            "connect timeout must build"
        );
        let insecure = HttpTransportOptions {
            connect_timeout_secs: None,
            danger_accept_invalid_certs: true,
        };
        assert!(
            build_http_client_with_options(60, None, &[], &insecure).is_ok(),
            "invalid-certs opt-in must build"
        );
        // Defaults stay backward compatible with the plain constructor.
        assert!(build_http_client(60, None, &[]).is_ok());
    }

    #[test]
    fn userinfo_detection_guards_persisted_routes() {
        assert!(!has_userinfo("http://127.0.0.1:7890"));
        assert!(has_userinfo("http://user:secret@127.0.0.1:7890"));
        assert!(reject_userinfo("http://127.0.0.1:7890").is_ok());
        assert_eq!(
            reject_userinfo("http://user:secret@127.0.0.1:7890").err(),
            Some(ProxyError::UserinfoForbidden)
        );
        assert_eq!(
            credential_key("http://127.0.0.1:7890"),
            "proxy:http://127.0.0.1:7890"
        );
    }

    #[test]
    fn composed_address_carries_credentials_transiently() {
        let composed =
            compose_proxy_url("http://127.0.0.1:7890", "corp-user", "s3cret").expect("compose");
        assert_eq!(composed, "http://corp-user:s3cret@127.0.0.1:7890");
        assert!(build_proxy(&composed, &[]).is_ok());
        let encoded =
            compose_proxy_url("http://127.0.0.1:7890", "u@corp", "p:ss").expect("compose");
        assert_eq!(encoded, "http://u%40corp:p%3Ass@127.0.0.1:7890");
        assert_eq!(
            compose_proxy_url("http://user:pw@127.0.0.1:7890", "a", "b").err(),
            Some(ProxyError::UserinfoForbidden)
        );
    }

    #[test]
    fn bypass_entries_are_forwarded_and_blanks_dropped() {
        assert!(build_proxy("http://127.0.0.1:7890", &[]).is_ok());
        assert!(build_proxy(
            "http://127.0.0.1:7890",
            &["localhost".to_string(), "  ".to_string()],
        )
        .is_ok());
        assert!(build_proxy(
            "http://127.0.0.1:7890",
            &[
                ".internal.example.com".to_string(),
                "10.0.0.0/8".to_string()
            ],
        )
        .is_ok());
    }
}
