//! Shared HTTP plumbing for the llm-suite clients.
//!
//! Every client that accepts a proxy URL funnels it through [`build_proxy`],
//! and every reqwest client is assembled through [`build_http_client`], so
//! the accepted schemes, the bypass list, timeouts and the failure messages
//! behave identically across the suite instead of drifting per crate.
//!
//! Two rules are deliberate:
//!
//! - Error messages never contain the configured URL: a proxy URL commonly
//!   carries credentials in its userinfo part, which must not reach logs or
//!   operator-facing error strings.
//! - A bypass list lives next to the proxy instead of being inferred, so a
//!   host decides locally which of its traffic is proxied.

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
    #[error("{0}")]
    ClientBuild(String),
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

/// Shared reqwest client construction for the suite's HTTP clients: applies
/// the timeout, wires the proxy (with its bypass list) when configured, and
/// builds the client. Proxy failures keep [`ProxyError`] semantics so the
/// credential-hiding rules above hold everywhere.
pub fn build_http_client(
    timeout_secs: u64,
    proxy: Option<&str>,
    no_proxy: &[String],
) -> Result<reqwest::Client, ProxyError> {
    let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(
        timeout_secs.max(1),
    ));
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
