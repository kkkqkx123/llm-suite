//! Shared HTTP header helpers for the llm-suite crates.
//!
//! Only pure parsing lives here; the `reqwest` types stay in the calling
//! crates so this module keeps zero HTTP-client dependencies.

/// Parses a `Retry-After` header value into milliseconds: a plain number of
/// seconds, or an HTTP date (RFC 2822 / IMF-fixdate) measured against now.
/// Unparseable or past dates yield `None` (bare rate limit, still retryable).
pub fn parse_retry_after_ms(value: Option<&str>) -> Option<u64> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(secs) = value.parse::<u64>() {
        return secs.checked_mul(1000);
    }
    let deadline = chrono::DateTime::parse_from_rfc2822(value)
        .ok()
        .map(|dated| dated.with_timezone(&chrono::Utc))
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT")
                .ok()
                .map(|naive| naive.and_utc())
        })?;
    Some(
        deadline
            .signed_duration_since(chrono::Utc::now())
            .num_milliseconds()
            .max(0) as u64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seconds_and_rejects_garbage() {
        assert_eq!(parse_retry_after_ms(None), None);
        assert_eq!(parse_retry_after_ms(Some("")), None);
        assert_eq!(parse_retry_after_ms(Some("2")), Some(2000));
        assert_eq!(parse_retry_after_ms(Some("not-a-date")), None);
    }

    #[test]
    fn parses_future_http_date() {
        let future = (chrono::Utc::now() + chrono::Duration::seconds(120)).to_rfc2822();
        let parsed = parse_retry_after_ms(Some(&future)).expect("http date must parse");
        assert!((110_000..=130_000).contains(&parsed));
    }
}
