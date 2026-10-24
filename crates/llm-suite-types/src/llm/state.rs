use serde::{Deserialize, Serialize};

/// Wire protocol format of an LLM endpoint.
///
/// A format describes how HTTP requests and responses are encoded, including
/// streaming chunks, tool schemas and token-counting endpoints. It does not
/// describe where the endpoint lives: connection details (base URL,
/// authentication, headers, model discovery) belong to
/// [`LlmProviderDefinition`](super::LlmProviderDefinition), and the model
/// binding belongs to [`LlmProfile`](super::LlmProfile).
///
/// The four built-in formats cover the supported protocols; `Custom`
/// addresses a codec registered at runtime through the codec registry
/// (plugin extension point).
///
/// Serde is implemented via the canonical string form: built-ins use their
/// `SCREAMING_SNAKE_CASE` name, any other string maps to `Custom(name)`.
#[derive(Debug, Clone, PartialEq)]
pub enum LlmFormat {
    OpenaiChat,
    OpenaiResponse,
    Anthropic,
    GeminiNative,
    /// Format whose codec is resolved through the runtime registry.
    Custom(String),
}

impl LlmFormat {
    /// Canonical string form: the `SCREAMING_SNAKE_CASE` name of the built-in
    /// formats, or the raw registered name for `Custom`.
    pub fn as_str(&self) -> &str {
        match self {
            LlmFormat::OpenaiChat => "OPENAI_CHAT",
            LlmFormat::OpenaiResponse => "OPENAI_RESPONSE",
            LlmFormat::Anthropic => "ANTHROPIC",
            LlmFormat::GeminiNative => "GEMINI_NATIVE",
            LlmFormat::Custom(name) => name,
        }
    }
}

impl std::str::FromStr for LlmFormat {
    type Err = core::convert::Infallible;

    /// Parse a format from its canonical string form. Every string is
    /// valid: built-ins map to their variant, anything else is `Custom`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "OPENAI_CHAT" => LlmFormat::OpenaiChat,
            "OPENAI_RESPONSE" => LlmFormat::OpenaiResponse,
            "ANTHROPIC" => LlmFormat::Anthropic,
            "GEMINI_NATIVE" => LlmFormat::GeminiNative,
            other => LlmFormat::Custom(other.to_string()),
        })
    }
}

/// Authentication method of a stored credential.
///
/// Serialized in snake_case so configuration files read naturally
/// (`method = "oauth"`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// Static API key resolved from a store, environment or request.
    ApiKey,
    /// OAuth credential with login and refresh lifecycle.
    #[serde(rename = "oauth")]
    OAuth,
}

/// Error returned when a string does not name an [`AuthMethod`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthMethodParseError(pub String);

impl std::fmt::Display for AuthMethodParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown auth method: {}", self.0)
    }
}

impl std::error::Error for AuthMethodParseError {}

impl std::str::FromStr for AuthMethod {
    type Err = AuthMethodParseError;

    /// Parse from the canonical snake_case string form (`api_key`, `oauth`).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "api_key" => Ok(Self::ApiKey),
            "oauth" => Ok(Self::OAuth),
            other => Err(AuthMethodParseError(other.to_string())),
        }
    }
}

impl Serialize for LlmFormat {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for LlmFormat {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.parse::<LlmFormat>() {
            Ok(format) => Ok(format),
            Err(infallible) => match infallible {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_roundtrip() {
        for format in [
            LlmFormat::OpenaiChat,
            LlmFormat::OpenaiResponse,
            LlmFormat::Anthropic,
            LlmFormat::GeminiNative,
        ] {
            let json = serde_json::to_string(&format).unwrap();
            let back: LlmFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(back, format);
        }
        assert_eq!(
            serde_json::to_string(&LlmFormat::OpenaiChat).unwrap(),
            "\"OPENAI_CHAT\""
        );
        assert_eq!(
            serde_json::from_str::<LlmFormat>("\"ANTHROPIC\"").unwrap(),
            LlmFormat::Anthropic
        );
    }

    #[test]
    fn custom_roundtrip() {
        let format = LlmFormat::Custom("my_provider".to_string());
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"my_provider\"");
        let back: LlmFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(back, format);

        let parsed: LlmFormat = serde_json::from_str("\"MY_GATEWAY_PROVIDER\"").unwrap();
        assert_eq!(parsed, LlmFormat::Custom("MY_GATEWAY_PROVIDER".to_string()));
    }

    #[test]
    fn as_str_matches_serde() {
        for format in [
            LlmFormat::OpenaiChat,
            LlmFormat::OpenaiResponse,
            LlmFormat::Anthropic,
            LlmFormat::GeminiNative,
        ] {
            let from_name = format.as_str().parse::<LlmFormat>().unwrap();
            assert_eq!(from_name, format);
        }
    }

    #[test]
    fn auth_method_roundtrip() {
        assert_eq!("api_key".parse::<AuthMethod>().unwrap(), AuthMethod::ApiKey);
        assert_eq!("oauth".parse::<AuthMethod>().unwrap(), AuthMethod::OAuth);
        assert_eq!(
            serde_json::to_string(&AuthMethod::ApiKey).unwrap(),
            "\"api_key\""
        );
        assert_eq!(
            serde_json::to_string(&AuthMethod::OAuth).unwrap(),
            "\"oauth\""
        );
        let back: AuthMethod = serde_json::from_str("\"oauth\"").unwrap();
        assert_eq!(back, AuthMethod::OAuth);
        let err = "totp".parse::<AuthMethod>().unwrap_err();
        assert_eq!(err, AuthMethodParseError("totp".to_string()));
    }
}
