//! Network model discovery for llm-suite providers.
//!
//! Split out of `llm-suite-config`: everything here needs an HTTP client
//! (`reqwest`), while the registry core in `llm-suite-config` stays
//! network-free. Hosts that only validate profiles and merge requests
//! never pay for the TLS stack; only the gateway and CLI listing tools
//! depend on this crate.
//!
//! Pure helpers over discovery results (`filter_models`) also live here,
//! next to their use site. Profile-side bridging
//! (`apply_catalog_metadata`) stays in `llm-suite-config`.

use llm_suite_types::llm::{LlmProviderDefinition, ModelDiscovery, ModelInfo};

use llm_suite_codec::error::{LlmError, LlmResult};

/// Default path of the models endpoint below the version segment.
pub const DEFAULT_MODELS_PATH: &str = "models";
/// Default extraction rule covering OpenAI-style `{ "data": [{ "id": .. }] }`.
pub const DEFAULT_MODELS_JSON_PATH: &str = "data[*].id";

/// Off-hot-path model listing for a provider definition.
///
/// Serves CLI listing, API listing and assembly-time checks. Discovery
/// failures are reported as errors to the caller; callers at assembly time
/// degrade them to warnings and never block profile registration.
#[derive(Clone)]
pub struct ModelCatalog {
    client: reqwest::Client,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Refresh the model listings of many provider definitions at once.
    ///
    /// Transactional in the reporting sense: every definition yields an
    /// entry, successes carry the discovered models and failures carry
    /// the discovery error. Callers degrade failures to warnings; a
    /// single unreachable endpoint never blocks the remaining providers.
    /// `api_keys` maps provider id to the effective key for that
    /// provider; missing entries list without authentication.
    pub async fn refresh_all(
        &self,
        definitions: &[LlmProviderDefinition],
        api_keys: &std::collections::HashMap<String, String>,
    ) -> std::collections::HashMap<String, LlmResult<Vec<ModelInfo>>> {
        let mut out = std::collections::HashMap::new();
        for definition in definitions {
            let key = api_keys.get(&definition.id).map(String::as_str);
            out.insert(
                definition.id.clone(),
                self.list_models(definition, key).await,
            );
        }
        out
    }

    /// Refresh every provider listing, then credential-filter the successes.
    ///
    /// Same transactional reporting as [`Self::refresh_all`]: every
    /// definition yields an entry, failures pass through untouched, and only
    /// successful listings are filtered by `predicate`. The predicate closes
    /// over the logged-in credential (allow-list, prefix, subscription
    /// flag); a plain function keeps the call free of trait objects.
    pub async fn refresh_filtered(
        &self,
        definitions: &[LlmProviderDefinition],
        api_keys: &std::collections::HashMap<String, String>,
        predicate: impl Fn(&ModelInfo) -> bool,
    ) -> std::collections::HashMap<String, LlmResult<Vec<ModelInfo>>> {
        let refreshed = self.refresh_all(definitions, api_keys).await;
        refreshed
            .into_iter()
            .map(|(id, result)| {
                let filtered = result.map(|models| filter_models(models, &predicate));
                (id, filtered)
            })
            .collect()
    }

    /// List the models advertized by a provider definition.
    ///
    /// `api_key` carries the effective profile key when the caller has one;
    /// definitions without remote discovery return without using it.
    pub async fn list_models(
        &self,
        definition: &LlmProviderDefinition,
        api_key: Option<&str>,
    ) -> LlmResult<Vec<ModelInfo>> {
        let discovery = definition
            .model_discovery
            .clone()
            .unwrap_or(ModelDiscovery::Disabled);
        match discovery {
            ModelDiscovery::Disabled => Ok(Vec::new()),
            ModelDiscovery::StaticList { models } => Ok(models),
            ModelDiscovery::ModelsEndpoint { path, json_path } => {
                let base_url = definition.base_url.clone().ok_or_else(|| {
                    LlmError::ConfigError(format!(
                        "Provider '{}' has no base_url for model discovery",
                        definition.id
                    ))
                })?;
                let url = join_models_url(
                    &base_url,
                    definition.api_version.as_deref(),
                    path.as_deref(),
                );
                let mut request = self.client.get(&url);
                request = apply_discovery_auth(
                    request,
                    definition.auth_type.as_deref().unwrap_or("native"),
                    api_key,
                );
                if let Some(headers) = definition.default_headers.as_ref() {
                    for (key, value) in headers {
                        request = request.header(key, value.as_str().unwrap_or(""));
                    }
                }
                let response = request.send().await?;
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if !status.is_success() {
                    return Err(LlmError::ProviderError {
                        status: Some(status.as_u16()),
                        message: body,
                        retry_after_ms: None,
                    });
                }
                let json: serde_json::Value = serde_json::from_str(&body)?;
                Ok(extract_models(
                    &json,
                    json_path.as_deref().unwrap_or(DEFAULT_MODELS_JSON_PATH),
                ))
            }
            ModelDiscovery::CustomEndpoint {
                url,
                method,
                headers,
                json_path,
            } => {
                let method = method.as_deref().unwrap_or("GET");
                let mut request = self.client.request(
                    method.parse().map_err(|_| {
                        LlmError::ConfigError(format!(
                            "Provider '{}' has an invalid discovery method '{}'",
                            definition.id, method
                        ))
                    })?,
                    &url,
                );
                if let Some(headers) = headers.as_ref() {
                    for (key, value) in headers {
                        request = request.header(key, value.as_str().unwrap_or(""));
                    }
                }
                let response = request.send().await?;
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if !status.is_success() {
                    return Err(LlmError::ProviderError {
                        status: Some(status.as_u16()),
                        message: body,
                        retry_after_ms: None,
                    });
                }
                let json: serde_json::Value = serde_json::from_str(&body)?;
                Ok(extract_models(
                    &json,
                    json_path.as_deref().unwrap_or(DEFAULT_MODELS_JSON_PATH),
                ))
            }
        }
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter a discovered model listing by credential ownership.
///
/// Free function with a plain predicate so no trait object is needed:
/// callers close over the logged-in credential (id prefix, allow-list,
/// subscription flag) and keep only the entries that credential may use.
pub fn filter_models(
    models: Vec<ModelInfo>,
    predicate: impl Fn(&ModelInfo) -> bool,
) -> Vec<ModelInfo> {
    models.into_iter().filter(|m| predicate(m)).collect()
}

/// Keep only the models whose id appears in `allowed`.
///
/// Convenience over [`filter_models`] for the common credential-filtering
/// shape: the logged-in credential owns an allow-list of model ids (pi
/// `filterModels` / `filterAllModels`), and every other entry is dropped.
/// An empty allow-list keeps nothing; callers that mean "no filtering"
/// skip this helper instead.
pub fn filter_models_by_allowlist(
    models: Vec<ModelInfo>,
    allowed: &std::collections::HashSet<String>,
) -> Vec<ModelInfo> {
    filter_models(models, |m| allowed.contains(&m.id))
}

/// Keep only the models whose id starts with one of `prefixes`.
///
/// Convenience over [`filter_models`] for prefix-scoped credentials
/// (e.g. an OAuth login that may use `gpt-*` but not internal eval
/// models). An empty prefix list keeps nothing.
pub fn filter_models_by_prefix(models: Vec<ModelInfo>, prefixes: &[String]) -> Vec<ModelInfo> {
    filter_models(models, |m| {
        prefixes.iter().any(|prefix| m.id.starts_with(prefix))
    })
}

/// Join `{base_url}/{api_version}/{path}` tolerating trailing slashes and a
/// base URL that already ends with the version segment.
pub fn join_models_url(base_url: &str, api_version: Option<&str>, path: Option<&str>) -> String {
    let mut url = base_url.trim_end_matches('/').to_string();
    let version = api_version.unwrap_or("v1");
    if !version.is_empty() && !url.ends_with(&format!("/{version}")) {
        url.push('/');
        url.push_str(version);
    }
    let path = path.unwrap_or(DEFAULT_MODELS_PATH);
    if !path.is_empty() {
        url.push('/');
        url.push_str(path.trim_start_matches('/'));
    }
    url
}

fn apply_discovery_auth(
    request: reqwest::RequestBuilder,
    auth_type: &str,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    let Some(api_key) = api_key else {
        return request;
    };
    match auth_type {
        "x-api-key" => request.header("x-api-key", api_key),
        "x-goog-api-key" => request.header("x-goog-api-key", api_key),
        _ => request.header("Authorization", format!("Bearer {api_key}")),
    }
}

/// Extract model ids from a JSON document following a `a.b[*].c` rule.
///
/// A `[*]` segment maps over arrays; a trailing scalar segment yields ids
/// directly, while objects yield their `id` field when present.
pub fn extract_models(value: &serde_json::Value, json_path: &str) -> Vec<ModelInfo> {
    let segments: Vec<&str> = json_path.split('.').collect();
    let mut current = vec![value];
    for segment in segments {
        let mut next = Vec::new();
        if let Some(key) = segment.strip_suffix("[*]") {
            for item in current {
                let target = if key.is_empty() { item } else { &item[key] };
                if let Some(array) = target.as_array() {
                    next.extend(array.iter());
                }
            }
        } else {
            for item in current {
                let target = &item[segment];
                if !target.is_null() {
                    next.push(target);
                }
            }
        }
        current = next;
    }
    current
        .into_iter()
        .filter_map(|item| {
            if let Some(id) = item.as_str() {
                Some(ModelInfo {
                    id: id.to_string(),
                    name: None,
                    context_window_size: None,
                    metadata: None,
                })
            } else {
                item.get("id")
                    .and_then(|id| id.as_str())
                    .map(|id| ModelInfo {
                        id: id.to_string(),
                        name: item
                            .get("name")
                            .and_then(|name| name.as_str())
                            .map(String::from),
                        context_window_size: None,
                        metadata: None,
                    })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_endpoint_without_duplicating_version() {
        assert_eq!(
            join_models_url("https://api.test.com", None, None),
            "https://api.test.com/v1/models"
        );
        assert_eq!(
            join_models_url("https://api.test.com/", Some("v1"), Some("models")),
            "https://api.test.com/v1/models"
        );
        assert_eq!(
            join_models_url("https://api.test.com/v1", None, None),
            "https://api.test.com/v1/models"
        );
        assert_eq!(
            join_models_url("https://api.test.com", Some("v1beta"), None),
            "https://api.test.com/v1beta/models"
        );
    }

    #[test]
    fn extracts_openai_style_ids() {
        let body: serde_json::Value = serde_json::json!({
            "data": [{"id": "a"}, {"id": "b"}]
        });
        let models = extract_models(&body, "data[*].id");
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn extracts_object_ids_with_names() {
        let body: serde_json::Value = serde_json::json!({
            "models": [{"id": "m1", "name": "First"}]
        });
        let models = extract_models(&body, "models[*]");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "m1");
        assert_eq!(models[0].name.as_deref(), Some("First"));
    }

    #[test]
    fn unknown_paths_yield_empty() {
        let body: serde_json::Value = serde_json::json!({"data": []});
        assert!(extract_models(&body, "missing[*].id").is_empty());
    }

    #[test]
    fn filter_models_keeps_only_matching_entries() {
        let models = vec![
            ModelInfo {
                id: "gpt-4o".to_string(),
                name: None,
                context_window_size: None,
                metadata: None,
            },
            ModelInfo {
                id: "internal-only".to_string(),
                name: None,
                context_window_size: None,
                metadata: None,
            },
        ];
        let kept = filter_models(models, |m| !m.id.contains("internal"));
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "gpt-4o");
    }

    fn model_info(id: &str) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            name: None,
            context_window_size: None,
            metadata: None,
        }
    }

    #[test]
    fn allowlist_keeps_only_listed_ids() {
        let models = vec![model_info("gpt-4o"), model_info("other")];
        let allowed: std::collections::HashSet<String> =
            ["gpt-4o".to_string()].into_iter().collect();
        let kept = filter_models_by_allowlist(models, &allowed);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "gpt-4o");
    }

    #[test]
    fn prefix_filter_keeps_matching_ids() {
        let models = vec![model_info("gpt-4o"), model_info("claude-x")];
        let kept = filter_models_by_prefix(models, &["gpt-".to_string()]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "gpt-4o");
    }

    #[tokio::test]
    async fn refresh_filtered_applies_predicate_to_successes() {
        use llm_suite_types::llm::{LlmFormat, LlmProviderDefinition, ModelDiscovery};
        let catalog = ModelCatalog::new();
        let definitions = vec![LlmProviderDefinition {
            id: "static".to_string(),
            name: None,
            description: None,
            base_url: None,
            auth_type: None,
            default_headers: None,
            format: LlmFormat::OpenaiChat,
            model_discovery: Some(ModelDiscovery::StaticList {
                models: vec![model_info("gpt-4o"), model_info("internal-only")],
            }),
            api_version: None,
            metadata: None,
            proxy: None,
            no_proxy: None,
            rate_limit: None,
            auth: None,
            compat: None,
        }];
        let refreshed = catalog
            .refresh_filtered(&definitions, &Default::default(), |m| {
                !m.id.contains("internal")
            })
            .await;
        let models = refreshed
            .get("static")
            .expect("entry")
            .as_ref()
            .expect("success");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4o");
    }
}
