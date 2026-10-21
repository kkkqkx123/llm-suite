//! Provider-agnostic building blocks shared by every codec: text-mode tool
//! negotiation, parameter merging and HTTP plumbing.

pub(crate) mod http;
pub(crate) mod params;
pub(crate) mod text_mode;

#[cfg(test)]
pub(crate) mod test_support;

pub(crate) use http::{apply_auth_and_headers, apply_custom_body, endpoint_url};
pub(crate) use params::{merge_and_apply_params, resolve_generation};
pub(crate) use text_mode::{
    convert_history_for_text_mode, is_text_mode, parse_text_tool_calls, text_mode_system_content,
};
