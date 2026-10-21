//! Tool-call text protocol parsing: one module per protocol, plus the
//! dispatch that picks between them.

pub(crate) mod call_id;
pub(crate) mod detect;
pub(crate) mod dispatch;
pub(crate) mod invoke;
pub(crate) mod json_raw;
pub(crate) mod json_wrapped;
pub(crate) mod options;
pub(crate) mod xml;

pub use detect::{has_json_tool_calls, has_raw_json_tool_calls, has_xml_tool_calls};
pub use dispatch::{parse_from_text, parse_partial};
pub use invoke::{parse_invoke_json_calls, parse_invoke_json_calls_detailed, InvokeParseError};
pub use json_raw::parse_raw_json_tool_calls;
pub use json_wrapped::parse_json_tool_calls;
pub use options::{ParseFormat, ToolCallParseOptions};
pub use xml::parse_xml_tool_calls;
