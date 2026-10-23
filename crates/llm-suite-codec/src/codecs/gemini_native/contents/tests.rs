
use super::*;
use crate::codecs::gemini_native::test_support::msg;
use llm_suite_types::message::MessageRole;

#[test]
fn rich_image_blocks_become_inline_data_or_file_data() {
    let image_msg = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![
            MessageContent::Text {
                text: "what is this".to_string(),
            },
            MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/png;base64,AAAA".to_string(),
                    detail: None,
                    file_id: None,
                },
            },
            MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "gs://bucket/a.jpg".to_string(),
                    detail: None,
                    file_id: None,
                },
            },
            MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://generativelanguage.googleapis.com/v1beta/files/abc".to_string(),
                    detail: None,
                    file_id: None,
                },
            },
            MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://youtube.com/watch?v=abc".to_string(),
                    detail: None,
                    file_id: None,
                },
            },
        ]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let entries = convert_messages(&[image_msg]).expect("must convert");
    assert_eq!(entries.len(), 1);
    let parts = entries[0]["parts"].as_array().expect("parts array");
    // Regression guard: none of the blocks may be silently dropped.
    assert_eq!(parts.len(), 5, "text + inlineData + 3 fileData");
    assert_eq!(parts[0]["text"], "what is this");
    assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
    assert_eq!(parts[1]["inlineData"]["data"], "AAAA");
    assert_eq!(parts[2]["fileData"]["fileUri"], "gs://bucket/a.jpg");
    assert_eq!(parts[2]["fileData"]["mimeType"], "image/jpeg");
    assert_eq!(
        parts[3]["fileData"]["fileUri"],
        "https://generativelanguage.googleapis.com/v1beta/files/abc"
    );
    assert_eq!(
        parts[4]["fileData"]["fileUri"],
        "https://youtube.com/watch?v=abc"
    );
}

#[test]
fn an_arbitrary_https_image_url_is_rejected_not_mapped_to_file_data() {
    let image_msg = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
            image_url: llm_suite_types::message::ImageUrlContent {
                url: "https://example.com/a.jpg".to_string(),
                detail: None,
                file_id: None,
            },
        }]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let err =
        convert_messages(&[image_msg]).expect_err("a plain https URL must not become a fileUri");
    assert!(format!("{err}").contains("fileData.fileUri"), "{err}");
}

#[test]
fn a_plain_text_tool_result_becomes_a_function_response() {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        call_id: None,
        function: llm_suite_types::message::LlmFunctionCall {
            name: "get_weather".to_string(),
            arguments: "{}".to_string(),
        },
        thought_signature: None,
    }]);
    let mut result = msg(MessageRole::Tool, "sunny");
    result.tool_call_id = Some("call_1".to_string());

    let contents = convert_messages(&[assistant, result]).expect("must convert");
    assert_eq!(contents[0]["role"], serde_json::json!("model"));
    assert_eq!(
        contents[0]["parts"][0]["functionCall"]["name"],
        serde_json::json!("get_weather")
    );
    assert_eq!(contents[1]["role"], serde_json::json!("user"));
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["name"],
        serde_json::json!("get_weather"),
        "the function name comes from the call the result answers"
    );
    assert!(
        contents[1]["parts"][0]["functionResponse"]
            .get("id")
            .is_none(),
        "a call id the provider never issued must not be echoed back"
    );
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["response"],
        serde_json::json!({"output": "sunny"}),
        "the response payload is an object, the text lands in `output`"
    );
}

#[test]
fn a_tool_result_without_a_known_function_is_rejected() {
    let mut result = msg(MessageRole::Tool, "sunny");
    result.tool_call_id = Some("call_unknown".to_string());
    let err = convert_messages(&[result]).expect_err("an unnamed function response is meaningless");
    assert!(format!("{err}").contains("call_unknown"), "{err}");
}

#[test]
fn a_rich_tool_use_block_becomes_a_function_call_part() {
    let mut assistant = msg(MessageRole::Assistant, "checking");
    assistant.content = MessageContentValue::Rich(vec![
        MessageContent::Text {
            text: "checking".to_string(),
        },
        MessageContent::ToolUse {
            tool_use: llm_suite_types::message::ToolUseContent {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({"city": "Beijing"}),
                thought_signature: None,
            },
        },
    ]);
    let contents = convert_messages(&[assistant]).expect("must convert");
    let parts = contents[0]["parts"].as_array().expect("parts");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["text"], serde_json::json!("checking"));
    assert_eq!(
        parts[1]["functionCall"]["name"],
        serde_json::json!("get_weather")
    );
    assert_eq!(
        parts[1]["functionCall"]["args"]["city"],
        serde_json::json!("Beijing")
    );
}

#[test]
fn a_rich_tool_result_names_the_function_of_its_tool_use_block() {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.content = MessageContentValue::Rich(vec![MessageContent::ToolUse {
        tool_use: llm_suite_types::message::ToolUseContent {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            input: serde_json::json!({}),
            thought_signature: None,
        },
    }]);
    let mut result = msg(MessageRole::Tool, "");
    result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
        tool_result: llm_suite_types::message::ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "sunny".to_string(),
            blocks: None,
            is_error: None,
        },
    }]);

    let contents = convert_messages(&[assistant, result]).expect("must convert");
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["name"],
        serde_json::json!("get_weather")
    );
    assert!(
        contents[1]["parts"][0]["functionResponse"]
            .get("id")
            .is_none(),
        "a call id the provider never issued must not be echoed back"
    );
}

#[test]
fn a_message_without_parts_is_skipped() {
    let empty = msg(MessageRole::User, "");
    let contents = convert_messages(&[empty, msg(MessageRole::User, "hi")]).expect("must convert");
    assert_eq!(contents.len(), 1);
    assert_eq!(contents[0]["parts"][0]["text"], serde_json::json!("hi"));
}

#[test]
fn the_part_fields_use_the_official_camel_case_names() {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        call_id: None,
        function: llm_suite_types::message::LlmFunctionCall {
            name: "get_weather".to_string(),
            arguments: "{}".to_string(),
        },
        thought_signature: None,
    }]);
    let mut result = msg(MessageRole::Tool, "sunny");
    result.tool_call_id = Some("call_1".to_string());

    let contents = convert_messages(&[assistant, result]).expect("must convert");
    let call = &contents[0]["parts"][0];
    assert!(
        call.get("functionCall").is_some(),
        "the call part uses the official name: {contents:?}"
    );
    assert!(call.get("function_call").is_none());
    let response = &contents[1]["parts"][0];
    assert!(
        response.get("functionResponse").is_some(),
        "the response part uses the official name: {contents:?}"
    );
    assert!(response.get("function_response").is_none());
}

#[test]
fn an_input_audio_block_becomes_inline_data() {
    let user = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![MessageContent::InputAudio {
            input_audio: llm_suite_types::message::InputAudioContent {
                data: "QUJD".to_string(),
                format: "wav".to_string(),
            },
        }]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let contents = convert_messages(&[user]).expect("must convert");
    let part = &contents[0]["parts"][0];
    assert_eq!(
        part["inlineData"]["mimeType"],
        serde_json::json!("audio/wav")
    );
    assert_eq!(part["inlineData"]["data"], serde_json::json!("QUJD"));
}

#[test]
fn a_base64_file_input_becomes_inline_data() {
    let user = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![MessageContent::FileInput {
            file: llm_suite_types::message::FileContent {
                source: llm_suite_types::message::FileSource::Base64 {
                    media_type: "application/pdf".to_string(),
                    data: "QUJD".to_string(),
                },
                filename: None,
                detail: None,
                citations_enabled: None,
                context: None,
            },
        }]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let contents = convert_messages(&[user]).expect("must convert");
    let part = &contents[0]["parts"][0];
    assert_eq!(
        part["inlineData"]["mimeType"],
        serde_json::json!("application/pdf")
    );
    assert_eq!(part["inlineData"]["data"], serde_json::json!("QUJD"));
}

#[test]
fn a_file_id_input_is_rejected() {
    let user = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![MessageContent::FileInput {
            file: llm_suite_types::message::FileContent {
                source: llm_suite_types::message::FileSource::FileId {
                    file_id: "file_1".to_string(),
                },
                filename: None,
                detail: None,
                citations_enabled: None,
                context: None,
            },
        }]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let err = convert_messages(&[user]).expect_err("gemini needs a URI, not a Files-API id");
    assert!(format!("{err}").contains("Files API"), "{err}");
}

#[test]
fn rich_tool_result_blocks_become_function_response_parts() {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        call_id: None,
        function: llm_suite_types::message::LlmFunctionCall {
            name: "screenshot".to_string(),
            arguments: "{}".to_string(),
        },
        thought_signature: None,
    }]);
    let mut result = msg(MessageRole::Tool, "attached");
    result.tool_call_id = Some("call_1".to_string());
    result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
        tool_result: llm_suite_types::message::ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "attached".to_string(),
            blocks: Some(vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/png;base64,QUJD".to_string(),
                    detail: None,
                    file_id: None,
                },
            }]),
            is_error: None,
        },
    }]);

    let contents = convert_messages(&[assistant, result]).expect("must convert");
    let part = &contents[1]["parts"][0]["functionResponse"];
    assert_eq!(part["response"]["output"], serde_json::json!("attached"));
    let parts = part["parts"].as_array().expect("rich parts array");
    assert_eq!(
        parts[0]["inlineData"]["mimeType"],
        serde_json::json!("image/png")
    );
    assert_eq!(parts[0]["inlineData"]["data"], serde_json::json!("QUJD"));
}

#[test]
fn a_document_uri_is_not_mislabelled_as_an_image() {
    let user = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![MessageContent::FileInput {
            file: llm_suite_types::message::FileContent {
                source: llm_suite_types::message::FileSource::Url {
                    url: "gs://bucket/report.pdf".to_string(),
                },
                filename: None,
                detail: None,
                citations_enabled: None,
                context: None,
            },
        }]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let contents = convert_messages(&[user]).expect("must convert");
    let part = &contents[0]["parts"][0];
    assert_eq!(
        part["fileData"]["mimeType"],
        serde_json::json!("application/pdf")
    );
    assert_eq!(
        part["fileData"]["fileUri"],
        serde_json::json!("gs://bucket/report.pdf")
    );
}

#[test]
fn a_document_in_a_tool_result_becomes_inline_data() {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        call_id: None,
        function: llm_suite_types::message::LlmFunctionCall {
            name: "report".to_string(),
            arguments: "{}".to_string(),
        },
        thought_signature: None,
    }]);
    let mut result = msg(MessageRole::Tool, "attached");
    result.tool_call_id = Some("call_1".to_string());
    result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
        tool_result: llm_suite_types::message::ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "attached".to_string(),
            blocks: Some(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Base64 {
                        media_type: "application/pdf".to_string(),
                        data: "QUJD".to_string(),
                    },
                    filename: None,
                    detail: None,
                    citations_enabled: None,
                    context: None,
                },
            }]),
            is_error: None,
        },
    }]);

    let contents = convert_messages(&[assistant, result]).expect("must convert");
    let parts = contents[1]["parts"][0]["functionResponse"]["parts"]
        .as_array()
        .expect("rich parts array");
    assert_eq!(
        parts[0]["inlineData"]["mimeType"],
        serde_json::json!("application/pdf")
    );
    assert_eq!(parts[0]["inlineData"]["data"], serde_json::json!("QUJD"));
}

#[test]
fn a_failed_tool_result_is_reported_under_the_error_key() {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        call_id: None,
        function: llm_suite_types::message::LlmFunctionCall {
            name: "boom".to_string(),
            arguments: "{}".to_string(),
        },
        thought_signature: None,
    }]);
    let mut result = msg(MessageRole::Tool, "it exploded");
    result.tool_call_id = Some("call_1".to_string());
    result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
        tool_result: llm_suite_types::message::ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "it exploded".to_string(),
            blocks: None,
            is_error: Some(true),
        },
    }]);

    let contents = convert_messages(&[assistant, result]).expect("must convert");
    let response = &contents[1]["parts"][0]["functionResponse"]["response"];
    assert_eq!(response["error"], serde_json::json!("it exploded"));
    assert!(response.get("output").is_none());
}

#[test]
fn a_native_part_is_replayed_verbatim() {
    let user = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![MessageContent::Native {
            provider: "gemini".to_string(),
            kind: "video_metadata".to_string(),
            payload: serde_json::json!({
                "inlineData": {"mimeType": "video/mp4", "data": "QUJD"},
                "videoMetadata": {"startOffset": "0s", "endOffset": "10s"},
            }),
        }]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let contents = convert_messages(&[user]).expect("must convert");
    let part = &contents[0]["parts"][0];
    assert_eq!(part["videoMetadata"]["endOffset"], serde_json::json!("10s"));
    assert_eq!(
        part["inlineData"]["mimeType"],
        serde_json::json!("video/mp4")
    );
}

#[test]
fn a_native_part_of_another_provider_is_rejected() {
    let user = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![MessageContent::Native {
            provider: "anthropic".to_string(),
            kind: "search_result".to_string(),
            payload: serde_json::json!({"type": "search_result"}),
        }]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    };
    let err = convert_messages(&[user]).expect_err("a block of another provider is refused");
    assert!(format!("{err}").contains("anthropic"), "{err}");
}
