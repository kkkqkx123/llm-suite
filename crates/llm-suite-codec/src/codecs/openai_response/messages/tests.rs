    use super::*;

    #[test]
    fn function_call_and_output_share_the_provider_call_id() {
        let mut assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Text(String::new()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: Some(vec![llm_suite_types::message::LlmToolCall {
                id: "fc_1".to_string(),
                r#type: "function".to_string(),
                call_id: Some("call_1".to_string()),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                },
                thought_signature: None,
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        // The result references the call by its item id.
        let mut result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Text("sunny".to_string()),
            timestamp: 0,
            tool_call_id: Some("fc_1".to_string()),
            tool_name: Some("get_weather".to_string()),
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        assistant.timestamp = 0;
        result.timestamp = 0;

        let items = convert_messages(&[assistant, result]).expect("must convert");
        let call = items
            .iter()
            .find(|item| item["type"] == serde_json::json!("function_call"))
            .expect("function call item present");
        let output = items
            .iter()
            .find(|item| item["type"] == serde_json::json!("function_call_output"))
            .expect("function call output item present");
        assert_eq!(call["call_id"], serde_json::json!("call_1"));
        assert_eq!(
            output["call_id"], call["call_id"],
            "the output must reference the same call"
        );
    }

    #[test]
    fn rich_image_blocks_map_to_input_image_parts() {
        let req_messages = vec![Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![
                MessageContent::Text {
                    text: "describe".to_string(),
                },
                MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "https://example.com/a.png".to_string(),
                        detail: Some(llm_suite_types::message::ImageDetail::Auto),
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
        }];
        let entries = convert_messages(&req_messages).expect("must convert");
        let parts = entries[0]["content"].as_array().expect("content array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["type"], "input_image");
        assert_eq!(parts[1]["image_url"], "https://example.com/a.png");
        assert_eq!(parts[1]["detail"], "auto");
    }

    #[test]
    fn an_image_without_a_detail_carries_the_provider_default() {
        let req_messages = vec![Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
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
        }];
        let entries = convert_messages(&req_messages).expect("must convert");
        assert_eq!(
            entries[0]["content"][0]["detail"],
            serde_json::json!("auto"),
            "the official image part carries a detail level"
        );
    }

    #[test]
    fn convert_messages_emits_function_call_and_output_items() {
        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Text("let me check".to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: Some(vec![llm_suite_types::message::LlmToolCall {
                id: "fc_1".to_string(),
                r#type: "function".to_string(),
                call_id: Some("call_abc".to_string()),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "get_weather".to_string(),
                    arguments: "{\"city\":\"Beijing\"}".to_string(),
                },
                thought_signature: None,
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let tool_result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Text("sunny".to_string()),
            timestamp: 0,
            tool_call_id: Some("call_abc".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let items = convert_messages(&[assistant, tool_result]).expect("must convert");

        // assistant text item + function_call item + function_call_output item
        assert_eq!(items.len(), 3);

        let func_call = items
            .iter()
            .find(|i| i.get("type").and_then(|v| v.as_str()) == Some("function_call"))
            .expect("function_call item present");
        assert_eq!(func_call["call_id"], serde_json::json!("call_abc"));
        assert_eq!(func_call["name"], serde_json::json!("get_weather"));
        assert_eq!(
            func_call["arguments"],
            serde_json::json!("{\"city\":\"Beijing\"}")
        );
        assert!(
            func_call.get("tool_calls").is_none(),
            "must not emit Chat-style tool_calls"
        );

        let func_output = items
            .iter()
            .find(|i| i.get("type").and_then(|v| v.as_str()) == Some("function_call_output"))
            .expect("function_call_output item present");
        assert_eq!(func_output["call_id"], serde_json::json!("call_abc"));
        assert_eq!(func_output["output"], serde_json::json!("sunny"));
    }

    #[test]
    fn a_rich_tool_use_block_becomes_a_function_call_item() {
        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Beijing"}),
                    thought_signature: None,
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
        let items = convert_messages(&[assistant]).expect("must convert");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], serde_json::json!("function_call"));
        assert_eq!(items[0]["call_id"], serde_json::json!("call_1"));
        assert_eq!(items[0]["name"], serde_json::json!("get_weather"));
        assert_eq!(
            items[0]["arguments"],
            serde_json::json!("{\"city\":\"Beijing\"}")
        );
    }

    #[test]
    fn a_rich_tool_result_block_becomes_a_function_call_output_item() {
        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
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
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "sunny".to_string(),
                    blocks: None,
                    is_error: None,
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
        let items = convert_messages(&[assistant, result]).expect("must convert");
        let call = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call"))
            .expect("the call must be present");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("the output must be present");
        assert_eq!(call["call_id"], serde_json::json!("call_1"));
        assert_eq!(
            output["call_id"], call["call_id"],
            "the result must reference the call it answers"
        );
        assert_eq!(output["output"], serde_json::json!("sunny"));
    }

    #[test]
    fn a_file_input_becomes_an_input_file_part_with_the_right_reference() {
        let msg = |source| Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source,
                    filename: Some("doc.pdf".to_string()),
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
        let items = convert_messages(&[msg(llm_suite_types::message::FileSource::Base64 {
            media_type: "application/pdf".to_string(),
            data: "QUJD".to_string(),
        })])
        .expect("must convert");
        let part = &items[0]["content"][0];
        assert_eq!(part["type"], serde_json::json!("input_file"));
        assert_eq!(part["file_data"], serde_json::json!("QUJD"));
        assert_eq!(part["filename"], serde_json::json!("doc.pdf"));

        let items = convert_messages(&[msg(llm_suite_types::message::FileSource::Url {
            url: "https://example.com/a.pdf".to_string(),
        })])
        .expect("must convert");
        assert_eq!(
            items[0]["content"][0]["file_url"],
            serde_json::json!("https://example.com/a.pdf")
        );

        let items = convert_messages(&[msg(llm_suite_types::message::FileSource::FileId {
            file_id: "file_1".to_string(),
        })])
        .expect("must convert");
        assert_eq!(
            items[0]["content"][0]["file_id"],
            serde_json::json!("file_1")
        );
    }

    #[test]
    fn a_file_id_image_uses_file_id_not_image_url() {
        let msg = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: String::new(),
                    detail: None,
                    file_id: Some("file_9".to_string()),
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
        let items = convert_messages(&[msg]).expect("must convert");
        let part = &items[0]["content"][0];
        assert_eq!(part["type"], serde_json::json!("input_image"));
        assert_eq!(part["file_id"], serde_json::json!("file_9"));
        assert!(part.get("image_url").is_none());
    }

    #[test]
    fn a_rich_tool_result_becomes_a_multimodal_function_call_output() {
        let mut assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Text(String::new()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: Some(vec![llm_suite_types::message::LlmToolCall {
                id: "fc_1".to_string(),
                r#type: "function".to_string(),
                call_id: Some("call_1".to_string()),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "screenshot".to_string(),
                    arguments: "{}".to_string(),
                },
                thought_signature: None,
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        assistant.timestamp = 0;
        let mut result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
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
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        result.timestamp = 0;

        let items = convert_messages(&[assistant, result]).expect("must convert");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("output item present");
        let parts = output["output"].as_array().expect("rich output array");
        assert_eq!(parts[0]["type"], serde_json::json!("input_text"));
        assert_eq!(parts[1]["type"], serde_json::json!("input_image"));
        assert_eq!(
            parts[1]["image_url"],
            serde_json::json!("data:image/png;base64,QUJD")
        );
    }

    #[test]
    fn a_tool_result_does_not_repeat_the_text_its_blocks_carry() {
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "the report".to_string(),
                    blocks: Some(vec![MessageContent::Text {
                        text: "the report".to_string(),
                    }]),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        let items = convert_messages(&[result]).expect("must convert");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("output item present");
        let parts = output["output"].as_array().expect("rich output array");
        assert_eq!(
            parts.len(),
            1,
            "the plain text would repeat the text of the block: {parts:?}"
        );
        assert_eq!(parts[0]["type"], serde_json::json!("input_text"));
    }

    #[test]
    fn a_tool_result_with_media_keeps_the_media_next_to_its_text() {
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "see attached".to_string(),
                    blocks: Some(vec![
                        MessageContent::Text {
                            text: "see attached".to_string(),
                        },
                        MessageContent::ImageUrl {
                            image_url: llm_suite_types::message::ImageUrlContent {
                                url: "data:image/png;base64,QUJD".to_string(),
                                detail: None,
                                file_id: None,
                            },
                        },
                    ]),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        let items = convert_messages(&[result]).expect("must convert");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("output item present");
        let parts = output["output"].as_array().expect("rich output array");
        assert_eq!(parts[0]["type"], serde_json::json!("input_text"));
        assert_eq!(
            parts[1]["type"],
            serde_json::json!("input_image"),
            "the media must not be flattened away: {parts:?}"
        );
    }

    #[test]
    fn a_tool_result_with_an_unsupported_block_is_reported() {
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "see attached".to_string(),
                    blocks: Some(vec![MessageContent::Thinking {
                        thinking: "let me verify".to_string(),
                        signature: Some("sig".to_string()),
                    }]),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        let err = convert_messages(&[result])
            .expect_err("a block the format cannot carry must be reported");
        assert!(format!("{err}").contains("thinking"), "{err}");
    }
