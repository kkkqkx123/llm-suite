    use super::*;
    use crate::codecs::anthropic::request::build_body;
    use crate::codecs::anthropic::test_support::{
        msg, profile, request, rich_msg, tool_call, tool_msg,
    };

    #[test]
    fn the_replayed_thinking_block_comes_first() {
        // The API requires the thinking blocks to precede the text and the
        // tool calls of the assistant turn.
        let mut assistant = msg(MessageRole::Assistant, "the answer");
        assistant.thinking = Some("Let me verify".to_string());
        assistant.thinking_signature = Some("sig-abc".to_string());

        let body = build_body(
            &request(vec![msg(MessageRole::User, "hi"), assistant], None),
            &profile(),
        )
        .expect("must build");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], serde_json::json!("thinking"));
        assert_eq!(content[0]["signature"], serde_json::json!("sig-abc"));
        assert_eq!(content[1]["type"], serde_json::json!("text"));
        assert_eq!(content[1]["text"], serde_json::json!("the answer"));
    }

    #[test]
    fn parallel_tool_results_merge_into_one_user_message() {
        let mut assistant = msg(MessageRole::Assistant, "Checking weather and time");
        assistant.tool_calls = Some(vec![
            tool_call("call_1", "get_weather", r#"{"city":"Beijing"}"#),
            tool_call("call_2", "get_time", "{}"),
        ]);

        let body = build_body(
            &request(
                vec![
                    msg(MessageRole::User, "weather and time?"),
                    assistant,
                    tool_msg("call_1", "sunny", false),
                    tool_msg("call_2", "rainy", false),
                ],
                None,
            ),
            &profile(),
        )
        .expect("build must succeed");

        let messages = body["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages.iter().filter_map(|m| m["role"].as_str()).collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "user"],
            "roles must alternate"
        );

        let assistant_blocks = messages[1]["content"].as_array().unwrap();
        assert_eq!(assistant_blocks.len(), 3);
        assert_eq!(assistant_blocks[0]["type"], serde_json::json!("text"));
        assert_eq!(assistant_blocks[1]["id"], serde_json::json!("call_1"));
        assert_eq!(assistant_blocks[2]["id"], serde_json::json!("call_2"));

        let user_blocks = messages[2]["content"].as_array().unwrap();
        assert_eq!(
            user_blocks.len(),
            2,
            "both tool results live in one user message"
        );
        assert_eq!(user_blocks[0]["type"], serde_json::json!("tool_result"));
        assert_eq!(user_blocks[0]["tool_use_id"], serde_json::json!("call_1"));
        assert_eq!(user_blocks[0]["content"], serde_json::json!("sunny"));
        assert_eq!(user_blocks[1]["tool_use_id"], serde_json::json!("call_2"));
        assert!(
            user_blocks[0].get("is_error").is_none(),
            "is_error only set when true"
        );
    }

    #[test]
    fn tool_result_is_error_is_preserved() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![tool_call("call_1", "boom", "{}")]);

        let body = build_body(
            &request(
                vec![
                    msg(MessageRole::User, "run"),
                    assistant,
                    tool_msg("call_1", "exploded", true),
                ],
                None,
            ),
            &profile(),
        )
        .expect("build must succeed");

        let block = &body["messages"][2]["content"][0];
        assert_eq!(block["is_error"], serde_json::json!(true));
    }

    #[test]
    fn thinking_block_replayed_only_with_signature() {
        let with_signature = rich_msg(
            MessageRole::Assistant,
            vec![
                MessageContent::Thinking {
                    thinking: "Let me verify".to_string(),
                    signature: Some("sig-abc".to_string()),
                },
                MessageContent::Text {
                    text: "42".to_string(),
                },
            ],
        );
        let body = build_body(
            &request(vec![msg(MessageRole::User, "hi"), with_signature], None),
            &profile(),
        )
        .expect("build must succeed");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], serde_json::json!("thinking"));
        assert_eq!(content[0]["thinking"], serde_json::json!("Let me verify"));
        assert_eq!(content[0]["signature"], serde_json::json!("sig-abc"));
        assert_eq!(content[1]["type"], serde_json::json!("text"));

        let without_signature = rich_msg(
            MessageRole::Assistant,
            vec![
                MessageContent::Thinking {
                    thinking: "unsigned".to_string(),
                    signature: None,
                },
                MessageContent::Text {
                    text: "42".to_string(),
                },
            ],
        );
        let body = build_body(
            &request(vec![msg(MessageRole::User, "hi"), without_signature], None),
            &profile(),
        )
        .expect("build must succeed");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(
            content.len(),
            1,
            "unsigned thinking must be dropped, not sent"
        );
        assert_eq!(content[0]["type"], serde_json::json!("text"));
    }

    #[test]
    fn https_image_uses_url_source() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/cat.png".to_string(),
                    detail: None,
                    file_id: None,
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("build must succeed");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], serde_json::json!("image"));
        assert_eq!(block["source"]["type"], serde_json::json!("url"));
        assert_eq!(
            block["source"]["url"],
            serde_json::json!("https://example.com/cat.png")
        );
    }

    #[test]
    fn data_uri_image_uses_base64_source_with_validated_media_type() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/webp;base64,QUJD".to_string(),
                    detail: None,
                    file_id: None,
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("build must succeed");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["source"]["type"], serde_json::json!("base64"));
        assert_eq!(
            block["source"]["media_type"],
            serde_json::json!("image/webp")
        );
        assert_eq!(block["source"]["data"], serde_json::json!("QUJD"));
    }

    #[test]
    fn unsupported_image_media_type_is_rejected() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/bmp;base64,QUJD".to_string(),
                    detail: None,
                    file_id: None,
                },
            }],
        );
        let err =
            build_body(&request(vec![user], None), &profile()).expect_err("bmp must be rejected");
        assert!(matches!(err, LlmError::ConfigError(_)));
    }

    #[test]
    fn empty_text_blocks_and_empty_messages_are_skipped() {
        let user = rich_msg(
            MessageRole::User,
            vec![
                MessageContent::Text {
                    text: String::new(),
                },
                MessageContent::Text {
                    text: "hi".to_string(),
                },
            ],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("build must succeed");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], serde_json::json!("hi"));

        // A message that only carries empty text produces no message at all.
        let ghost = rich_msg(
            MessageRole::User,
            vec![MessageContent::Text {
                text: "  ".to_string(),
            }],
        );
        let body = build_body(&request(vec![ghost], None), &profile()).expect("build must succeed");
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn a_stored_thinking_block_comes_before_the_text() {
        // The API requires the thinking blocks to lead the assistant turn,
        // no matter where the history stored them.
        let assistant = rich_msg(
            MessageRole::Assistant,
            vec![
                MessageContent::Text {
                    text: "the answer".to_string(),
                },
                MessageContent::Thinking {
                    thinking: "Let me verify".to_string(),
                    signature: Some("sig-abc".to_string()),
                },
            ],
        );

        let body = build_body(
            &request(vec![msg(MessageRole::User, "hi"), assistant], None),
            &profile(),
        )
        .expect("must build");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], serde_json::json!("thinking"));
        assert_eq!(content[0]["signature"], serde_json::json!("sig-abc"));
        assert_eq!(content[1]["type"], serde_json::json!("text"));
        assert_eq!(content[1]["text"], serde_json::json!("the answer"));
    }

    #[test]
    fn a_base64_pdf_file_input_becomes_a_document_block() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Base64 {
                        media_type: "application/pdf".to_string(),
                        data: "AAAA".to_string(),
                    },
                    filename: Some("report.pdf".to_string()),
                    detail: None,
                    citations_enabled: None,
                    context: None,
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("must build");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], serde_json::json!("document"));
        assert_eq!(block["source"]["type"], serde_json::json!("base64"));
        assert_eq!(
            block["source"]["media_type"],
            serde_json::json!("application/pdf")
        );
        assert_eq!(block["title"], serde_json::json!("report.pdf"));
    }

    #[test]
    fn a_plain_text_file_input_becomes_a_text_document_source() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Base64 {
                        media_type: "text/plain".to_string(),
                        data: "hello".to_string(),
                    },
                    filename: None,
                    detail: None,
                    citations_enabled: None,
                    context: None,
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("must build");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["source"]["type"], serde_json::json!("text"));
        assert_eq!(
            block["source"]["media_type"],
            serde_json::json!("text/plain")
        );
    }

    #[test]
    fn a_file_id_image_becomes_a_file_source() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: String::new(),
                    detail: None,
                    file_id: Some("file_123".to_string()),
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("must build");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], serde_json::json!("image"));
        assert_eq!(block["source"]["type"], serde_json::json!("file"));
        assert_eq!(block["source"]["file_id"], serde_json::json!("file_123"));
    }

    #[test]
    fn audio_input_is_rejected() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::InputAudio {
                input_audio: llm_suite_types::message::InputAudioContent {
                    data: "AAAA".to_string(),
                    format: "wav".to_string(),
                },
            }],
        );
        let err = build_body(&request(vec![user], None), &profile())
            .expect_err("anthropic has no audio input");
        assert!(format!("{err}").contains("audio"), "{err}");
    }

    #[test]
    fn rich_tool_result_blocks_become_a_content_array() {
        let assistant = msg(MessageRole::Assistant, "");
        let mut result = tool_msg("call_1", "see attached", false);
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "see attached".to_string(),
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

        let body =
            build_body(&request(vec![assistant, result], None), &profile()).expect("must build");
        // The assistant turn has no prose, so the tool-result user message
        // is the only wire message.
        let messages = body["messages"].as_array().unwrap();
        let tool_block = &messages[messages.len() - 1]["content"][0];
        assert_eq!(tool_block["type"], serde_json::json!("tool_result"));
        let content = tool_block["content"].as_array().expect("block array");
        // The plain text of the result leads the array, the image follows.
        assert_eq!(content[0]["type"], serde_json::json!("text"));
        assert_eq!(content[0]["text"], serde_json::json!("see attached"));
        assert_eq!(content[1]["type"], serde_json::json!("image"));
        assert_eq!(content[1]["source"]["type"], serde_json::json!("base64"));
    }

    #[test]
    fn a_document_in_a_tool_result_becomes_a_document_block() {
        let assistant = msg(MessageRole::Assistant, "");
        let mut result = tool_msg("call_1", "the report", false);
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "the report".to_string(),
                blocks: Some(vec![MessageContent::FileInput {
                    file: llm_suite_types::message::FileContent {
                        source: llm_suite_types::message::FileSource::Base64 {
                            media_type: "application/pdf".to_string(),
                            data: "QUJD".to_string(),
                        },
                        filename: Some("report.pdf".to_string()),
                        detail: None,
                        citations_enabled: None,
                        context: None,
                    },
                }]),
                is_error: None,
            },
        }]);

        let body =
            build_body(&request(vec![assistant, result], None), &profile()).expect("must build");
        let messages = body["messages"].as_array().unwrap();
        let tool_block = &messages[messages.len() - 1]["content"][0];
        let content = tool_block["content"].as_array().expect("block array");
        // The plain text of the result leads the array, the document
        // follows.
        assert_eq!(content[0]["type"], serde_json::json!("text"));
        assert_eq!(content[1]["type"], serde_json::json!("document"));
        assert_eq!(content[1]["source"]["type"], serde_json::json!("base64"));
        assert_eq!(
            content[1]["source"]["media_type"],
            serde_json::json!("application/pdf")
        );
    }

    #[test]
    fn a_broken_image_in_a_tool_result_is_reported() {
        let assistant = msg(MessageRole::Assistant, "");
        let mut result = tool_msg("call_1", "broken", false);
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "broken".to_string(),
                blocks: Some(vec![MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "data:image/bmp;base64,QUJD".to_string(),
                        detail: None,
                        file_id: None,
                    },
                }]),
                is_error: None,
            },
        }]);

        let err = build_body(&request(vec![assistant, result], None), &profile())
            .expect_err("an unusable image must be reported, not dropped");
        assert!(matches!(err, LlmError::ConfigError(_)), "{err}");
    }

    #[test]
    fn a_document_built_from_blocks_uses_the_content_source() {        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Content {
                        blocks: vec![
                            MessageContent::Text {
                                text: "Chapter 1".to_string(),
                            },
                            MessageContent::ImageUrl {
                                image_url: llm_suite_types::message::ImageUrlContent {
                                    url: "data:image/png;base64,QUJD".to_string(),
                                    detail: None,
                                    file_id: None,
                                },
                            },
                        ],
                    },
                    filename: Some("manual".to_string()),
                    detail: None,
                    citations_enabled: Some(true),
                    context: Some("the product manual".to_string()),
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("must build");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], serde_json::json!("document"));
        assert_eq!(block["source"]["type"], serde_json::json!("content"));
        let content = block["source"]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(content[0]["type"], serde_json::json!("text"));
        assert_eq!(content[1]["type"], serde_json::json!("image"));
        assert_eq!(block["citations"]["enabled"], serde_json::json!(true));
        assert_eq!(block["context"], serde_json::json!("the product manual"));
    }

    #[test]
    fn a_native_block_is_replayed_verbatim() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::Native {
                provider: "anthropic".to_string(),
                kind: "search_result".to_string(),
                payload: serde_json::json!({
                    "type": "search_result",
                    "source": "https://example.com",
                    "title": "Example",
                    "content": [{"type": "text", "text": "found"}],
                }),
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("must build");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], serde_json::json!("search_result"));
        assert_eq!(block["title"], serde_json::json!("Example"));
    }

    #[test]
    fn a_native_block_of_another_provider_is_rejected() {
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::Native {
                provider: "openai".to_string(),
                kind: "refusal".to_string(),
                payload: serde_json::json!({"type": "refusal", "refusal": "no"}),
            }],
        );
        let err = build_body(&request(vec![user], None), &profile())
            .expect_err("a block of another provider cannot be carried");
        assert!(format!("{err}").contains("openai"), "{err}");
    }

    #[test]
    fn a_tool_result_does_not_repeat_the_text_its_blocks_carry() {
        let assistant = msg(MessageRole::Assistant, "");
        let mut result = tool_msg("call_1", "the report", false);
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "the report".to_string(),
                blocks: Some(vec![MessageContent::Text {
                    text: "the report".to_string(),
                }]),
                is_error: None,
            },
        }]);

        let body =
            build_body(&request(vec![assistant, result], None), &profile()).expect("must build");
        let messages = body["messages"].as_array().unwrap();
        let tool_block = &messages[messages.len() - 1]["content"][0];
        let content = tool_block["content"].as_array().expect("block array");
        assert_eq!(
            content.len(),
            1,
            "the plain text would repeat the text of the block: {content:?}"
        );
        assert_eq!(content[0]["type"], serde_json::json!("text"));
    }
