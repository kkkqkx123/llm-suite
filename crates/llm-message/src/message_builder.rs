use llm_types::message::{ImageUrlContent, Message, MessageContent, MessageContentValue};

pub fn user_text(text: impl Into<String>) -> Message {
    Message::user_text(text.into())
}

pub fn system_text(text: impl Into<String>) -> Message {
    Message::system_text(text.into())
}

pub fn assistant_text(text: impl Into<String>) -> Message {
    let mut msg = Message::user_text(text.into());
    msg.role = llm_types::message::MessageRole::Assistant;
    msg
}

/// Build a single `ImageUrl` content block.
pub fn image_url(url: impl Into<String>, detail: Option<String>) -> MessageContent {
    MessageContent::ImageUrl {
        image_url: ImageUrlContent {
            url: url.into(),
            detail,
        },
    }
}

/// Build a user message carrying text plus images.
///
/// Falls back to the plain-text form when no images are given.
pub fn user_text_with_images(
    text: impl Into<String>,
    images: impl IntoIterator<Item = ImageUrlContent>,
) -> Message {
    let mut msg = Message::user_text(text.into());
    let images: Vec<MessageContent> = images
        .into_iter()
        .map(|image_url| MessageContent::ImageUrl { image_url })
        .collect();
    if !images.is_empty() {
        let text = match msg.content {
            MessageContentValue::Text(text) => text,
            other => return Message { content: other, ..msg },
        };
        let mut blocks = Vec::with_capacity(images.len() + 1);
        if !text.is_empty() {
            blocks.push(MessageContent::Text { text });
        }
        blocks.extend(images);
        msg.content = MessageContentValue::Rich(blocks);
    }
    msg
}

pub fn tool_result_message(tool_call_id: impl Into<String>, content: impl Into<String>) -> Message {
    Message::tool_result(tool_call_id.into(), None, content.into(), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_types::message::{MessageContentValue, MessageRole};

    #[test]
    fn user_text_builds_user_role_message() {
        let msg = user_text("hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, MessageContentValue::Text("hello".to_string()));
        assert!(msg.tool_call_id.is_none());
        assert!(msg.tool_calls.is_none());
        assert!(msg.timestamp > 0);
    }

    #[test]
    fn system_text_builds_system_role_message() {
        let msg = system_text("be helpful");
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(
            msg.content,
            MessageContentValue::Text("be helpful".to_string())
        );
    }

    #[test]
    fn tool_result_message_carries_call_id() {
        let msg = tool_result_message("call_42", "the result");
        assert_eq!(msg.role, MessageRole::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_42"));
        assert_eq!(
            msg.content,
            MessageContentValue::Text("the result".to_string())
        );
    }

    #[test]
    fn assistant_text_builds_assistant_role_message() {
        let msg = assistant_text("hi there");
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.content, MessageContentValue::Text("hi there".to_string()));
    }

    #[test]
    fn image_url_builds_image_block() {
        let block = image_url("https://example.com/a.png", Some("auto".to_string()));
        match block {
            MessageContent::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/a.png");
                assert_eq!(image_url.detail.as_deref(), Some("auto"));
            }
            other => panic!("expected ImageUrl, got {:?}", other),
        }
    }

    #[test]
    fn user_text_with_images_builds_rich_content() {
        let msg = user_text_with_images(
            "look",
            [
                ImageUrlContent { url: "data:image/png;base64,AA".to_string(), detail: None },
                ImageUrlContent { url: "https://e.com/b.jpg".to_string(), detail: Some("low".to_string()) },
            ],
        );
        assert_eq!(msg.role, MessageRole::User);
        match msg.content {
            MessageContentValue::Rich(blocks) => {
                assert_eq!(blocks.len(), 3);
                assert!(matches!(&blocks[0], MessageContent::Text { text } if text == "look"));
                assert!(matches!(&blocks[1], MessageContent::ImageUrl { .. }));
                assert!(matches!(&blocks[2], MessageContent::ImageUrl { .. }));
            }
            other => panic!("expected Rich, got {:?}", other),
        }
    }

    #[test]
    fn user_text_with_images_without_images_stays_text() {
        let msg = user_text_with_images("plain", std::iter::empty());
        assert_eq!(msg.content, MessageContentValue::Text("plain".to_string()));
    }
}
