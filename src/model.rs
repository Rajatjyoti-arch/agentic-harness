use crate::types::{Message, StopReason, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Errors encountered during model communication or response decoding.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("HTTP transport error: {0}")]
    Transport(#[from] reqwest::Error),

    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Model protocol error: {0}")]
    Protocol(String),
}

/// Abstract contract for language model providers.
pub trait Model: Send + Sync {
    /// Generates the next response message given conversation history and available tools.
    fn respond<'a>(
        &'a self,
        history: &'a [Message],
        tools: &'a [ToolDefinition],
    ) -> BoxFuture<'a, Result<Message, ModelError>>;
}

/// Scripted mock model for offline execution, unit tests, and deterministic demo runs.
#[derive(Default)]
pub struct MockModel {
    responses: Vec<Message>,
    index: AtomicUsize,
}

impl MockModel {
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
            index: AtomicUsize::new(0),
        }
    }

    pub fn with_responses(responses: Vec<Message>) -> Self {
        Self {
            responses,
            index: AtomicUsize::new(0),
        }
    }
}

impl Model for MockModel {
    fn respond<'a>(
        &'a self,
        _history: &'a [Message],
        _tools: &'a [ToolDefinition],
    ) -> BoxFuture<'a, Result<Message, ModelError>> {
        Box::pin(async move {
            let idx = self.index.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                Ok(Message::assistant_text("Task completed."))
            }
        })
    }
}

/// Ollama local language model provider communicating over HTTP `/api/chat`.
pub struct OllamaModel {
    client: reqwest::Client,
    base_url: String,
    model_name: String,
}

impl OllamaModel {
    pub fn new(base_url: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model_name: model_name.into(),
        }
    }
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
}

#[derive(Deserialize)]
struct OllamaFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Deserialize)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Deserialize)]
struct OllamaMessageResponse {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessageResponse,
    #[serde(default)]
    done_reason: Option<String>,
}

impl Model for OllamaModel {
    fn respond<'a>(
        &'a self,
        history: &'a [Message],
        tools: &'a [ToolDefinition],
    ) -> BoxFuture<'a, Result<Message, ModelError>> {
        Box::pin(async move {
            let endpoint = format!("{}/api/chat", self.base_url.trim_end_matches('/'));

            let mut messages_payload = Vec::new();
            for msg in history {
                match msg {
                    Message::System { content } => {
                        messages_payload.push(json!({
                            "role": "system",
                            "content": content,
                        }));
                    }
                    Message::User { content } => {
                        messages_payload.push(json!({
                            "role": "user",
                            "content": content,
                        }));
                    }
                    Message::Assistant {
                        content,
                        tool_calls,
                        ..
                    } => {
                        if tool_calls.is_empty() {
                            messages_payload.push(json!({
                                "role": "assistant",
                                "content": content.clone().unwrap_or_default(),
                            }));
                        } else {
                            let ollama_tools: Vec<Value> = tool_calls
                                .iter()
                                .map(|tc| {
                                    json!({
                                        "function": {
                                            "name": tc.name,
                                            "arguments": tc.arguments,
                                        }
                                    })
                                })
                                .collect();
                            messages_payload.push(json!({
                                "role": "assistant",
                                "content": content.clone().unwrap_or_default(),
                                "tool_calls": ollama_tools,
                            }));
                        }
                    }
                    Message::ToolResult { content, .. } => {
                        messages_payload.push(json!({
                            "role": "tool",
                            "content": content,
                        }));
                    }
                }
            }

            let tools_payload: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();

            let req = OllamaChatRequest {
                model: &self.model_name,
                messages: messages_payload,
                stream: false,
                tools: tools_payload,
            };

            let res = self
                .client
                .post(&endpoint)
                .json(&req)
                .send()
                .await?
                .error_for_status()?;

            let chat_res: OllamaChatResponse = res.json().await?;

            let mut tool_calls: Vec<ToolCall> = chat_res
                .message
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(idx, tc)| ToolCall {
                    id: format!("call_{idx}"),
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                })
                .collect();

            // Robust fallback for local models that output JSON tool calls in message.content
            if tool_calls.is_empty()
                && let Some(ref text) = chat_res.message.content
            {
                let trimmed = text.trim();
                let cleaned = if let Some(stripped) = trimmed.strip_prefix("```json") {
                    stripped.strip_suffix("```").unwrap_or(stripped).trim()
                } else if let Some(stripped) = trimmed.strip_prefix("```") {
                    stripped.strip_suffix("```").unwrap_or(stripped).trim()
                } else {
                    trimmed
                };

                let stream = serde_json::Deserializer::from_str(cleaned).into_iter::<Value>();
                for val in stream.flatten() {
                    if let Some(obj) = val.as_object() {
                        if let (Some(name_val), Some(args_val)) =
                            (obj.get("name"), obj.get("arguments"))
                            && let Some(name_str) = name_val.as_str()
                        {
                            let args = if let Some(s) = args_val.as_str() {
                                serde_json::from_str(s).unwrap_or_else(|_| args_val.clone())
                            } else {
                                args_val.clone()
                            };
                            let idx = tool_calls.len();
                            tool_calls.push(ToolCall {
                                id: format!("call_{idx}"),
                                name: name_str.to_string(),
                                arguments: args,
                            });
                        }
                    } else if let Some(arr) = val.as_array() {
                        for item in arr {
                            if let (Some(name_val), Some(args_val)) =
                                (item.get("name"), item.get("arguments"))
                                && let Some(name_str) = name_val.as_str()
                            {
                                let args = if let Some(s) = args_val.as_str() {
                                    serde_json::from_str(s).unwrap_or_else(|_| args_val.clone())
                                } else {
                                    args_val.clone()
                                };
                                let idx = tool_calls.len();
                                tool_calls.push(ToolCall {
                                    id: format!("call_{idx}"),
                                    name: name_str.to_string(),
                                    arguments: args,
                                });
                            }
                        }
                    }
                }
            }

            let stop_reason = match chat_res.done_reason.as_deref() {
                Some("length") => Some(StopReason::Length),
                Some("stop") if !tool_calls.is_empty() => Some(StopReason::ToolUse),
                Some("stop") => Some(StopReason::Stop),
                _ => None,
            };

            Ok(Message::Assistant {
                content: chat_res.message.content,
                tool_calls,
                stop_reason,
            })
        })
    }
}
