use crate::types::{Message, StopReason, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Errors encountered during model communication or response decoding.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("HTTP transport error: {0}")]
    Transport(#[from] reqwest::Error),

    #[error("Model provider timeout: {0:?}")]
    Timeout(Duration),

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

/// Configuration for Ollama model provider.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model_name: String,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub num_ctx: usize,
    pub temperature: f64,
}

impl OllamaConfig {
    pub fn new(base_url: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model_name: model_name.into(),
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(300),
            num_ctx: 8192,
            temperature: 0.2,
        }
    }

    pub fn with_timeouts(mut self, connect: Duration, request: Duration) -> Self {
        self.connect_timeout = connect;
        self.request_timeout = request;
        self
    }

    pub fn with_options(mut self, num_ctx: usize, temperature: f64) -> Self {
        self.num_ctx = num_ctx;
        self.temperature = temperature;
        self
    }
}

/// Ollama local language model provider communicating over HTTP `/api/chat`.
pub struct OllamaModel {
    client: reqwest::Client,
    config: OllamaConfig,
}

impl OllamaModel {
    pub fn new(base_url: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self::from_config(OllamaConfig::new(base_url, model_name))
            .expect("Failed to build OllamaModel with default config")
    }

    pub fn from_config(config: OllamaConfig) -> Result<Self, ModelError> {
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .build()?;
        Ok(Self { client, config })
    }

    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    options: Value,
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
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

impl Model for OllamaModel {
    fn respond<'a>(
        &'a self,
        history: &'a [Message],
        tools: &'a [ToolDefinition],
    ) -> BoxFuture<'a, Result<Message, ModelError>> {
        Box::pin(async move {
            let endpoint = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));

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
                    Message::ToolResult {
                        tool_call_id,
                        tool_name,
                        content,
                        ..
                    } => {
                        let mut map = serde_json::Map::new();
                        map.insert("role".to_string(), json!("tool"));
                        map.insert("content".to_string(), json!(content));
                        map.insert("tool_call_id".to_string(), json!(tool_call_id));
                        if let Some(name) = tool_name {
                            map.insert("name".to_string(), json!(name));
                        }
                        messages_payload.push(Value::Object(map));
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
                model: &self.config.model_name,
                messages: messages_payload,
                stream: false,
                tools: tools_payload,
                options: json!({
                    "num_ctx": self.config.num_ctx,
                    "temperature": self.config.temperature,
                }),
            };

            let res = self
                .client
                .post(&endpoint)
                .json(&req)
                .send()
                .await?
                .error_for_status()?;

            let chat_res: OllamaChatResponse = res.json().await?;

            let turn = history
                .iter()
                .filter(|m| matches!(m, Message::Assistant { .. }))
                .count()
                + 1;

            let (tool_calls, path_used) = if !chat_res.message.tool_calls.is_empty() {
                let tcs: Vec<ToolCall> = chat_res
                    .message
                    .tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(idx, tc)| ToolCall {
                        id: format!("call_{turn}_{idx}"),
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    })
                    .collect();
                (tcs, "native message.tool_calls")
            } else if let Some(ref text) = chat_res.message.content
                && let Some(parsed) = parse_single_tool_call_from_text(text, turn)
            {
                (vec![parsed], "text fallback (single JSON object)")
            } else {
                (Vec::new(), "none")
            };

            if path_used != "none" {
                println!(
                    "[Agent Trace] Turn {}: Tool dispatch protocol: {}",
                    turn, path_used
                );
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
                prompt_eval_count: chat_res.prompt_eval_count,
                eval_count: chat_res.eval_count,
            })
        })
    }
}

/// Parses a single tool call JSON object from assistant message text if present.
///
/// Strict rules:
/// - Accepts ONLY a single JSON object that is either the entire message (trimmed)
///   or inside a single fenced code block (` ```json ... ``` ` or ` ``` ... ``` `).
/// - Does NOT extract tool calls described in prose.
/// - Rejects concatenated objects (e.g. `{"name": "a"}{"name": "b"}`).
pub fn parse_single_tool_call_from_text(text: &str, turn: usize) -> Option<ToolCall> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // 1. Check if the entire message is a single JSON object
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        if let Ok(val) = serde_json::from_str::<Value>(trimmed)
            && let Some(obj) = val.as_object()
            && let Some(name_val) = obj.get("name")
            && let Some(name_str) = name_val.as_str()
        {
            let args = match obj.get("arguments") {
                Some(Value::String(s)) => {
                    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                }
                Some(other) => other.clone(),
                None => json!({}),
            };
            return Some(ToolCall {
                id: format!("call_{turn}_0"),
                name: name_str.to_string(),
                arguments: args,
            });
        }
        return None;
    }

    // 2. Check for a single fenced code block
    if let Some(start_idx) = trimmed.find("```") {
        let after_open = &trimmed[start_idx + 3..];
        let block_start = if let Some(rest) = after_open.strip_prefix("json") {
            rest
        } else {
            after_open
        };
        if let Some(end_idx) = block_start.find("```") {
            let code_content = block_start[..end_idx].trim();
            let after_close = &block_start[end_idx + 3..];
            // Ensure no extra fences and that code_content is a single JSON object
            if !after_close.contains("```")
                && code_content.starts_with('{')
                && code_content.ends_with('}')
                && let Ok(val) = serde_json::from_str::<Value>(code_content)
                && let Some(obj) = val.as_object()
                && let Some(name_val) = obj.get("name")
                && let Some(name_str) = name_val.as_str()
            {
                let args = match obj.get("arguments") {
                    Some(Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                    }
                    Some(other) => other.clone(),
                    None => json!({}),
                };
                return Some(ToolCall {
                    id: format!("call_{turn}_0"),
                    name: name_str.to_string(),
                    arguments: args,
                });
            }
        }
    }

    None
}

/// Detects if text contains multiple/concatenated JSON objects.
pub fn has_concatenated_objects(text: &str) -> bool {
    let trimmed = text.trim();
    let content_to_check = if let Some(start_idx) = trimmed.find("```") {
        let after_open = &trimmed[start_idx + 3..];
        let block_start = if let Some(rest) = after_open.strip_prefix("json") {
            rest
        } else {
            after_open
        };
        if let Some(end_idx) = block_start.find("```") {
            block_start[..end_idx].trim()
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    let stream = serde_json::Deserializer::from_str(content_to_check).into_iter::<Value>();
    let mut count = 0;
    for item in stream {
        if let Ok(val) = item {
            if val.is_object() {
                count += 1;
            }
        } else {
            break;
        }
    }
    count > 1
}
