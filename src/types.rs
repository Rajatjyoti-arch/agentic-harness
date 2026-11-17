use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a model-directed tool invocation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call execution.
    pub id: String,
    /// Function/tool name to invoke (e.g., "read_file", "run_command").
    pub name: String,
    /// JSON arguments passed by the model.
    pub arguments: Value,
}

/// A conversation turn in the agent execution trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// System prompt setting agent instructions and persona.
    System { content: String },
    /// User query or input prompt.
    User { content: String },
    /// Model response, containing text and/or tool invocation requests.
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<StopReason>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_eval_count: Option<usize>,
    },
    /// Result returned to the model after tool execution.
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            content,
            tool_calls,
            stop_reason: None,
            prompt_eval_count: None,
        }
    }

    pub fn assistant_with_stop_reason(
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
        stop_reason: Option<StopReason>,
    ) -> Self {
        Self::Assistant {
            content,
            tool_calls,
            stop_reason,
            prompt_eval_count: None,
        }
    }

    pub fn assistant_with_eval_count(
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
        prompt_eval_count: Option<usize>,
    ) -> Self {
        Self::Assistant {
            content,
            tool_calls,
            stop_reason: None,
            prompt_eval_count,
        }
    }

    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            tool_calls: Vec::new(),
            stop_reason: None,
            prompt_eval_count: None,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            is_error,
        }
    }
}

/// The termination condition for an agent loop invocation or model turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Assistant completed response without requesting further tools.
    Stop,
    /// Model requested tool executions.
    ToolUse,
    /// Model generation truncated by maximum token limit (partial JSON risk).
    Length,
    /// Verification check repeatedly failed and exhausted maximum self-repair attempts.
    RepairExhausted,
    /// Loop reached configured maximum turn budget.
    MaxTurns,
    /// Loop stopped due to an unrecoverable model or execution error.
    Error,
    /// Model provider request timed out.
    ProviderTimeout,
    /// Execution cancelled cooperatively.
    Cancelled,
    /// Context window usage exceeded safe threshold (>90% num_ctx).
    ContextNearLimit,
}

/// Schema definition of a tool made available to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}
