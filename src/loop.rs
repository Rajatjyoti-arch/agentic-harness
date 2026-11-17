use crate::model::{Model, ModelError};
use crate::policy::{Policy, PolicyDecision};
use crate::tool::Tool;
use crate::types::{Message, StopReason, ToolCall, ToolDefinition};
use serde_json::json;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

/// Execution errors that occur during the agent loop.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Model failure: {0}")]
    Model(#[from] ModelError),
}

/// Configuration for automated verification checks executed after mutating tool actions.
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    pub enabled: bool,
    pub command: String,
    pub args: Vec<String>,
    pub max_repairs: usize,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "--quiet".to_string()],
            max_repairs: 3,
        }
    }
}

/// The result returned after agent execution finishes.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    /// Full transcript of messages exchanged during the run.
    pub history: Vec<Message>,
    /// The condition under which the run terminated.
    pub stop_reason: StopReason,
    /// Number of turns executed.
    pub turns_executed: usize,
}

/// Executes the agent loop with default settings.
pub async fn run(
    model: &dyn Model,
    tools: &[Box<dyn Tool>],
    task: &str,
    policy: &dyn Policy,
    max_turns: usize,
    cancel_token: CancellationToken,
) -> Result<AgentOutput, AgentError> {
    run_with_verification(model, tools, task, policy, None, max_turns, cancel_token).await
}

/// Executes the agent loop with policy-gated tool execution, length-truncation guards,
/// cooperative cancellation, and post-mutation self-repair verification cycles.
pub async fn run_with_verification(
    model: &dyn Model,
    tools: &[Box<dyn Tool>],
    task: &str,
    policy: &dyn Policy,
    verification: Option<VerificationConfig>,
    max_turns: usize,
    cancel_token: CancellationToken,
) -> Result<AgentOutput, AgentError> {
    let mut history = vec![
        Message::system(
            "You are an autonomous engineering assistant operating within a sandboxed runtime. \
            Execute tool calls cleanly, inspect outputs, and respect security policy bounds. \
            If verification fails, inspect the diagnostics and issue a repair.",
        ),
        Message::user(task),
    ];

    let tool_definitions: Vec<ToolDefinition> = tools.iter().map(|t| t.definition()).collect();

    println!(
        "[Agent Trace] Initializing agent loop for task: \"{}\"",
        task
    );
    println!(
        "[Agent Trace] Registered tools: {:?}",
        tool_definitions.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    let mut repair_attempts = 0usize;

    for turn in 1..=max_turns {
        if cancel_token.is_cancelled() {
            println!("[Agent Trace] Run cancelled by token.");
            return Ok(AgentOutput {
                history,
                stop_reason: StopReason::Error,
                turns_executed: turn,
            });
        }

        println!("\n[Agent Trace] === Turn {} / {} ===", turn, max_turns);

        let assistant_msg = model.respond(&history, &tool_definitions).await?;

        match &assistant_msg {
            Message::Assistant {
                content,
                tool_calls,
                stop_reason,
            } => {
                if let Some(text) = content {
                    println!(
                        "[Agent Trace] Turn {}: Assistant thought/message:\n{}",
                        turn, text
                    );
                }

                // Guard: If model stopped due to length truncation, do NOT execute tool calls
                if *stop_reason == Some(StopReason::Length) {
                    println!(
                        "[Agent Trace] Turn {}: GUARD TRIGGERED: Model response truncated (StopReason::Length). \
                        Aborting tool execution to prevent executing partial/malformed payloads.",
                        turn
                    );
                    history.push(assistant_msg.clone());
                    return Ok(AgentOutput {
                        history,
                        stop_reason: StopReason::Length,
                        turns_executed: turn,
                    });
                }

                history.push(assistant_msg.clone());

                if tool_calls.is_empty() {
                    println!(
                        "[Agent Trace] Turn {}: No tools requested. Task complete.",
                        turn
                    );
                    return Ok(AgentOutput {
                        history,
                        stop_reason: StopReason::Stop,
                        turns_executed: turn,
                    });
                }

                for tool_call in tool_calls {
                    println!(
                        "[Agent Trace] Turn {}: Model proposed tool '{}' (id: {}) with args: {}",
                        turn, tool_call.name, tool_call.id, tool_call.arguments
                    );

                    let matching_tool = tools.iter().find(|t| t.name() == tool_call.name);

                    match matching_tool {
                        Some(tool) => {
                            // Evaluate Policy BEFORE execution
                            let decision = policy.evaluate(tool_call, tool.as_ref());
                            match decision {
                                PolicyDecision::Deny(reason) => {
                                    println!(
                                        "[Agent Trace] Turn {}: POLICY DECISION: DENY -> {}",
                                        turn, reason
                                    );
                                    // Append error observation and continue loop
                                    history.push(Message::tool_result(
                                        &tool_call.id,
                                        format!("Policy violation: {}", reason),
                                        true,
                                    ));
                                }
                                PolicyDecision::RequireApproval(reason) => {
                                    println!(
                                        "[Agent Trace] Turn {}: POLICY DECISION: REQUIRE APPROVAL -> {}",
                                        turn, reason
                                    );
                                    history.push(Message::tool_result(
                                        &tool_call.id,
                                        format!("Policy requires human approval: {}", reason),
                                        true,
                                    ));
                                }
                                PolicyDecision::Allow => {
                                    println!("[Agent Trace] Turn {}: POLICY DECISION: ALLOW", turn);
                                    let start = Instant::now();
                                    match tool.run(tool_call.arguments.clone()).await {
                                        Ok(mut out) => {
                                            let duration = start.elapsed();
                                            println!(
                                                "[Agent Trace] Turn {}: Tool '{}' executed successfully in {:?} ({} bytes output)",
                                                turn,
                                                tool_call.name,
                                                duration,
                                                out.len()
                                            );

                                            // Self-repair verification trigger after mutating actions
                                            let mut verification_failed = false;
                                            if let Some(v_config) =
                                                verification.as_ref().filter(|v| {
                                                    v.enabled && tool.name() == "write_file"
                                                })
                                            {
                                                println!(
                                                    "[Agent Trace] Turn {}: Mutating action detected. Triggering automated verification: '{} {:?}'",
                                                    turn, v_config.command, v_config.args
                                                );

                                                let verify_call = ToolCall {
                                                    id: format!("verify_call_{turn}"),
                                                    name: "run_command".to_string(),
                                                    arguments: json!({
                                                        "cmd": v_config.command,
                                                        "args": v_config.args
                                                    }),
                                                };

                                                if let Some(run_tool) =
                                                    tools.iter().find(|t| t.name() == "run_command")
                                                {
                                                    let v_decision = policy
                                                        .evaluate(&verify_call, run_tool.as_ref());
                                                    println!(
                                                        "[Agent Trace] Turn {}: Verification command POLICY DECISION: {:?}",
                                                        turn, v_decision
                                                    );

                                                    if v_decision == PolicyDecision::Allow {
                                                        let v_start = Instant::now();
                                                        match run_tool
                                                            .run(verify_call.arguments)
                                                            .await
                                                        {
                                                            Ok(v_out) => {
                                                                let v_dur = v_start.elapsed();
                                                                if v_out.contains("Exit Code: 0") {
                                                                    println!(
                                                                        "[Agent Trace] Turn {}: Automated verification PASSED in {:?}",
                                                                        turn, v_dur
                                                                    );
                                                                    out.push_str(&format!("\n\n[AUTOMATED VERIFICATION PASSED in {:?}]\n{}", v_dur, v_out));
                                                                } else {
                                                                    repair_attempts += 1;
                                                                    println!(
                                                                        "[Agent Trace] Turn {}: Automated verification FAILED (Attempt {}/{}) in {:?}: Exit code non-zero",
                                                                        turn,
                                                                        repair_attempts,
                                                                        v_config.max_repairs,
                                                                        v_dur
                                                                    );
                                                                    out.push_str(&format!(
                                                                                "\n\n[AUTOMATED VERIFICATION FAILED - REPAIR REQUIRED (Attempt {}/{})]\n{}",
                                                                                repair_attempts, v_config.max_repairs, v_out
                                                                            ));
                                                                    verification_failed = true;
                                                                }
                                                            }
                                                            Err(v_err) => {
                                                                repair_attempts += 1;
                                                                println!(
                                                                    "[Agent Trace] Turn {}: Verification execution error (Attempt {}/{}): {}",
                                                                    turn,
                                                                    repair_attempts,
                                                                    v_config.max_repairs,
                                                                    v_err
                                                                );
                                                                out.push_str(&format!(
                                                                            "\n\n[AUTOMATED VERIFICATION FAILED - EXECUTION ERROR (Attempt {}/{})]\n{}",
                                                                            repair_attempts, v_config.max_repairs, v_err
                                                                        ));
                                                                verification_failed = true;
                                                            }
                                                        }
                                                    } else {
                                                        println!(
                                                            "[Agent Trace] Turn {}: Verification command blocked by policy",
                                                            turn
                                                        );
                                                    }
                                                }
                                            }

                                            history.push(Message::tool_result(
                                                &tool_call.id,
                                                out,
                                                verification_failed,
                                            ));

                                            if let Some(v_config) =
                                                verification.as_ref().filter(|v| {
                                                    verification_failed
                                                        && repair_attempts >= v.max_repairs
                                                })
                                            {
                                                println!(
                                                    "[Agent Trace] Turn {}: Maximum verification repair attempts ({}) exhausted. Terminating.",
                                                    turn, v_config.max_repairs
                                                );
                                                return Ok(AgentOutput {
                                                    history,
                                                    stop_reason: StopReason::RepairExhausted,
                                                    turns_executed: turn,
                                                });
                                            }
                                        }
                                        Err(err) => {
                                            println!(
                                                "[Agent Trace] Turn {}: Tool '{}' failed: {}",
                                                turn, tool_call.name, err
                                            );
                                            history.push(Message::tool_result(
                                                &tool_call.id,
                                                format!("Tool execution error: {}", err),
                                                true,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            let msg = format!("Error: tool '{}' not found", tool_call.name);
                            println!("[Agent Trace] Turn {}: {}", turn, msg);
                            history.push(Message::tool_result(&tool_call.id, msg, true));
                        }
                    }
                }
            }
            _ => {
                return Ok(AgentOutput {
                    history,
                    stop_reason: StopReason::Error,
                    turns_executed: turn,
                });
            }
        }
    }

    println!(
        "[Agent Trace] Reached maximum allowed turns ({}) without final completion.",
        max_turns
    );
    Ok(AgentOutput {
        history,
        stop_reason: StopReason::MaxTurns,
        turns_executed: max_turns,
    })
}
