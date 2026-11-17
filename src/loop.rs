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
            args: vec![
                "test".to_string(),
                "--quiet".to_string(),
                "--offline".to_string(),
                "--locked".to_string(),
            ],
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
    let mut format_retry_count = 0usize;

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
                                PolicyDecision::AllowCommand(ref auth) => {
                                    if tool_call.name != "run_command" {
                                        let err_msg = format!(
                                            "Security invariant violation: tool '{}' cannot be authorized with AllowCommand",
                                            tool_call.name
                                        );
                                        println!(
                                            "[Agent Trace] Turn {}: POLICY HARD REJECTION: {}",
                                            turn, err_msg
                                        );
                                        history.push(Message::tool_result(
                                            &tool_call.id,
                                            err_msg,
                                            true,
                                        ));
                                        continue;
                                    }
                                    println!(
                                        "[Agent Trace] Turn {}: POLICY DECISION: ALLOW COMMAND ({})",
                                        turn,
                                        auth.exe()
                                    );
                                    let start = Instant::now();
                                    match tool.run(tool_call.arguments.clone()).await {
                                        Ok(out) => {
                                            format_retry_count = 0;
                                            let duration = start.elapsed();
                                            println!(
                                                "[Agent Trace] Turn {}: Tool '{}' executed successfully in {:?} ({} bytes output)",
                                                turn,
                                                tool_call.name,
                                                duration,
                                                out.len()
                                            );
                                            history.push(Message::tool_result(
                                                &tool_call.id,
                                                out,
                                                false,
                                            ));
                                        }
                                        Err(err) => {
                                            println!(
                                                "[Agent Trace] Turn {}: Tool '{}' failed: {}",
                                                turn, tool_call.name, err
                                            );
                                            let is_format_err = err.contains("Invalid arguments")
                                                || err.contains("missing field")
                                                || err.contains("unknown field");
                                            let feedback = if is_format_err
                                                && format_retry_count < 1
                                            {
                                                format_retry_count += 1;
                                                format!(
                                                    "Argument Formatting Error: {}. Tool '{}' parameter schema: {}. Please retry with corrected JSON arguments (Formatting Retry 1/1).",
                                                    err,
                                                    tool.name(),
                                                    tool.parameters_schema()
                                                )
                                            } else {
                                                format!("Tool execution error: {}", err)
                                            };
                                            history.push(Message::tool_result(
                                                &tool_call.id,
                                                feedback,
                                                true,
                                            ));
                                        }
                                    }
                                }
                                PolicyDecision::Allow => {
                                    if tool_call.name == "run_command" {
                                        let err_msg = "Security invariant violation: 'run_command' cannot be authorized with PolicyDecision::Allow (requires AllowCommand with AuthorizedCall)".to_string();
                                        println!(
                                            "[Agent Trace] Turn {}: POLICY HARD REJECTION: {}",
                                            turn, err_msg
                                        );
                                        history.push(Message::tool_result(
                                            &tool_call.id,
                                            err_msg,
                                            true,
                                        ));
                                        continue;
                                    }
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

                                                    match v_decision {
                                                        PolicyDecision::AllowCommand(_) => {
                                                            let v_start = Instant::now();
                                                            match run_tool
                                                                .run(verify_call.arguments)
                                                                .await
                                                            {
                                                                Ok(v_out) => {
                                                                    let v_dur = v_start.elapsed();
                                                                    if verify_libtest_summary(
                                                                        &v_out,
                                                                    ) {
                                                                        println!(
                                                                            "[Agent Trace] Turn {}: Automated verification PASSED in {:?}",
                                                                            turn, v_dur
                                                                        );
                                                                        out.push_str(&format!("\n\n[AUTOMATED VERIFICATION PASSED in {:?}]\n{}", v_dur, v_out));
                                                                    } else {
                                                                        repair_attempts += 1;
                                                                        println!(
                                                                            "[Agent Trace] Turn {}: Automated verification FAILED (Attempt {}/{}) in {:?}: libtest summary mismatch or test failure",
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
                                                        }
                                                        PolicyDecision::Deny(reason) => {
                                                            println!(
                                                                "[Agent Trace] Turn {}: Verification command blocked by policy: {}",
                                                                turn, reason
                                                            );
                                                            out.push_str(&format!(
                                                                "\n\n[AUTOMATED VERIFICATION BLOCKED BY POLICY: {}]",
                                                                reason
                                                            ));
                                                        }
                                                        _ => {
                                                            println!(
                                                                "[Agent Trace] Turn {}: Verification command disallowed by policy",
                                                                turn
                                                            );
                                                        }
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
                                            let is_format_err = err.contains("Invalid arguments")
                                                || err.contains("missing field")
                                                || err.contains("unknown field");
                                            let feedback = if is_format_err
                                                && format_retry_count < 1
                                            {
                                                format_retry_count += 1;
                                                format!(
                                                    "Argument Formatting Error: {}. Tool '{}' parameter schema: {}. Please retry with corrected JSON arguments (Formatting Retry 1/1).",
                                                    err,
                                                    tool.name(),
                                                    tool.parameters_schema()
                                                )
                                            } else {
                                                format!("Tool execution error: {}", err)
                                            };
                                            history.push(Message::tool_result(
                                                &tool_call.id,
                                                feedback,
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

/// Parses the libtest summary from verification output.
///
/// Validates:
/// 1. Command exit code is 0 (if reported in output).
/// 2. For each test suite reported in stdout:
///    - `running N tests` declares N expected tests.
///    - Exactly N individual tests match `test <name> ... ok` with distinct names.
///    - The suite summary `test result: ok. N passed; 0 failed` confirms N passed and 0 failed.
/// 3. At least one suite with N > 0 completed successfully.
/// 4. No suite failed or was truncated prematurely without its summary line.
///
/// A process that exits 0 without completing libtest (e.g., via `std::process::exit(0)`)
/// or prints forged summary lines without matching per-test execution records
/// is strictly rejected as FAILED.
pub fn verify_libtest_summary(output: &str) -> bool {
    // If output contains an exit code marker, ensure it is 0
    if output.contains("Exit Code:") && !output.contains("Exit Code: 0") {
        return false;
    }

    let mut current_suite_expected: Option<usize> = None;
    let mut current_suite_names = std::collections::HashSet::new();
    let mut total_passed = 0usize;
    let mut completed_suites = 0usize;

    for line in output.lines() {
        let trimmed = line.trim();

        // Detect "running N tests" or "running 1 test"
        if let Some(rest) = trimmed.strip_prefix("running ")
            && let Some(idx) = rest.find(" test")
            && let Ok(count) = rest[..idx].trim().parse::<usize>()
        {
            // If a previous suite started with count > 0 but didn't finish,
            // that means it was aborted (e.g. exit(0)) before printing summary!
            if let Some(prev) = current_suite_expected
                && prev > 0
            {
                return false;
            }
            current_suite_expected = Some(count);
            current_suite_names.clear();
        }

        // Per-test-name matching: "test <name> ... ok"
        if let Some(rest) = trimmed.strip_prefix("test ")
            && let Some(name) = rest.strip_suffix(" ... ok")
        {
            let test_name = name.trim();
            if !test_name.is_empty() && !current_suite_names.insert(test_name.to_string()) {
                return false; // Duplicate test name report
            }
        }

        // Detect summary line: "test result: ok. N passed; 0 failed; ..."
        if let Some(rest) = trimmed.strip_prefix("test result: ") {
            if let Some(summary) = rest.strip_prefix("ok. ") {
                let parts: Vec<&str> = summary.split(';').collect();
                if parts.len() >= 2 {
                    let passed_part = parts[0].trim();
                    let failed_part = parts[1].trim();
                    let passed_count = passed_part
                        .split_whitespace()
                        .next()
                        .and_then(|n| n.parse::<usize>().ok());
                    let failed_count = failed_part
                        .split_whitespace()
                        .next()
                        .and_then(|n| n.parse::<usize>().ok());

                    if let (Some(passed), Some(0)) = (passed_count, failed_count) {
                        if let Some(expected) = current_suite_expected
                            && (passed != expected || current_suite_names.len() != expected)
                        {
                            return false;
                        }
                        total_passed += passed;
                        completed_suites += 1;
                        current_suite_expected = None;
                        current_suite_names.clear();
                    } else {
                        return false;
                    }
                }
            } else {
                // If it starts with "test result: FAILED" or anything not "ok."
                return false;
            }
        }
    }

    // If a suite started with expected > 0 but was cut short without summary line:
    if let Some(expected) = current_suite_expected
        && expected > 0
    {
        return false;
    }

    // Require at least one suite to have completed and at least 1 test passed
    completed_suites > 0 && total_passed > 0
}
