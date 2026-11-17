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

    #[error("Baseline test compilation failed: {0}")]
    BaselineCompilationFailed(String),

    #[error("Baseline test suite is empty")]
    BaselineEmpty,
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
    println!("[Agent Trace] Model request options: num_ctx=8192, temperature=0.2");
    println!(
        "[Agent Trace] Registered tools: {:?}",
        tool_definitions.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    let mut repair_attempts = 0usize;
    let mut format_retry_count = 0usize;

    // Baseline Test Suite Collection (captured from pristine workspace before any model edits)
    let mut baseline_test_names = std::collections::HashSet::new();
    if let Some(v_config) = verification.as_ref().filter(|v| v.enabled)
        && v_config.command == "cargo"
        && let Some(run_tool) = tools.iter().find(|t| t.name() == "run_command")
    {
        let baseline_list_call = ToolCall {
            id: "baseline_list".to_string(),
            name: "run_command".to_string(),
            arguments: json!({
                "cmd": "cargo",
                "args": ["test", "--offline", "--locked", "--", "--list"]
            }),
        };
        if let PolicyDecision::AllowCommand(_) =
            policy.evaluate(&baseline_list_call, run_tool.as_ref())
        {
            match run_tool.run(baseline_list_call.arguments).await {
                Ok(base_list_out) => {
                    if (base_list_out.contains("Exit Code:")
                        && !base_list_out.contains("Exit Code: 0"))
                        || base_list_out.contains("error[E")
                        || base_list_out.contains("error: could not compile")
                    {
                        println!(
                            "[Agent Trace] Baseline compilation failed:\n{}",
                            base_list_out
                        );
                        return Err(AgentError::BaselineCompilationFailed(base_list_out));
                    }
                    baseline_test_names = extract_test_names_from_list(&base_list_out);
                    if baseline_test_names.is_empty() {
                        return Err(AgentError::BaselineEmpty);
                    }
                    println!(
                        "[Agent Trace] Pristine baseline test suite captured ({} tests: {:?})",
                        baseline_test_names.len(),
                        baseline_test_names
                    );
                }
                Err(err) => return Err(AgentError::BaselineCompilationFailed(err)),
            }
        }
    }

    for turn in 1..=max_turns {
        if cancel_token.is_cancelled() {
            println!("[Agent Trace] Run cancelled by token.");
            return Ok(AgentOutput {
                history,
                stop_reason: StopReason::Cancelled,
                turns_executed: turn,
            });
        }

        println!("\n[Agent Trace] === Turn {} / {} ===", turn, max_turns);

        let model_res = tokio::select! {
            res = model.respond(&history, &tool_definitions) => res,
            _ = cancel_token.cancelled() => {
                println!("[Agent Trace] Turn {}: Cancelled mid-model call.", turn);
                return Ok(AgentOutput {
                    history,
                    stop_reason: StopReason::Cancelled,
                    turns_executed: turn,
                });
            }
        };

        let mut assistant_msg = match model_res {
            Ok(msg) => msg,
            Err(ModelError::Transport(ref e)) if e.is_timeout() => {
                println!("[Agent Trace] Turn {}: Model provider timed out.", turn);
                return Ok(AgentOutput {
                    history,
                    stop_reason: StopReason::ProviderTimeout,
                    turns_executed: turn,
                });
            }
            Err(ModelError::Timeout(_)) => {
                println!("[Agent Trace] Turn {}: Model provider timed out.", turn);
                return Ok(AgentOutput {
                    history,
                    stop_reason: StopReason::ProviderTimeout,
                    turns_executed: turn,
                });
            }
            Err(err) => return Err(AgentError::Model(err)),
        };

        // Enforce consistent per-turn unique tool call IDs: call_{turn}_{index}
        if let Message::Assistant {
            ref mut tool_calls, ..
        } = assistant_msg
        {
            for (idx, tc) in tool_calls.iter_mut().enumerate() {
                tc.id = format!("call_{turn}_{idx}");
            }
        }

        match &assistant_msg {
            Message::Assistant {
                content,
                tool_calls,
                stop_reason,
                prompt_eval_count,
                eval_count,
            } => {
                let p_str = prompt_eval_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let e_str = eval_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".to_string());

                // Local estimate: serialized bytes / 3 over messages + tool schemas
                let messages_bytes = serde_json::to_string(&history)
                    .map(|s| s.len())
                    .unwrap_or(0);
                let tools_bytes = serde_json::to_string(&tool_definitions)
                    .map(|s| s.len())
                    .unwrap_or(0);
                let local_estimate = (messages_bytes + tools_bytes) / 3;

                let reported_total = match (prompt_eval_count, eval_count) {
                    (Some(p), Some(e)) => Some(p + e),
                    (Some(p), None) => Some(*p),
                    (None, Some(e)) => Some(*e),
                    (None, None) => None,
                };

                let effective_tokens = match reported_total {
                    Some(rep) => rep.max(local_estimate),
                    None => local_estimate,
                };

                println!(
                    "[Agent Trace] Turn {}: Token usage: prompt_eval_count={}, eval_count={}, local_estimate={}, effective_tokens={}",
                    turn, p_str, e_str, local_estimate, effective_tokens
                );

                // Abort with distinct StopReason::ContextNearLimit above 90% of num_ctx (8192 * 0.9 = 7372)
                if effective_tokens > (8192 * 9 / 10) {
                    println!(
                        "[Agent Trace] Turn {}: ABORT: effective_tokens ({}) exceeded 90% of num_ctx (7372). Stopping with ContextNearLimit.",
                        turn, effective_tokens
                    );
                    history.push(assistant_msg.clone());
                    return Ok(AgentOutput {
                        history,
                        stop_reason: StopReason::ContextNearLimit,
                        turns_executed: turn,
                    });
                }

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
                    if let Some(text) = content
                        && crate::model::has_concatenated_objects(text)
                        && format_retry_count < 1
                    {
                        format_retry_count += 1;
                        println!(
                            "[Agent Trace] Turn {}: Concatenated JSON objects rejected. Triggering format repair retry 1/1.",
                            turn
                        );
                        history.push(Message::user(
                            "Formatting Error: Concatenated tool calls are rejected. Please emit only a single JSON tool call object per turn (Formatting Retry 1/1)."
                        ));
                        continue;
                    }

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
                                    history.push(Message::tool_result_full(
                                        &tool_call.id,
                                        Some(tool_call.name.clone()),
                                        format!("Policy violation: {}", reason),
                                        true,
                                    ));
                                }
                                PolicyDecision::RequireApproval(reason) => {
                                    println!(
                                        "[Agent Trace] Turn {}: POLICY DECISION: REQUIRE APPROVAL -> {}",
                                        turn, reason
                                    );
                                    history.push(Message::tool_result_full(
                                        &tool_call.id,
                                        Some(tool_call.name.clone()),
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
                                        history.push(Message::tool_result_full(
                                            &tool_call.id,
                                            Some(tool_call.name.clone()),
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
                                            history.push(Message::tool_result_full(
                                                &tool_call.id,
                                                Some(tool_call.name.clone()),
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
                                            history.push(Message::tool_result_full(
                                                &tool_call.id,
                                                Some(tool_call.name.clone()),
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
                                        history.push(Message::tool_result_full(
                                            &tool_call.id,
                                            Some(tool_call.name.clone()),
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
                                                                        Some(&baseline_test_names),
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

                                            history.push(Message::tool_result_full(
                                                &tool_call.id,
                                                Some(tool_call.name.clone()),
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
                                            history.push(Message::tool_result_full(
                                                &tool_call.id,
                                                Some(tool_call.name.clone()),
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
                            history.push(Message::tool_result_full(
                                &tool_call.id,
                                Some(tool_call.name.clone()),
                                msg,
                                true,
                            ));
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

/// Extracts all test names from test output lines matching `test <name> ... <status>`.
pub fn extract_test_names(output: &str) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("test ")
            && let Some(idx) = rest.find(" ... ")
        {
            let name = rest[..idx].trim();
            if !name.is_empty() {
                names.insert(name.to_string());
            }
        }
    }
    names
}

/// Extracts all test names from `cargo test -- --list` output.
///
/// Parses lines ending with `: test` (e.g. `test_foo: test` or `tests::test_foo: test`).
pub fn extract_test_names_from_list(output: &str) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_suffix(": test") {
            let test_name = name.trim();
            if !test_name.is_empty() {
                names.insert(test_name.to_string());
            }
        }
    }
    names
}

/// Parses the libtest summary across all test suites in verification output.
///
/// Validates:
/// 1. Command exit code is 0 (if reported in output).
/// 2. For each test suite reported in stdout:
///    - Sums passed, failed, ignored, and filtered out counts across all `test result:` lines.
///    - Requires exactly 0 failed, 0 ignored, and 0 filtered out.
///    - Validates that each individual test `test <name> ... ok` appears exactly once with unique names.
/// 3. If a pristine baseline test suite was captured:
///    - Every baseline test name must be present in the passed test names set exactly once.
///    - Total passed must equal the baseline test count.
///    - Any deleted test or #[ignore] test causes verification to FAIL.
pub fn verify_libtest_summary(
    output: &str,
    baseline: Option<&std::collections::HashSet<String>>,
) -> bool {
    // If output contains an exit code marker, ensure it is 0
    if output.contains("Exit Code:") && !output.contains("Exit Code: 0") {
        return false;
    }

    let mut passed_names = std::collections::HashSet::new();
    let mut total_passed = 0usize;
    let mut total_failed = 0usize;
    let mut total_ignored = 0usize;
    let mut total_filtered_out = 0usize;
    let mut summary_lines_count = 0usize;

    for line in output.lines() {
        let trimmed = line.trim();

        // Track passed test names: "test <name> ... ok"
        if let Some(rest) = trimmed.strip_prefix("test ")
            && let Some(name) = rest.strip_suffix(" ... ok")
        {
            let test_name = name.trim();
            if !test_name.is_empty() && !passed_names.insert(test_name.to_string()) {
                return false; // Duplicate test name execution record
            }
        }

        // Sum across ALL "test result:" summary lines
        if let Some(rest) = trimmed.strip_prefix("test result:") {
            summary_lines_count += 1;
            let parts: Vec<&str> = rest.split(';').collect();
            if parts.len() >= 5 {
                let passed = parts[0]
                    .split_whitespace()
                    .filter_map(|w| w.parse::<usize>().ok())
                    .next_back()
                    .unwrap_or(0);
                let failed = parts[1]
                    .split_whitespace()
                    .filter_map(|w| w.parse::<usize>().ok())
                    .next_back()
                    .unwrap_or(0);
                let ignored = parts[2]
                    .split_whitespace()
                    .filter_map(|w| w.parse::<usize>().ok())
                    .next_back()
                    .unwrap_or(0);
                let filtered = parts[4]
                    .split_whitespace()
                    .filter_map(|w| w.parse::<usize>().ok())
                    .next_back()
                    .unwrap_or(0);

                total_passed += passed;
                total_failed += failed;
                total_ignored += ignored;
                total_filtered_out += filtered;
            } else {
                return false;
            }
        }
    }

    // Strict verifier requirements:
    // 1. At least one summary line parsed and at least 1 test passed
    // 2. Exactly 0 failed
    // 3. Exactly 0 ignored
    // 4. Exactly 0 filtered out
    // 5. Total passed count equals unique test names recorded
    if summary_lines_count == 0
        || total_passed == 0
        || total_failed != 0
        || total_ignored != 0
        || total_filtered_out != 0
        || passed_names.len() != total_passed
    {
        return false;
    }

    // 6. If baseline test suite was captured, require EVERY baseline test name present exactly once
    if let Some(expected_baseline) = baseline
        && !expected_baseline.is_empty()
    {
        if total_passed != expected_baseline.len() || passed_names.len() != expected_baseline.len()
        {
            return false;
        }
        for base_name in expected_baseline {
            if !passed_names.contains(base_name) {
                return false;
            }
        }
    }

    true
}
