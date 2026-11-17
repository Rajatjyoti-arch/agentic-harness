use agentic_harness::{
    DefaultPolicy, ListDirTool, Message, OllamaConfig, OllamaModel, ReadFileTool, RunCommandTool,
    ScopedRunner, Tool, VerificationConfig, WriteFileTool, run_agent_with_verification,
    verify_libtest_summary,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

const VERBATIM_PROMPT: &str = "The test suite in tests/ is failing. Diagnose the failure and fix the implementation in src/. You may not modify files in tests/.";

const SYSTEM_PROMPT: &str = "You are an autonomous engineering assistant operating within a sandboxed runtime. \
Execute tool calls cleanly, inspect outputs, and respect security policy bounds. \
If verification fails, inspect the diagnostics and issue a repair.";

fn compute_sha256(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("Failed to execute sha256sum");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string()
}

fn compute_diff(pristine_src: &str, final_path: &Path) -> String {
    let temp_pristine = std::env::temp_dir().join("pristine_lib.rs");
    let _ = fs::write(&temp_pristine, pristine_src);
    let output = Command::new("diff")
        .arg("-u")
        .arg(&temp_pristine)
        .arg(final_path)
        .output();
    let _ = fs::remove_file(&temp_pristine);
    match output {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout).to_string();
            if s.is_empty() {
                "No changes (identical)".to_string()
            } else {
                s
            }
        }
        Err(e) => format!("Failed to compute diff: {}", e),
    }
}

fn setup_fresh_fixture(workspace: &Path) -> (String, String) {
    let _ = fs::remove_dir_all(workspace);
    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::create_dir_all(workspace.join("tests")).unwrap();

    let cargo_toml = r#"[package]
name = "reconciliation_fixture"
version = "0.1.0"
edition = "2021"

[dependencies]
"#;
    fs::write(workspace.join("Cargo.toml"), cargo_toml).unwrap();

    let test_content = r#"use reconciliation_fixture::calculate_variance;

#[test]
fn test_variance_under_disbursement() {
    assert_eq!(calculate_variance(5000, 3500), 1500);
}

#[test]
fn test_variance_exact_match() {
    assert_eq!(calculate_variance(10000, 10000), 0);
}

#[test]
fn test_variance_over_disbursement() {
    assert_eq!(calculate_variance(3000, 4500), -1500);
}

#[test]
fn test_variance_large_scale() {
    assert_eq!(calculate_variance(100_000_000, 50_000_000), 50_000_000);
}
"#;
    fs::write(workspace.join("tests/reconcile_test.rs"), test_content).unwrap();

    let broken_src = r#"pub fn calculate_variance(sanctioned: i64, disbursed: i64) -> i64 {
    sanctioned + disbursed
}
"#;
    fs::write(workspace.join("src/lib.rs"), broken_src).unwrap();

    let sha_before = compute_sha256(&workspace.join("tests/reconcile_test.rs"));
    (broken_src.to_string(), sha_before)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_name = "hf.co/IamLucif3r/Qwen3-4B-Instruct-2507-Abliterated:Q4_K_M".to_string();
    let mut total_runs = 3usize;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_name = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--runs" && i + 1 < args.len() {
            total_runs = args[i + 1].parse().unwrap_or(3);
            i += 2;
        } else {
            i += 1;
        }
    }

    println!("========================================================");
    println!("HARNESS LIVE MODEL RUNS: {}", model_name);
    println!("Num Ctx: 8192 | Temperature: 0.2 | Timeout: 300s");
    println!("Total Runs: {}", total_runs);
    println!("========================================================");

    fs::create_dir_all("docs/traces")?;

    for run_idx in 1..=total_runs {
        println!("\n>>> STARTING RUN {} / {} <<<", run_idx, total_runs);
        let workspace = PathBuf::from(format!("/tmp/live_qwen3_4b_run_{}", run_idx));
        let (pristine_src, sha_before) = setup_fresh_fixture(&workspace);

        let policy = Arc::new(DefaultPolicy::new(&workspace));
        let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
        let cancel = CancellationToken::new();

        // 1. Pristine Baseline Verification BEFORE Turn 1
        let baseline_auth = policy
            .validate_command("cargo", &["test".to_string(), "--offline".to_string()])
            .expect("Failed to authorize baseline verification command");
        let base_out = runner.execute(&baseline_auth, &cancel).await?;

        // Requirement A2: Baseline MUST FAIL on pristine tree (if it passes: abort "nothing to fix")
        if verify_libtest_summary(&base_out.stdout, None) {
            panic!("ABORT: Baseline verification passed on pristine tree: nothing to fix!");
        }
        println!(
            "[Run {}] Baseline verification confirmed FAILING as required.",
            run_idx
        );

        // Feed baseline verification output into transcript
        let first_user_message = format!(
            "{}\n\nBaseline verification:\n{}",
            VERBATIM_PROMPT, base_out.stdout
        );

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(ReadFileTool::new(&workspace)),
            Box::new(WriteFileTool::new(&workspace)),
            Box::new(ListDirTool::new(&workspace)),
            Box::new(RunCommandTool::new(
                runner.clone(),
                policy.clone(),
                cancel.clone(),
            )),
        ];

        let config = OllamaConfig::new("http://localhost:11434", &model_name)
            .with_options(8192, 0.2)
            .with_timeouts(Duration::from_secs(5), Duration::from_secs(300));
        let model = OllamaModel::from_config(config)?;

        let v_config = VerificationConfig {
            enabled: true,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "--offline".to_string()],
            max_repairs: 3,
        };

        let start_time = Instant::now();
        let output_res = run_agent_with_verification(
            &model,
            &tools,
            &first_user_message,
            policy.as_ref(),
            Some(v_config),
            8,
            cancel.clone(),
        )
        .await;
        let duration = start_time.elapsed();

        let sha_after = compute_sha256(&workspace.join("tests/reconcile_test.rs"));
        assert_eq!(
            sha_before, sha_after,
            "Security invariant violation: tests/ was modified!"
        );

        let final_diff = compute_diff(&pristine_src, &workspace.join("src/lib.rs"));

        let trace_path = format!("docs/traces/live_qwen3_4b_run{}.txt", run_idx);

        match output_res {
            Ok(output) => {
                println!(
                    "[Run {} RESULT] Stop Reason: {:?}, Turns: {}, Duration: {:?}",
                    run_idx, output.stop_reason, output.turns_executed, duration
                );

                // Build complete, unsummarized transcript
                let mut trace = String::new();
                trace.push_str("================================================================================\n");
                trace.push_str(&format!("LIVE MODEL TRACE — RUN {}\n", run_idx));
                trace.push_str(&format!("Model: {}\n", model_name));
                trace.push_str("Options: num_ctx=8192, temperature=0.2, request_timeout=300s\n");
                trace.push_str(&format!("Date/Time: {}\n", chrono_lite_timestamp()));
                trace.push_str(&format!("Stop Reason: {:?}\n", output.stop_reason));
                trace.push_str(&format!("Turns Executed: {}\n", output.turns_executed));
                trace.push_str(&format!("Wall Time: {:?}\n", duration));
                trace.push_str(&format!("Tests/ SHA256 Before: {}\n", sha_before));
                trace.push_str(&format!("Tests/ SHA256 After:  {}\n", sha_after));
                trace.push_str("================================================================================\n\n");

                trace.push_str("--- PROMPTS ---\n");
                trace.push_str(&format!("[System Prompt]\n{}\n\n", SYSTEM_PROMPT));
                trace.push_str(&format!("[First User Message]\n{}\n\n", first_user_message));

                trace.push_str("--- FULL TURN-BY-TURN TRANSCRIPT ---\n");
                let mut turn_counter = 0usize;
                for msg in &output.history {
                    match msg {
                        Message::System { content } => {
                            trace.push_str(&format!("ROLE: SYSTEM\n{}\n\n", content));
                        }
                        Message::User { content } => {
                            trace.push_str(&format!("ROLE: USER\n{}\n\n", content));
                        }
                        Message::Assistant {
                            content,
                            tool_calls,
                            stop_reason,
                            prompt_eval_count,
                            eval_count,
                        } => {
                            turn_counter += 1;
                            trace.push_str(&format!("ROLE: ASSISTANT (Turn {})\n", turn_counter));
                            trace.push_str(&format!(
                                "prompt_eval_count: {}\n",
                                prompt_eval_count
                                    .map(|c| c.to_string())
                                    .unwrap_or_else(|| "unknown".to_string())
                            ));
                            trace.push_str(&format!(
                                "eval_count: {}\n",
                                eval_count
                                    .map(|c| c.to_string())
                                    .unwrap_or_else(|| "unknown".to_string())
                            ));
                            trace.push_str(&format!("stop_reason: {:?}\n", stop_reason));
                            if let Some(text) = content {
                                trace.push_str(&format!("Raw Assistant Text:\n{}\n", text));
                            }
                            if !tool_calls.is_empty() {
                                trace.push_str(&format!(
                                    "Tool Calls ({}):\n{}\n",
                                    tool_calls.len(),
                                    serde_json::to_string_pretty(&tool_calls).unwrap()
                                ));
                            }
                            trace.push('\n');
                        }
                        Message::ToolResult {
                            tool_call_id,
                            tool_name,
                            content,
                            is_error,
                        } => {
                            trace.push_str(&format!(
                                "ROLE: TOOL RESULT (id: {}, name: {:?}, is_error: {})\n{}\n\n",
                                tool_call_id, tool_name, is_error, content
                            ));
                        }
                    }
                }

                trace.push_str("--- UNIFIED DIFF OF src/ (Pristine vs Final) ---\n");
                trace.push_str(&final_diff);
                trace.push_str("\n\n================================================================================\n");

                fs::write(&trace_path, trace)?;
                println!("[Run {}] Full trace written to {}", run_idx, trace_path);
            }
            Err(err) => {
                println!(
                    "[Run {} FAILED] Agent Error: {:?}, Duration: {:?}",
                    run_idx, err, duration
                );
                let mut trace = String::new();
                trace.push_str(&format!("LIVE MODEL TRACE — RUN {} FAILED\n", run_idx));
                trace.push_str(&format!("Model: {}\n", model_name));
                trace.push_str(&format!("Error: {:?}\n", err));
                trace.push_str(&format!("Wall Time: {:?}\n", duration));
                fs::write(&trace_path, trace)?;
            }
        }

        let _ = fs::remove_dir_all(&workspace);
    }

    Ok(())
}

fn chrono_lite_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("Unix Epoch: {}s", duration.as_secs())
}
