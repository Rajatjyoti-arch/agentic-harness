use agentic_harness::{
    DefaultPolicy, ListDirTool, OllamaModel, ReadFileTool, RunCommandTool, ScopedRunner, Tool,
    VerificationConfig, WriteFileTool, run_agent_with_verification,
};
use std::fs;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace = std::env::temp_dir().join("live_ollama_fixture_repair");
    let _ = fs::remove_dir_all(&workspace);
    fs::create_dir_all(workspace.join("src"))?;
    fs::create_dir_all(workspace.join("tests"))?;

    println!("========================================================");
    println!("HARNESS LIVE MODEL TEST: qwen2.5-coder:7b ON OLLAMA");
    println!("Workspace: {}", workspace.display());
    println!("========================================================");

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "reconciliation_fixture"
version = "0.1.0"
edition = "2021"

[dependencies]
"#,
    )?;

    // Immutable test suite with four test cases (under-disbursement, exact match, negative, large scale)
    fs::write(
        workspace.join("tests/reconcile_test.rs"),
        r#"use reconciliation_fixture::calculate_variance;

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
"#,
    )?;

    // Initial broken implementation
    fs::write(
        workspace.join("src/lib.rs"),
        r#"pub fn calculate_variance(sanctioned: i64, disbursed: i64) -> i64 {
    sanctioned + disbursed
}
"#,
    )?;

    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
    let cancel = CancellationToken::new();

    // Pre-build offline metadata
    let prebuild_auth = policy
        .validate_command(
            "cargo",
            &[
                "test".to_string(),
                "--offline".to_string(),
                "--no-run".to_string(),
            ],
        )
        .unwrap();
    let _ = runner.execute(&prebuild_auth, &cancel).await?;

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

    let model_name = "qwen2.5-coder:7b";
    let model = OllamaModel::new("http://localhost:11434", model_name);

    let v_config = VerificationConfig {
        enabled: true,
        command: "cargo".to_string(),
        args: vec!["test".to_string(), "--offline".to_string()],
        max_repairs: 3,
    };

    let task = "You are a software engineer. The project has a failing unit test in tests/reconcile_test.rs. \
    Inspect the failure and examine src/lib.rs. Use write_file to repair src/lib.rs so that \
    calculate_variance(sanctioned: i64, disbursed: i64) -> i64 computes sanctioned - disbursed. \
    You can only write to files inside src/. When finished and tests pass, reply that the task is complete.";

    let start_total = Instant::now();
    let output = run_agent_with_verification(
        &model,
        &tools,
        task,
        policy.as_ref(),
        Some(v_config),
        8,
        cancel.clone(),
    )
    .await?;
    let total_duration = start_total.elapsed();

    println!("\n========================================================");
    println!("LIVE RUN SUMMARY");
    println!("Model: {}", model_name);
    println!("Stop Reason: {:?}", output.stop_reason);
    println!("Turns Executed: {}", output.turns_executed);
    println!("Total Duration: {:?}", total_duration);
    println!("Total Transcript Messages: {}", output.history.len());
    println!("========================================================");

    // Run final cargo test to verify real disk state
    let test_auth = policy
        .validate_command("cargo", &["test".to_string(), "--offline".to_string()])
        .unwrap();
    let test_res = runner.execute(&test_auth, &cancel).await?;
    println!("Final Cargo Test Exit Code: {:?}", test_res.exit_code);
    println!("Final Cargo Test Output:\n{}", test_res.stdout);

    let _ = fs::remove_dir_all(&workspace);
    Ok(())
}
