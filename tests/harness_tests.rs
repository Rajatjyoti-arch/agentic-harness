use agentic_harness::{
    DefaultPolicy, ExecutionError, ListDirTool, Message, MockModel, Policy, PolicyDecision,
    ReadFileTool, RunCommandTool, ScopedRunner, StopReason, Tool, ToolCall, WriteFileTool,
    run_agent,
};
use serde_json::json;
use std::fs;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn setup_test_workspace(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("agentic_harness_test_{}", name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn test_t1_traversal_path_denied() {
    let workspace = setup_test_workspace("t1");
    let policy = DefaultPolicy::new(&workspace);
    let tool = ReadFileTool::new(&workspace);

    // 1. Relative path escaping via `..`
    let call_rel = ToolCall {
        id: "call_t1_1".to_string(),
        name: "read_file".to_string(),
        arguments: json!({ "path": "../../etc/passwd" }),
    };
    match policy.evaluate(&call_rel, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(
                reason.contains("resolves outside workspace jail"),
                "Expected traversal rejection, got: {}",
                reason
            );
        }
        other => panic!("Expected PolicyDecision::Deny, got: {:?}", other),
    }

    // 2. Absolute path outside workspace
    let call_abs = ToolCall {
        id: "call_t1_2".to_string(),
        name: "read_file".to_string(),
        arguments: json!({ "path": "/etc/shadow" }),
    };
    match policy.evaluate(&call_abs, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains("resolves outside workspace jail"));
        }
        other => panic!("Expected PolicyDecision::Deny, got: {:?}", other),
    }

    // 3. User home shortcut expansion
    let call_home = ToolCall {
        id: "call_t1_3".to_string(),
        name: "read_file".to_string(),
        arguments: json!({ "path": "~/.bashrc" }),
    };
    match policy.evaluate(&call_home, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains("user home"));
        }
        other => panic!("Expected PolicyDecision::Deny, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_t2_write_outside_jail_denied() {
    let workspace = setup_test_workspace("t2");
    let policy = DefaultPolicy::new(&workspace);
    let tool = WriteFileTool::new(&workspace);

    // 1. Write outside workspace
    let call_outside = ToolCall {
        id: "call_t2_1".to_string(),
        name: "write_file".to_string(),
        arguments: json!({
            "path": "../malicious.sh",
            "content": "echo bad"
        }),
    };
    match policy.evaluate(&call_outside, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains("resolves outside workspace jail"));
        }
        other => panic!("Expected PolicyDecision::Deny, got: {:?}", other),
    }

    // 2. Protected .git access
    let call_git = ToolCall {
        id: "call_t2_2".to_string(),
        name: "write_file".to_string(),
        arguments: json!({
            "path": ".git/hooks/pre-commit",
            "content": "#!/bin/sh\nexit 0"
        }),
    };
    match policy.evaluate(&call_git, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains(".git"));
        }
        other => panic!("Expected PolicyDecision::Deny, got: {:?}", other),
    }

    // 3. Protected .env access
    let call_env = ToolCall {
        id: "call_t2_3".to_string(),
        name: "write_file".to_string(),
        arguments: json!({
            "path": ".env",
            "content": "SECRET=123"
        }),
    };
    match policy.evaluate(&call_env, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains(".env"));
        }
        other => panic!("Expected PolicyDecision::Deny, got: {:?}", other),
    }

    // 4. Valid write inside jail is allowed
    let call_valid = ToolCall {
        id: "call_t2_valid".to_string(),
        name: "write_file".to_string(),
        arguments: json!({
            "path": "src/main.rs",
            "content": "fn main() {}"
        }),
    };
    assert_eq!(policy.evaluate(&call_valid, &tool), PolicyDecision::Allow);

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_t3_timeout_kills_whole_process_group() {
    let workspace = setup_test_workspace("t3");
    let runner = ScopedRunner::new(&workspace).with_timeout(Duration::from_millis(400));
    let cancel = CancellationToken::new();

    // Spawn a shell that spawns a sleep child in the background and waits:
    // Without process-group kill (-pgid), killing the shell orphans the sleep process!
    let marker_tag = "sleep 299";
    let cmd = "sh";
    let args = vec!["-c".to_string(), format!("{} & wait", marker_tag)];

    let result = runner.run(cmd, &args, &cancel).await;
    match result {
        Err(ExecutionError::Timeout(_)) => {}
        other => panic!("Expected ExecutionError::Timeout, got: {:?}", other),
    }

    // Allow kernel process cleanup to settle
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify no surviving orphaned sleep child remains in process table
    let check_output = std::process::Command::new("pgrep")
        .args(["-f", marker_tag])
        .output()
        .expect("Failed to execute pgrep");

    let pids = String::from_utf8_lossy(&check_output.stdout);
    assert!(
        pids.trim().is_empty(),
        "CRITICAL DEFECT: Orphaned child process survived! PIDs found: {}",
        pids
    );

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_t4_output_truncation_flag_set() {
    let workspace = setup_test_workspace("t4");
    // Limit buffer to 1024 bytes
    let runner = ScopedRunner::new(&workspace)
        .with_max_output_bytes(1024)
        .with_timeout(Duration::from_secs(5));
    let cancel = CancellationToken::new();

    // Generate ~4000 bytes of output
    let cmd = "sh";
    let args = vec![
        "-c".to_string(),
        "head -c 4096 /dev/zero | tr '\\0' 'A'".to_string(),
    ];

    let output = runner.run(cmd, &args, &cancel).await.unwrap();

    assert!(
        output.truncated,
        "Expected output.truncated to be true when exceeding buffer"
    );
    assert_eq!(
        output.stdout.len(),
        1024,
        "Expected stdout to be truncated to exactly 1024 bytes"
    );

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_t5_policy_deny_produces_error_observation_and_loop_continues() {
    let workspace = setup_test_workspace("t5");
    let policy = DefaultPolicy::new(&workspace);
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(ReadFileTool::new(&workspace)),
        Box::new(WriteFileTool::new(&workspace)),
    ];

    // Scripted two-turn interaction:
    // Turn 1: Model tries to read an illegal path outside jail.
    // Policy denies it -> ToolResult{ is_error: true }
    // Turn 2: Model observes policy error and self-corrects by writing a valid local file.
    let turn1_call = Message::assistant(
        Some("Attempting to inspect host system configuration...".to_string()),
        vec![ToolCall {
            id: "call_deny_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "../../etc/shadow" }),
        }],
    );

    let turn2_response = Message::assistant_text(
        "I observed that accessing outside the workspace is denied by policy. \
        Task aborted safely without violating boundary.",
    );

    let model = MockModel::with_responses(vec![turn1_call, turn2_response]);
    let cancel = CancellationToken::new();

    let output = run_agent(
        &model,
        &tools,
        "Audit server configuration",
        &policy,
        8,
        cancel,
    )
    .await
    .unwrap();

    assert_eq!(output.stop_reason, StopReason::Stop);
    assert_eq!(output.turns_executed, 2);

    // Verify transcript:
    // [0] System, [1] User, [2] Assistant tool call, [3] ToolResult (Denied), [4] Assistant final
    assert_eq!(output.history.len(), 5);

    match &output.history[3] {
        Message::ToolResult {
            tool_call_id,
            content,
            is_error,
        } => {
            assert_eq!(tool_call_id, "call_deny_1");
            assert!(
                *is_error,
                "Expected ToolResult.is_error to be true for policy denial"
            );
            assert!(content.contains("Policy violation"));
            assert!(content.contains("resolves outside workspace jail"));
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    match &output.history[4] {
        Message::Assistant { content, .. } => {
            let text = content.as_ref().unwrap();
            assert!(text.contains("denied by policy"));
        }
        other => panic!("Expected Assistant message at index 4, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_gated_write_and_run_command_trace() {
    let workspace = setup_test_workspace("gated_run");
    let policy = DefaultPolicy::new(&workspace);
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner.clone(), cancel.clone())),
        Box::new(ListDirTool::new(&workspace)),
    ];

    // Scripted agent workflow:
    // Turn 1: Write a file atomically
    // Turn 2: Run a command executing the file
    // Turn 3: Complete with analysis
    let turn1 = Message::assistant(
        Some("Writing test automation script...".to_string()),
        vec![ToolCall {
            id: "call_write_1".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "hello.sh",
                "content": "echo 'Hello from gated runtime!'"
            }),
        }],
    );

    let turn2 = Message::assistant(
        Some("Executing automation script in ScopedRunner...".to_string()),
        vec![ToolCall {
            id: "call_run_1".to_string(),
            name: "run_command".to_string(),
            arguments: json!({
                "cmd": "sh",
                "args": ["hello.sh"]
            }),
        }],
    );

    let turn3 = Message::assistant_text("Gated execution pipeline completed successfully.");

    let model = MockModel::with_responses(vec![turn1, turn2, turn3]);

    let output = run_agent(
        &model,
        &tools,
        "Run hello world automation",
        &policy,
        8,
        cancel,
    )
    .await
    .unwrap();

    assert_eq!(output.stop_reason, StopReason::Stop);
    assert_eq!(output.turns_executed, 3);
    assert_eq!(output.history.len(), 7);

    // Verify turn 1 write_file produced unified diff
    match &output.history[3] {
        Message::ToolResult {
            tool_call_id,
            content,
            is_error,
        } => {
            assert_eq!(tool_call_id, "call_write_1");
            assert!(!*is_error);
            assert!(content.contains("+++ b/hello.sh"));
            assert!(content.contains("+echo 'Hello from gated runtime!'"));
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    // Verify turn 2 run_command executed inside workspace jail
    match &output.history[5] {
        Message::ToolResult {
            tool_call_id,
            content,
            is_error,
        } => {
            assert_eq!(tool_call_id, "call_run_1");
            assert!(!*is_error);
            assert!(content.contains("Exit Code: 0"));
            assert!(content.contains("Hello from gated runtime!"));
        }
        other => panic!("Expected ToolResult at index 5, got: {:?}", other),
    }

    match &output.history[6] {
        Message::Assistant { content, .. } => {
            assert!(
                content
                    .as_ref()
                    .unwrap()
                    .contains("Gated execution pipeline completed")
            );
        }
        other => panic!("Expected Assistant message at index 6, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}
