use agentic_harness::{
    DefaultPolicy, ExecutionError, ListDirArgs, ListDirTool, Message, MockModel, Policy,
    PolicyDecision, ReadFileArgs, ReadFileTool, RunCommandArgs, RunCommandTool, ScopedRunner,
    StopReason, Tool, ToolCall, VerificationConfig, WriteFileArgs, WriteFileTool, run_agent,
    run_agent_with_verification,
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
async fn test_t3_rewritten_timeout_kills_grandchild_process_group() {
    let workspace = setup_test_workspace("t3_rewritten");
    // Explicitly test raw ScopedRunner process group killing without container layers
    let runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_timeout(Duration::from_millis(500));
    let cancel = CancellationToken::new();

    let pid_file = workspace.join("grandchild.pid");
    let pid_path_str = pid_file.to_string_lossy().to_string();

    // Spawn a shell that spawns a grandchild writing its own PID to a file and sleeping:
    // Without process group kill (-pgid), killing the shell orphans the grandchild sleep process!
    let cmd = "sh";
    let script = format!(
        "sh -c 'echo $$ > \"{}\" && exec sleep 300' & wait",
        pid_path_str
    );
    let args = vec!["-c".to_string(), script];

    let result = runner.run(cmd, &args, &cancel).await;
    match result {
        Err(ExecutionError::Timeout(_)) => {}
        other => panic!("Expected ExecutionError::Timeout, got: {:?}", other),
    }

    // Verify the grandchild PID file was written before the timeout fired
    assert!(
        pid_file.exists(),
        "Grandchild failed to write PID file before timeout"
    );
    let pid_content = fs::read_to_string(&pid_file).expect("Failed to read PID file");
    let grandchild_pid: i32 = pid_content
        .trim()
        .parse()
        .expect("Failed to parse integer PID from grandchild file");

    assert!(grandchild_pid > 0);

    // Allow process termination to settle in the kernel
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Kernel-level existence probe: kill(pid, 0)
    // If process is dead, kill(pid, 0) returns -1 and errno is ESRCH (No such process).
    let kill_probe = unsafe { libc::kill(grandchild_pid, 0) };
    let errno = std::io::Error::last_os_error();

    assert_eq!(
        kill_probe, -1,
        "Grandchild process {} is still running! Expected ESRCH.",
        grandchild_pid
    );
    assert_eq!(
        errno.raw_os_error(),
        Some(libc::ESRCH),
        "Expected errno ESRCH (3), got: {:?}",
        errno
    );

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_t4_output_truncation_flag_set() {
    let workspace = setup_test_workspace("t4");
    let runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_max_output_bytes(1024)
        .with_timeout(Duration::from_secs(5));
    let cancel = CancellationToken::new();

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

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_new_a_run_command_interpreter_denied_by_default() {
    let workspace = setup_test_workspace("interp_deny");
    let policy = DefaultPolicy::new(&workspace);
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();
    let tool = RunCommandTool::new(runner, cancel);

    let call_sh = ToolCall {
        id: "call_sh_deny".to_string(),
        name: "run_command".to_string(),
        arguments: json!({
            "cmd": "sh",
            "args": ["-c", "echo hi"]
        }),
    };

    match policy.evaluate(&call_sh, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(
                reason.contains("arbitrary scripting is prohibited"),
                "Expected interpreter rejection, got: {}",
                reason
            );
        }
        other => panic!("Expected PolicyDecision::Deny for sh, got: {:?}", other),
    }

    let call_python = ToolCall {
        id: "call_py_deny".to_string(),
        name: "run_command".to_string(),
        arguments: json!({
            "cmd": "python3",
            "args": ["-c", "import os; print(os.getcwd())"]
        }),
    };

    match policy.evaluate(&call_python, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains("arbitrary scripting is prohibited"));
        }
        other => panic!(
            "Expected PolicyDecision::Deny for python3, got: {:?}",
            other
        ),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_new_b_allowlisted_executable_permitted() {
    let workspace = setup_test_workspace("allowlisted_permitted");
    let policy = DefaultPolicy::new(&workspace);
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();
    let tool = RunCommandTool::new(runner, cancel);

    // ls is in ALLOWED_EXECUTABLES
    let call_ls = ToolCall {
        id: "call_ls".to_string(),
        name: "run_command".to_string(),
        arguments: json!({
            "cmd": "ls",
            "args": ["-la"]
        }),
    };
    assert_eq!(policy.evaluate(&call_ls, &tool), PolicyDecision::Allow);

    // git status is allowed (read-only)
    let call_git_status = ToolCall {
        id: "call_git".to_string(),
        name: "run_command".to_string(),
        arguments: json!({
            "cmd": "git",
            "args": ["status"]
        }),
    };
    assert_eq!(
        policy.evaluate(&call_git_status, &tool),
        PolicyDecision::Allow
    );

    // git push is BLOCKED
    let call_git_push = ToolCall {
        id: "call_git_push".to_string(),
        name: "run_command".to_string(),
        arguments: json!({
            "cmd": "git",
            "args": ["push", "origin", "main"]
        }),
    };
    match policy.evaluate(&call_git_push, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains("only read-only subcommands"));
        }
        other => panic!("Expected Deny for git push, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_new_c_cancellation_token_kills_group() {
    let workspace = setup_test_workspace("cancel_group");
    let runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_timeout(Duration::from_secs(30));
    let cancel = CancellationToken::new();

    let pid_file = workspace.join("child_sleep.pid");
    let pid_path_str = pid_file.to_string_lossy().to_string();

    let cancel_trigger = cancel.clone();
    let cmd = "sh";
    let args = vec![
        "-c".to_string(),
        format!(
            "sh -c 'echo $$ > \"{}\" && exec sleep 300' & wait",
            pid_path_str
        ),
    ];

    // Spawn async task that cancels the token after 150ms
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(150)).await;
        cancel_trigger.cancel();
    });

    let result = runner.run(cmd, &args, &cancel).await;
    match result {
        Err(ExecutionError::Cancelled) => {}
        other => panic!("Expected ExecutionError::Cancelled, got: {:?}", other),
    }

    // Verify child PID was recorded and is dead
    assert!(pid_file.exists());
    let pid: i32 = fs::read_to_string(&pid_file)
        .unwrap()
        .trim()
        .parse()
        .unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;
    let probe = unsafe { libc::kill(pid, 0) };
    let errno = std::io::Error::last_os_error();

    assert_eq!(probe, -1);
    assert_eq!(errno.raw_os_error(), Some(libc::ESRCH));

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_new_d_tool_schemas_match_serde_structs() {
    // 1. ReadFileArgs: valid round-trip and unknown field rejection
    let valid_read = json!({ "path": "src/main.rs" });
    let parsed_read: Result<ReadFileArgs, _> = serde_json::from_value(valid_read);
    assert!(parsed_read.is_ok());

    let invalid_read = json!({ "path": "src/main.rs", "extra_bad_field": 123 });
    let err_read: Result<ReadFileArgs, _> = serde_json::from_value(invalid_read);
    assert!(err_read.is_err(), "Must reject unknown fields");

    // 2. WriteFileArgs: valid round-trip and unknown field rejection
    let valid_write = json!({ "path": "src/main.rs", "content": "fn main() {}" });
    let parsed_write: Result<WriteFileArgs, _> = serde_json::from_value(valid_write);
    assert!(parsed_write.is_ok());

    let invalid_write = json!({ "path": "src/main.rs", "content": "fn main() {}", "mode": 0o777 });
    let err_write: Result<WriteFileArgs, _> = serde_json::from_value(invalid_write);
    assert!(err_write.is_err(), "Must reject unknown fields");

    // 3. ListDirArgs: valid round-trip and unknown field rejection
    let valid_list = json!({ "path": "tests" });
    let parsed_list: Result<ListDirArgs, _> = serde_json::from_value(valid_list);
    assert!(parsed_list.is_ok());

    let invalid_list = json!({ "path": "tests", "recursive": true });
    let err_list: Result<ListDirArgs, _> = serde_json::from_value(invalid_list);
    assert!(err_list.is_err(), "Must reject unknown fields");

    // 4. RunCommandArgs: valid round-trip and unknown field rejection
    let valid_cmd = json!({ "cmd": "cargo", "args": ["test"] });
    let parsed_cmd: Result<RunCommandArgs, _> = serde_json::from_value(valid_cmd);
    assert!(parsed_cmd.is_ok());

    let invalid_cmd = json!({ "cmd": "cargo", "args": ["test"], "env": {"FOO": "BAR"} });
    let err_cmd: Result<RunCommandArgs, _> = serde_json::from_value(invalid_cmd);
    assert!(err_cmd.is_err(), "Must reject unknown fields");
}

#[tokio::test]
async fn test_self_repair_loop_full_lifecycle_and_trace() {
    let workspace = setup_test_workspace("self_repair_demo");
    let policy = DefaultPolicy::new(&workspace);
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner.clone(), cancel.clone())),
        Box::new(ListDirTool::new(&workspace)),
        Box::new(ReadFileTool::new(&workspace)),
    ];

    // Scripted multi-turn agent demonstrating the exact demo scenario:
    // Turn 1: DENY — Model tries unconfined interpreter execution `sh -c 'rm -rf *'`
    // Turn 2: ALLOW + FAILED VERIFICATION — Model writes a file with a deliberate syntax error
    // Turn 3: REPAIR + PASSED VERIFICATION — Model inspects compiler error and fixes the file
    // Turn 4: Final completion
    let turn1_call = Message::assistant(
        Some("Attempting to run clean script via shell interpreter...".to_string()),
        vec![ToolCall {
            id: "call_deny_sh".to_string(),
            name: "run_command".to_string(),
            arguments: json!({
                "cmd": "sh",
                "args": ["-c", "cat /etc/passwd"]
            }),
        }],
    );

    let turn2_call = Message::assistant(
        Some("Shell was denied by policy. Creating a Rust code module with an initial implementation...".to_string()),
        vec![ToolCall {
            id: "call_write_broken".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "test_script.sh",
                "content": "echo 'Triggering intentional test failure'\nexit 1"
            }),
        }],
    );

    let turn3_call = Message::assistant(
        Some(
            "Observed verification test failure. Repairing the script to return success..."
                .to_string(),
        ),
        vec![ToolCall {
            id: "call_write_fixed".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "test_script.sh",
                "content": "echo 'Repaired implementation'\nexit 0"
            }),
        }],
    );

    let turn4_response =
        Message::assistant_text("Verification passed! Self-repair cycle complete.");

    let model = MockModel::with_responses(vec![turn1_call, turn2_call, turn3_call, turn4_response]);

    // Configure verification to execute test_script.sh via bwrap/cat or cargo
    // In this test, we configure verification to check test_script.sh
    let v_config = VerificationConfig {
        enabled: true,
        command: "sh".to_string(), // In ScopedRunner when invoked for verification
        args: vec!["test_script.sh".to_string()],
        max_repairs: 3,
    };

    // To allow the verification runner to execute the test script while keeping policy gate for agent calls,
    // we use a policy that allows the verification script
    struct TestPolicyWithVerification {
        inner: DefaultPolicy,
    }
    impl Policy for TestPolicyWithVerification {
        fn evaluate(&self, call: &ToolCall, tool: &dyn Tool) -> PolicyDecision {
            if call.id.starts_with("verify_call_") {
                // Verification commands are permitted by test harness
                PolicyDecision::Allow
            } else {
                self.inner.evaluate(call, tool)
            }
        }
    }

    let composite_policy = TestPolicyWithVerification { inner: policy };

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Implement and verify script automation",
        &composite_policy,
        Some(v_config),
        8,
        cancel,
    )
    .await
    .unwrap();

    assert_eq!(output.stop_reason, StopReason::Stop);
    assert_eq!(output.turns_executed, 4);

    // Verify transcript:
    // [0] System, [1] User
    // [2] Assistant (DENY call: sh)
    // [3] ToolResult (Policy violation for sh)
    // [4] Assistant (ALLOW call: write_file broken)
    // [5] ToolResult (Failed verification)
    // [6] Assistant (REPAIR call: write_file fixed)
    // [7] ToolResult (Passed verification)
    // [8] Assistant (final message)
    assert_eq!(output.history.len(), 9);

    // Verify Turn 1 DENY
    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error);
            assert!(content.contains("Policy violation"));
            assert!(content.contains("arbitrary scripting is prohibited"));
        }
        other => panic!("Expected ToolResult at index 3, got {:?}", other),
    }

    // Verify Turn 2 ALLOW + Failed verification
    match &output.history[5] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error, "Expected verification failure");
            assert!(content.contains("AUTOMATED VERIFICATION FAILED - REPAIR REQUIRED"));
        }
        other => panic!("Expected ToolResult at index 5, got {:?}", other),
    }

    // Verify Turn 3 REPAIR + Passed verification
    match &output.history[7] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(!*is_error, "Expected verification success after repair");
            assert!(content.contains("AUTOMATED VERIFICATION PASSED"));
        }
        other => panic!("Expected ToolResult at index 7, got {:?}", other),
    }

    // Verify Turn 4 Final Completion
    match &output.history[8] {
        Message::Assistant { content, .. } => {
            assert!(
                content
                    .as_ref()
                    .unwrap()
                    .contains("Self-repair cycle complete")
            );
        }
        other => panic!("Expected Assistant message at index 8, got {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}
