use agentic_harness::{
    DefaultPolicy, ExecutionError, ListDirArgs, ListDirTool, Message, MockModel, Policy,
    PolicyDecision, ReadFileArgs, ReadFileTool, RunCommandArgs, RunCommandTool, ScopedRunner,
    StopReason, Tool, ToolCall, VerificationConfig, WriteFileArgs, WriteFileTool, run_agent,
    run_agent_with_verification,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn setup_test_workspace(name: &str) -> PathBuf {
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
                reason.contains("escapes workspace jail"),
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
            assert!(reason.contains("escapes workspace jail"));
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
            assert!(reason.contains("escapes workspace jail"));
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

    // 4. Valid write inside src/ is allowed
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
    let workspace = setup_test_workspace("t3_timeout_pgid");
    let policy = DefaultPolicy::new(&workspace);
    let runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_timeout(Duration::from_millis(500));
    let cancel = CancellationToken::new();

    // Prepare a Cargo crate that spawns a sleep process and writes its PID
    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "pgid_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "grandchild_sleeper"
path = "src/main.rs"
"#,
    )
    .unwrap();

    let pid_file = workspace.join("grandchild.pid");
    fs::write(
        workspace.join("src/main.rs"),
        format!(
            r#"use std::fs;
use std::thread::sleep;
use std::time::Duration;
fn main() {{
    fs::write("{}", std::process::id().to_string()).unwrap();
    sleep(Duration::from_secs(300));
}}
"#,
            pid_file.to_string_lossy()
        ),
    )
    .unwrap();

    // Pre-build binary so the test timeout measures actual execution, not compilation
    let prebuild_auth = policy
        .validate_command(
            "cargo",
            &[
                "build".to_string(),
                "--bin".to_string(),
                "grandchild_sleeper".to_string(),
            ],
        )
        .expect("cargo build must be authorized");
    let prebuild_runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_timeout(Duration::from_secs(30));
    let _ = prebuild_runner
        .execute(&prebuild_auth, &cancel)
        .await
        .unwrap();

    // Now execute through AuthorizedCall with short 500ms timeout
    let auth_call = policy
        .validate_command(
            "cargo",
            &[
                "run".to_string(),
                "--bin".to_string(),
                "grandchild_sleeper".to_string(),
            ],
        )
        .expect("cargo run must be authorized");

    let result = runner.execute(&auth_call, &cancel).await;
    match result {
        Err(ExecutionError::Timeout(_)) => {}
        other => panic!("Expected ExecutionError::Timeout, got: {:?}", other),
    }

    assert!(pid_file.exists(), "Grandchild PID file was not created");
    let pid_str = fs::read_to_string(&pid_file).unwrap();
    let grandchild_pid: i32 = pid_str.trim().parse().unwrap();
    assert!(grandchild_pid > 0);

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Kernel-level existence probe: kill(pid, 0)
    let probe = unsafe { libc::kill(grandchild_pid, 0) };
    let errno = std::io::Error::last_os_error();

    assert_eq!(probe, -1, "Grandchild is still alive! Expected ESRCH.");
    assert_eq!(errno.raw_os_error(), Some(libc::ESRCH));

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_t4_output_truncation_flag_set() {
    let workspace = setup_test_workspace("t4_truncation");
    let policy = DefaultPolicy::new(&workspace);
    let runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_max_output_bytes(1024)
        .with_timeout(Duration::from_secs(30));
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "trunc_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "gen_output"
path = "src/main.rs"
"#,
    )
    .unwrap();

    fs::write(
        workspace.join("src/main.rs"),
        r#"fn main() {
    print!("{}", "A".repeat(4096));
}
"#,
    )
    .unwrap();

    let auth_call = policy
        .validate_command(
            "cargo",
            &[
                "run".to_string(),
                "--bin".to_string(),
                "gen_output".to_string(),
            ],
        )
        .unwrap();

    let output = runner.execute(&auth_call, &cancel).await.unwrap();

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
            assert!(content.contains("escapes workspace jail"));
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_new_a_run_command_interpreter_denied_by_default() {
    let workspace = setup_test_workspace("interp_deny");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();
    let tool = RunCommandTool::new(runner, policy.clone(), cancel);

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
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();
    let tool = RunCommandTool::new(runner, policy.clone(), cancel);

    // cargo is ALLOWED and produces an AuthorizedCall
    let call_cargo = ToolCall {
        id: "call_cargo".to_string(),
        name: "run_command".to_string(),
        arguments: json!({
            "cmd": "cargo",
            "args": ["--version"]
        }),
    };
    match policy.evaluate(&call_cargo, &tool) {
        PolicyDecision::AllowCommand(auth) => {
            assert_eq!(auth.exe(), "cargo");
            assert_eq!(auth.args(), &["--version"]);
        }
        other => panic!("Expected AllowCommand for cargo, got: {:?}", other),
    }

    // cat, ls, git were removed from allowlist -> strictly DENIED
    for blocked in &["cat", "ls", "git"] {
        let call = ToolCall {
            id: format!("call_{blocked}"),
            name: "run_command".to_string(),
            arguments: json!({
                "cmd": blocked,
                "args": []
            }),
        };
        match policy.evaluate(&call, &tool) {
            PolicyDecision::Deny(reason) => {
                assert!(reason.contains("not on the policy allowlist"));
            }
            other => panic!("Expected Deny for '{}', got: {:?}", blocked, other),
        }
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_new_c_cancellation_token_kills_group() {
    let workspace = setup_test_workspace("cancel_group");
    let policy = DefaultPolicy::new(&workspace);
    let runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_timeout(Duration::from_secs(30));
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "cancel_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "cancel_sleeper"
path = "src/main.rs"
"#,
    )
    .unwrap();

    let pid_file = workspace.join("child_sleep.pid");
    fs::write(
        workspace.join("src/main.rs"),
        format!(
            r#"use std::fs;
use std::thread::sleep;
use std::time::Duration;
fn main() {{
    fs::write("{}", std::process::id().to_string()).unwrap();
    sleep(Duration::from_secs(300));
}}
"#,
            pid_file.to_string_lossy()
        ),
    )
    .unwrap();

    // Prebuild so cancellation aborts the running process
    let prebuild_auth = policy
        .validate_command(
            "cargo",
            &[
                "build".to_string(),
                "--bin".to_string(),
                "cancel_sleeper".to_string(),
            ],
        )
        .unwrap();
    let prebuild_runner = ScopedRunner::new(&workspace)
        .with_bwrap(false)
        .with_timeout(Duration::from_secs(30));
    let _ = prebuild_runner
        .execute(&prebuild_auth, &cancel)
        .await
        .unwrap();

    let auth_call = policy
        .validate_command(
            "cargo",
            &[
                "run".to_string(),
                "--bin".to_string(),
                "cancel_sleeper".to_string(),
            ],
        )
        .unwrap();

    let cancel_trigger = cancel.clone();
    let pid_check = pid_file.clone();
    tokio::spawn(async move {
        for _ in 0..50 {
            if pid_check.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        cancel_trigger.cancel();
    });

    let result = runner.execute(&auth_call, &cancel).await;
    match result {
        Err(ExecutionError::Cancelled) => {}
        other => panic!("Expected ExecutionError::Cancelled, got: {:?}", other),
    }

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
    let valid_read = json!({ "path": "src/main.rs" });
    assert!(serde_json::from_value::<ReadFileArgs>(valid_read).is_ok());

    let invalid_read = json!({ "path": "src/main.rs", "extra_field": "hack" });
    assert!(serde_json::from_value::<ReadFileArgs>(invalid_read).is_err());

    let valid_write = json!({ "path": "src/lib.rs", "content": "pub fn foo() {}" });
    assert!(serde_json::from_value::<WriteFileArgs>(valid_write).is_ok());

    let invalid_write = json!({ "path": "src/lib.rs", "content": "code", "mode": "0777" });
    assert!(serde_json::from_value::<WriteFileArgs>(invalid_write).is_err());

    let valid_list = json!({ "path": "src" });
    assert!(serde_json::from_value::<ListDirArgs>(valid_list).is_ok());

    let invalid_list = json!({ "path": "src", "recursive": true });
    assert!(serde_json::from_value::<ListDirArgs>(invalid_list).is_err());

    let valid_run = json!({ "cmd": "cargo", "args": ["test"] });
    assert!(serde_json::from_value::<RunCommandArgs>(valid_run).is_ok());

    let invalid_run = json!({ "cmd": "cargo", "args": ["test"], "env": {"EVIL": "1"} });
    assert!(serde_json::from_value::<RunCommandArgs>(invalid_run).is_err());
}

#[tokio::test]
async fn test_part_a_verification_interpreter_denied_nothing_executed() {
    let workspace = setup_test_workspace("verify_sh_denied");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();

    let marker_file = workspace.join("marker_should_not_exist.txt");
    fs::create_dir_all(workspace.join("src")).unwrap();

    let script_file = workspace.join("src/malicious_verifier.sh");
    fs::write(
        &script_file,
        format!(
            "#!/bin/sh\necho 'ran' > {}\n",
            marker_file.to_string_lossy()
        ),
    )
    .unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner, policy.clone(), cancel.clone())),
    ];

    // Configure verification to run `sh` (which must be denied with ZERO exemptions)
    let v_config = VerificationConfig {
        enabled: true,
        command: "sh".to_string(),
        args: vec![script_file.to_string_lossy().to_string()],
        max_repairs: 2,
    };

    let model = MockModel::with_responses(vec![
        Message::assistant(
            Some("Writing patch...".to_string()),
            vec![ToolCall {
                id: "call_write".to_string(),
                name: "write_file".to_string(),
                arguments: json!({
                    "path": "src/lib.rs",
                    "content": "pub fn hello() {}"
                }),
            }],
        ),
        Message::assistant_text("Done."),
    ]);

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Trigger mutating action with illegal verifier",
        policy.as_ref(),
        Some(v_config),
        4,
        cancel,
    )
    .await
    .unwrap();

    // Verify transcript records the verification command was blocked by policy
    let transcript = format!("{:?}", output.history);
    assert!(
        transcript.contains("AUTOMATED VERIFICATION BLOCKED BY POLICY"),
        "Expected policy denial in verification transcript"
    );
    assert!(
        transcript.contains("Execution of interpreter 'sh' is blocked by policy"),
        "Expected interpreter block reason"
    );

    // Marker file was NEVER created
    assert!(
        !marker_file.exists(),
        "CRITICAL: Verifier command executed despite interpreter policy!"
    );

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_part_b_verifier_integrity_agent_write_to_tests_denied() {
    let workspace = setup_test_workspace("verifier_integrity");
    let policy = DefaultPolicy::new(&workspace);
    let tool = WriteFileTool::new(&workspace);

    // 1. Agent tries to overwrite unit test file in tests/ -> DENIED
    let call_overwrite_test = ToolCall {
        id: "call_hack_test".to_string(),
        name: "write_file".to_string(),
        arguments: json!({
            "path": "tests/reconcile_test.rs",
            "content": "#[test] fn test_always_pass() { assert!(true); }"
        }),
    };
    match policy.evaluate(&call_overwrite_test, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains("restricted to writing only within the 'src/' directory"));
        }
        other => panic!("Expected Deny for write to tests/, got: {:?}", other),
    }

    // 2. Agent tries to overwrite Cargo.toml -> DENIED
    let call_overwrite_cargo = ToolCall {
        id: "call_hack_cargo".to_string(),
        name: "write_file".to_string(),
        arguments: json!({
            "path": "Cargo.toml",
            "content": "[package]\nname = 'hacked'"
        }),
    };
    match policy.evaluate(&call_overwrite_cargo, &tool) {
        PolicyDecision::Deny(reason) => {
            assert!(reason.contains("restricted to writing only within the 'src/' directory"));
        }
        other => panic!("Expected Deny for write to Cargo.toml, got: {:?}", other),
    }

    // 3. Agent writes to src/lib.rs -> ALLOWED
    let call_valid_src = ToolCall {
        id: "call_src_lib".to_string(),
        name: "write_file".to_string(),
        arguments: json!({
            "path": "src/lib.rs",
            "content": "pub fn legit() {}"
        }),
    };
    assert_eq!(
        policy.evaluate(&call_valid_src, &tool),
        PolicyDecision::Allow
    );

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_part_c_secret_planted_outside_workspace_unreadable_in_sandbox() {
    let workspace = setup_test_workspace("secret_isolation");
    let policy = DefaultPolicy::new(&workspace);
    let runner = ScopedRunner::new(&workspace).with_bwrap(true);
    let cancel = CancellationToken::new();

    // Plant secret on host outside workspace (in $HOME)
    let home = std::env::var("HOME").expect("HOME must be set");
    let secret_path = PathBuf::from(home).join("agent_planted_secret_test.txt");
    fs::write(&secret_path, "SUPER_SECRET_DO_NOT_LEAK").unwrap();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "leak_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "read_host_secret"
path = "src/main.rs"
"#,
    )
    .unwrap();

    fs::write(
        workspace.join("src/main.rs"),
        format!(
            r#"use std::fs;
fn main() {{
    match fs::read_to_string("{}") {{
        Ok(secret) => panic!("SECRET LEAKED: {{}}", secret),
        Err(e) => {{
            println!("SAFE: cannot read host secret: {{}}", e);
        }}
    }}
}}
"#,
            secret_path.to_string_lossy()
        ),
    )
    .unwrap();

    let auth_call = policy
        .validate_command(
            "cargo",
            &[
                "run".to_string(),
                "--bin".to_string(),
                "read_host_secret".to_string(),
            ],
        )
        .unwrap();

    let output = runner.execute(&auth_call, &cancel).await.unwrap();

    // Remove host secret
    let _ = fs::remove_file(&secret_path);

    assert_eq!(output.exit_code, Some(0));
    assert!(
        output.stdout.contains("SAFE: cannot read host secret"),
        "Host secret was exposed inside sandbox! Output: {}",
        output.stdout
    );

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_self_repair_real_fixture_crate_trace() {
    let workspace = setup_test_workspace("real_fixture_repair");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
    let cancel = CancellationToken::new();

    // Construct genuine fixture crate with immutable test in tests/
    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::create_dir_all(workspace.join("tests")).unwrap();

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "reconciliation_fixture"
version = "0.1.0"
edition = "2021"

[dependencies]
"#,
    )
    .unwrap();

    // Read-only verifier test suite with held-out cases: zero-variance, negative-variance, large scale
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
    )
    .unwrap();

    // Broken initial implementation: additions instead of subtractions
    fs::write(
        workspace.join("src/lib.rs"),
        r#"pub fn calculate_variance(sanctioned: i64, disbursed: i64) -> i64 {
    sanctioned + disbursed
}
"#,
    )
    .unwrap();

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
    let _ = runner.execute(&prebuild_auth, &cancel).await.unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner, policy.clone(), cancel.clone())),
        Box::new(ListDirTool::new(&workspace)),
        Box::new(ReadFileTool::new(&workspace)),
    ];

    // Scripted multi-turn agent demonstrating genuine repair:
    // Turn 1: DENY — Model tries unconfined interpreter execution `sh -c 'cat /etc/passwd'`
    // Turn 2: ALLOW + FAILED VERIFICATION — Model proposes a flawed implementation in src/lib.rs
    // Turn 3: REPAIR + PASSED VERIFICATION — Model inspects compiler/test error and fixes src/lib.rs
    // Turn 4: Final completion
    let turn1_call = Message::assistant(
        Some("Attempting to run diagnostic script via shell interpreter...".to_string()),
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
        Some("Shell interpreter denied by policy. Creating an initial implementation of calculate_variance in src/lib.rs...".to_string()),
        vec![ToolCall {
            id: "call_write_flawed".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "src/lib.rs",
                "content": "pub fn calculate_variance(sanctioned: i64, disbursed: i64) -> i64 {\n    sanctioned - disbursed - 999\n}\n"
            }),
        }],
    );

    let turn3_call = Message::assistant(
        Some("Observed unit test failure from cargo test: 1500 != 501. Fixing variance calculation logic in src/lib.rs...".to_string()),
        vec![ToolCall {
            id: "call_write_fixed".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "src/lib.rs",
                "content": "pub fn calculate_variance(sanctioned: i64, disbursed: i64) -> i64 {\n    sanctioned - disbursed\n}\n"
            }),
        }],
    );

    let turn4_response = Message::assistant_text(
        "Automated verification passed! Real fixture repair cycle complete.",
    );

    let model = MockModel::with_responses(vec![turn1_call, turn2_call, turn3_call, turn4_response]);

    let v_config = VerificationConfig {
        enabled: true,
        command: "cargo".to_string(),
        args: vec!["test".to_string(), "--offline".to_string()],
        max_repairs: 3,
    };

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Fix failing unit test in reconciliation fixture crate",
        policy.as_ref(),
        Some(v_config),
        8,
        cancel,
    )
    .await
    .unwrap();

    assert_eq!(output.stop_reason, StopReason::Stop);
    assert_eq!(output.turns_executed, 4);
    assert_eq!(output.history.len(), 9);

    // Turn 1: DENY check
    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error);
            assert!(content.contains("Policy violation"));
            assert!(content.contains("arbitrary scripting is prohibited"));
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    // Turn 2: FAILED VERIFICATION check
    match &output.history[5] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error);
            assert!(content.contains("AUTOMATED VERIFICATION FAILED - REPAIR REQUIRED"));
            assert!(content.contains("FAILED"));
        }
        other => panic!("Expected ToolResult at index 5, got: {:?}", other),
    }

    // Turn 3: PASSED VERIFICATION check
    match &output.history[7] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(!*is_error);
            assert!(content.contains("AUTOMATED VERIFICATION PASSED"));
            assert!(content.contains("test result: ok. 4 passed"));
        }
        other => panic!("Expected ToolResult at index 7, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_part_a_fail_closed_allow_on_run_command_rejected() {
    let workspace = setup_test_workspace("fail_closed_allow_cmd");
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();

    // Defective policy that attempts to return Allow instead of AllowCommand for run_command
    struct DefectivePolicy;
    impl Policy for DefectivePolicy {
        #[allow(clippy::if_same_then_else)]
        fn evaluate(&self, call: &ToolCall, _tool: &dyn Tool) -> PolicyDecision {
            if call.name == "run_command" {
                PolicyDecision::Allow // ILLEGAL: must return AllowCommand
            } else {
                PolicyDecision::Allow
            }
        }
    }

    let policy = DefectivePolicy;
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(RunCommandTool::new(
        runner,
        Arc::new(DefaultPolicy::new(&workspace)),
        cancel.clone(),
    ))];

    let model = MockModel::with_responses(vec![
        Message::assistant(
            Some("Trying run_command with defective allow policy...".to_string()),
            vec![ToolCall {
                id: "call_defective".to_string(),
                name: "run_command".to_string(),
                arguments: json!({ "cmd": "cargo", "args": ["--version"] }),
            }],
        ),
        Message::assistant_text("Task finished after rejection."),
    ]);

    let output = run_agent(
        &model,
        &tools,
        "Test fail closed on illegal Allow decision",
        &policy,
        4,
        cancel,
    )
    .await
    .unwrap();

    // Verify the illegal Allow decision was caught and rejected as invariant violation
    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error);
            assert!(
                content.contains("Security invariant violation"),
                "Expected invariant violation, got: {}",
                content
            );
            assert!(content.contains("cannot be authorized with PolicyDecision::Allow"));
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_part_a_repair_exhausted_halts_with_distinct_stop_reason() {
    let workspace = setup_test_workspace("repair_exhausted");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::create_dir_all(workspace.join("tests")).unwrap();

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "exhaust_fixture"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();

    fs::write(
        workspace.join("tests/fail_test.rs"),
        r#"#[test] fn test_always_fail() { panic!("Persistent failure"); }"#,
    )
    .unwrap();

    fs::write(workspace.join("src/lib.rs"), "pub fn dummy() {}").unwrap();

    // Prebuild offline metadata
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
    let _ = runner.execute(&prebuild_auth, &cancel).await.unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner, policy.clone(), cancel.clone())),
    ];

    // Model proposes 3 consecutive broken edits
    let model = MockModel::with_responses(vec![
        Message::assistant(
            None,
            vec![ToolCall {
                id: "call_w1".to_string(),
                name: "write_file".to_string(),
                arguments: json!({ "path": "src/lib.rs", "content": "// edit 1" }),
            }],
        ),
        Message::assistant(
            None,
            vec![ToolCall {
                id: "call_w2".to_string(),
                name: "write_file".to_string(),
                arguments: json!({ "path": "src/lib.rs", "content": "// edit 2" }),
            }],
        ),
        Message::assistant(
            None,
            vec![ToolCall {
                id: "call_w3".to_string(),
                name: "write_file".to_string(),
                arguments: json!({ "path": "src/lib.rs", "content": "// edit 3" }),
            }],
        ),
    ]);

    let v_config = VerificationConfig {
        enabled: true,
        command: "cargo".to_string(),
        args: vec!["test".to_string(), "--offline".to_string()],
        max_repairs: 2, // Limit to 2 repairs
    };

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Trigger repair exhaustion",
        policy.as_ref(),
        Some(v_config),
        10, // max_turns = 10 (proves it stops at repair budget, not turn limit)
        cancel,
    )
    .await
    .unwrap();

    // Must halt with distinct StopReason::RepairExhausted, NOT MaxTurns or Stop
    assert_eq!(output.stop_reason, StopReason::RepairExhausted);
    assert_eq!(output.turns_executed, 2);

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_part_a_require_approval_blocks_execution() {
    let workspace = setup_test_workspace("require_approval");
    let runner = Arc::new(ScopedRunner::new(&workspace));
    let cancel = CancellationToken::new();

    struct ApprovalPolicy;
    impl Policy for ApprovalPolicy {
        fn evaluate(&self, _call: &ToolCall, _tool: &dyn Tool) -> PolicyDecision {
            PolicyDecision::RequireApproval(
                "Modification requires administrative sign-off".to_string(),
            )
        }
    }

    let policy = ApprovalPolicy;
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(
            runner,
            Arc::new(DefaultPolicy::new(&workspace)),
            cancel.clone(),
        )),
    ];

    let target_file = workspace.join("src/should_not_be_written.txt");
    fs::create_dir_all(workspace.join("src")).unwrap();

    let model = MockModel::with_responses(vec![
        Message::assistant(
            None,
            vec![ToolCall {
                id: "call_appr".to_string(),
                name: "write_file".to_string(),
                arguments: json!({ "path": "src/should_not_be_written.txt", "content": "secret" }),
            }],
        ),
        Message::assistant_text("Acknowledged approval requirement."),
    ]);

    let output = run_agent(&model, &tools, "Test approval gating", &policy, 4, cancel)
        .await
        .unwrap();

    // Assert tool was blocked
    assert!(
        !target_file.exists(),
        "File was written despite RequireApproval!"
    );

    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error);
            assert!(content.contains("Policy requires human approval"));
            assert!(content.contains("administrative sign-off"));
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_part_b_formatting_repair_retry_bounded_cap_one() {
    let workspace = setup_test_workspace("format_retry_cap");
    let policy = DefaultPolicy::new(&workspace);
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(WriteFileTool::new(&workspace))];
    let cancel = CancellationToken::new();

    // Turn 1: Model emits malformed args (missing required 'content' field)
    let turn1_call = Message::assistant(
        Some("Proposing malformed write_file call...".to_string()),
        vec![ToolCall {
            id: "call_bad_fmt1".to_string(),
            name: "write_file".to_string(),
            arguments: json!({ "path": "src/lib.rs" }), // missing "content"
        }],
    );

    // Turn 2: Model receives corrective schema observation and fixes arguments
    let turn2_call = Message::assistant(
        Some("Correcting arguments based on schema feedback...".to_string()),
        vec![ToolCall {
            id: "call_good_fmt".to_string(),
            name: "write_file".to_string(),
            arguments: json!({ "path": "src/lib.rs", "content": "pub fn fixed() {}" }),
        }],
    );

    let turn3_resp = Message::assistant_text("Formatting repaired successfully.");

    let model = MockModel::with_responses(vec![turn1_call, turn2_call, turn3_resp]);

    let output = run_agent(
        &model,
        &tools,
        "Test bounded formatting retry",
        &policy,
        6,
        cancel,
    )
    .await
    .unwrap();

    assert_eq!(output.turns_executed, 3);
    assert_eq!(output.stop_reason, StopReason::Stop);

    // Check Turn 1 response received corrective schema observation
    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error);
            assert!(
                content.contains("Argument Formatting Error"),
                "Expected formatting error observation, got: {}",
                content
            );
            assert!(content.contains("Formatting Retry 1/1"));
            assert!(content.contains("parameter schema:"));
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    // Check Turn 2 write succeeded
    match &output.history[5] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(!*is_error);
            assert!(content.contains("+pub fn fixed() {}"));
        }
        other => panic!("Expected ToolResult at index 5, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_e1_sandbox_write_to_tests_dir_fails_with_erofs() {
    let workspace = setup_test_workspace("e1_erofs");
    let policy = DefaultPolicy::new(&workspace);
    let runner = ScopedRunner::new(&workspace).with_bwrap(true);
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::create_dir_all(workspace.join("tests")).unwrap();

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "erofs_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "tamper"
path = "src/main.rs"
"#,
    )
    .unwrap();

    fs::write(
        workspace.join("src/main.rs"),
        r#"use std::fs::OpenOptions;
use std::io::ErrorKind;

fn main() {
    let res = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("tests/tamper.txt");

    match res {
        Ok(_) => panic!("Tampering succeeded! tests/ was NOT mounted read-only!"),
        Err(e) => {
            let is_erofs = e.kind() == ErrorKind::ReadOnlyFilesystem
                || e.raw_os_error() == Some(30) // EROFS on Linux
                || e.raw_os_error() == Some(13) // EACCES on Linux
                || e.to_string().contains("Read-only file system");
            assert!(is_erofs, "Expected EROFS or ReadOnlyFilesystem, got: {:?}", e);
            println!("EROFS_VERIFIED: {}", e);
        }
    }
}
"#,
    )
    .unwrap();

    // Prebuild so bwrap runs the binary directly
    let prebuild_auth = policy
        .validate_command(
            "cargo",
            &[
                "build".to_string(),
                "--bin".to_string(),
                "tamper".to_string(),
            ],
        )
        .unwrap();
    let _ = runner.execute(&prebuild_auth, &cancel).await.unwrap();

    let auth_call = policy
        .validate_command(
            "cargo",
            &["run".to_string(), "--bin".to_string(), "tamper".to_string()],
        )
        .unwrap();
    let output = runner.execute(&auth_call, &cancel).await.unwrap();

    assert_eq!(output.exit_code, Some(0));
    assert!(
        output.stdout.contains("EROFS_VERIFIED"),
        "Output: {}",
        output.stdout
    );
    assert!(
        !workspace.join("tests/tamper.txt").exists(),
        "tamper.txt exists on disk!"
    );

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_e2_process_exit_zero_in_function_under_test_fails_verifier() {
    let workspace = setup_test_workspace("e2_exit_zero");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::create_dir_all(workspace.join("tests")).unwrap();

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "exit_zero_fixture"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();

    fs::write(
        workspace.join("tests/reconcile_test.rs"),
        r#"use exit_zero_fixture::calculate_variance;

#[test]
fn test_variance_under_disbursement() {
    assert_eq!(calculate_variance(5000, 3500), 1500);
}
"#,
    )
    .unwrap();

    // Broken initial lib
    fs::write(
        workspace.join("src/lib.rs"),
        r#"pub fn calculate_variance(sanctioned: i64, disbursed: i64) -> i64 {
    sanctioned + disbursed
}
"#,
    )
    .unwrap();

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
    let _ = runner.execute(&prebuild_auth, &cancel).await.unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner, policy.clone(), cancel.clone())),
    ];

    // Model proposes an exploit: calling std::process::exit(0) inside the function under test!
    let turn1_call = Message::assistant(
        Some("Attempting to bypass verifier by calling std::process::exit(0)...".to_string()),
        vec![ToolCall {
            id: "call_exploit_exit0".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "src/lib.rs",
                "content": "pub fn calculate_variance(_sanctioned: i64, _disbursed: i64) -> i64 {\n    std::process::exit(0);\n}\n"
            }),
        }],
    );

    let model = MockModel::with_responses(vec![turn1_call]);

    let v_config = VerificationConfig {
        enabled: true,
        command: "cargo".to_string(),
        args: vec!["test".to_string(), "--offline".to_string()],
        max_repairs: 1, // Single repair attempt allowed
    };

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Fix failing unit test",
        policy.as_ref(),
        Some(v_config),
        4,
        cancel,
    )
    .await
    .unwrap();

    // Verifier MUST reject the bare exit 0 because libtest never printed the summary line!
    // And since max_repairs: 1 is reached, execution terminates with StopReason::RepairExhausted.
    assert_eq!(output.stop_reason, StopReason::RepairExhausted);

    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error, "Verifier allowed exit(0) bypass!");
            assert!(
                content.contains("AUTOMATED VERIFICATION FAILED - REPAIR REQUIRED"),
                "Expected verification failure in tool output: {}",
                content
            );
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_f1_workspace_cargo_toml_ro_src_rw_confinement() {
    let workspace = setup_test_workspace("f1_ro_cargo");
    let policy = DefaultPolicy::new(&workspace);
    let runner = ScopedRunner::new(&workspace).with_bwrap(true);
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();

    let cargo_toml_content = r#"[package]
name = "confinement_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "tamper_cargo"
path = "src/main.rs"
"#;
    fs::write(workspace.join("Cargo.toml"), cargo_toml_content).unwrap();

    fs::write(
        workspace.join("src/main.rs"),
        r#"use std::fs::{self, OpenOptions};
use std::io::ErrorKind;

fn main() {
    // 1. Attempt to overwrite Cargo.toml -> MUST FAIL WITH EROFS
    let res = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open("Cargo.toml");

    match res {
        Ok(_) => panic!("Tampering Cargo.toml succeeded! Workspace root was NOT mounted read-only!"),
        Err(e) => {
            let is_erofs = e.kind() == ErrorKind::ReadOnlyFilesystem
                || e.raw_os_error() == Some(30) // EROFS on Linux
                || e.raw_os_error() == Some(13) // EACCES on Linux
                || e.to_string().contains("Read-only file system");
            assert!(is_erofs, "Expected EROFS for Cargo.toml write, got: {:?}", e);
            println!("CARGO_TOML_EROFS_VERIFIED: {}", e);
        }
    }

    // 2. Control: writing inside src/ MUST SUCCEED (read-write overlay)
    fs::write("src/control_write.txt", "CONTROL_OK").expect("Write inside src/ failed!");
    println!("SRC_CONTROL_WRITE_SUCCEEDED");
}
"#,
    )
    .unwrap();

    // Prebuild so bwrap runs the binary directly
    let prebuild_auth = policy
        .validate_command(
            "cargo",
            &[
                "build".to_string(),
                "--bin".to_string(),
                "tamper_cargo".to_string(),
            ],
        )
        .unwrap();
    let _ = runner.execute(&prebuild_auth, &cancel).await.unwrap();

    let auth_call = policy
        .validate_command(
            "cargo",
            &[
                "run".to_string(),
                "--bin".to_string(),
                "tamper_cargo".to_string(),
            ],
        )
        .unwrap();
    let output = runner.execute(&auth_call, &cancel).await.unwrap();

    assert_eq!(output.exit_code, Some(0));
    assert!(
        output.stdout.contains("CARGO_TOML_EROFS_VERIFIED"),
        "Output missing EROFS confirmation: {}",
        output.stdout
    );
    assert!(
        output.stdout.contains("SRC_CONTROL_WRITE_SUCCEEDED"),
        "Output missing control confirmation: {}",
        output.stdout
    );

    // Verify Cargo.toml remained untouched on host
    let current_toml = fs::read_to_string(workspace.join("Cargo.toml")).unwrap();
    assert_eq!(current_toml, cargo_toml_content);

    // Verify control write succeeded in src/
    let control_content = fs::read_to_string(workspace.join("src/control_write.txt")).unwrap();
    assert_eq!(control_content, "CONTROL_OK");

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_f4_forged_summary_line_fails_verification() {
    let workspace = setup_test_workspace("f4_forge");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::create_dir_all(workspace.join("tests")).unwrap();

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "forge_fixture"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();

    fs::write(
        workspace.join("tests/reconcile_test.rs"),
        r#"use forge_fixture::calculate_variance;

#[test]
fn test_variance_under_disbursement() {
    assert_eq!(calculate_variance(5000, 3500), 1500);
}
"#,
    )
    .unwrap();

    // Broken initial lib
    fs::write(
        workspace.join("src/lib.rs"),
        r#"pub fn calculate_variance(sanctioned: i64, disbursed: i64) -> i64 {
    sanctioned + disbursed
}
"#,
    )
    .unwrap();

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
    let _ = runner.execute(&prebuild_auth, &cancel).await.unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner, policy.clone(), cancel.clone())),
    ];

    // Model proposes an exploit: printing a forged libtest summary line then calling exit(0)
    let turn1_call = Message::assistant(
        Some("Attempting to forge libtest summary line...".to_string()),
        vec![ToolCall {
            id: "call_exploit_forge".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "src/lib.rs",
                "content": "pub fn calculate_variance(_sanctioned: i64, _disbursed: i64) -> i64 {\n    println!(\"test result: ok. 1 passed; 0 failed; finished in 0.00s\");\n    std::process::exit(0);\n}\n"
            }),
        }],
    );

    let model = MockModel::with_responses(vec![turn1_call]);

    let v_config = VerificationConfig {
        enabled: true,
        command: "cargo".to_string(),
        args: vec!["test".to_string(), "--offline".to_string()],
        max_repairs: 1,
    };

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Fix failing unit test",
        policy.as_ref(),
        Some(v_config),
        4,
        cancel,
    )
    .await
    .unwrap();

    // The forged summary line MUST be rejected by the verifier because:
    // 1. libtest captures test stdout in thread-local storage, so println! is not flushed on exit(0).
    // 2. Even if flushed, per-test-name matching requires `test <name> ... ok` which was missing.
    assert_eq!(output.stop_reason, StopReason::RepairExhausted);

    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error, "Verifier allowed forged summary bypass!");
            assert!(
                content.contains("AUTOMATED VERIFICATION FAILED - REPAIR REQUIRED"),
                "Expected verification failure in tool output: {}",
                content
            );
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_g1_target_dir_tmpfs_confinement_prevents_cache_poisoning() {
    let workspace = setup_test_workspace("g1_target_tmpfs");
    let policy = DefaultPolicy::new(&workspace);
    let runner = ScopedRunner::new(&workspace).with_bwrap(true);
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::create_dir_all(workspace.join("target")).unwrap();

    let cargo_toml = r#"[package]
name = "poison_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "poisoner"
path = "src/main.rs"
"#;
    fs::write(workspace.join("Cargo.toml"), cargo_toml).unwrap();

    fs::write(
        workspace.join("src/main.rs"),
        r#"use std::fs::OpenOptions;
use std::io::ErrorKind;

fn main() {
    // Attempt to write/plant a file inside workspace target/ directory
    let res = OpenOptions::new()
        .write(true)
        .create(true)
        .open("target/poison_artifact");

    match res {
        Ok(_) => panic!("Target dir was writable! Exploitable to cache poisoning!"),
        Err(e) => {
            let is_erofs = e.kind() == ErrorKind::ReadOnlyFilesystem
                || e.raw_os_error() == Some(30)
                || e.to_string().contains("Read-only file system");
            assert!(is_erofs, "Expected EROFS for target write, got: {:?}", e);
            println!("TARGET_EROFS_VERIFIED: {}", e);
        }
    }
}
"#,
    )
    .unwrap();

    let prebuild_auth = policy
        .validate_command(
            "cargo",
            &[
                "build".to_string(),
                "--bin".to_string(),
                "poisoner".to_string(),
            ],
        )
        .unwrap();
    let _ = runner.execute(&prebuild_auth, &cancel).await.unwrap();

    let auth_call = policy
        .validate_command(
            "cargo",
            &[
                "run".to_string(),
                "--bin".to_string(),
                "poisoner".to_string(),
            ],
        )
        .unwrap();
    let output = runner.execute(&auth_call, &cancel).await.unwrap();

    assert_eq!(output.exit_code, Some(0));
    assert!(output.stdout.contains("TARGET_EROFS_VERIFIED"));
    assert!(!workspace.join("target/poison_artifact").exists());

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_g2_verifier_fails_on_deleted_unit_test() {
    let workspace = setup_test_workspace("g2_deleted_test");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "deleted_test_pkg"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();

    // Pristine baseline has two tests: test_alpha and test_beta
    fs::write(
        workspace.join("src/lib.rs"),
        r#"pub fn add(a: i64, b: i64) -> i64 { a + b }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_alpha() {
        assert_eq!(add(1, 1), 2);
    }
    #[test]
    fn test_beta() {
        assert_eq!(add(2, 2), 4);
    }
}
"#,
    )
    .unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner, policy.clone(), cancel.clone())),
    ];

    // Model proposes an edit that DELETES test_beta
    let turn1_call = Message::assistant(
        Some("Deleting test_beta to reduce test surface...".to_string()),
        vec![ToolCall {
            id: "call_delete_test".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "src/lib.rs",
                "content": "pub fn add(a: i64, b: i64) -> i64 { a + b }\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn test_alpha() {\n        assert_eq!(add(1, 1), 2);\n    }\n}\n"
            }),
        }],
    );

    let model = MockModel::with_responses(vec![turn1_call]);

    let v_config = VerificationConfig {
        enabled: true,
        command: "cargo".to_string(),
        args: vec!["test".to_string(), "--offline".to_string()],
        max_repairs: 1,
    };

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Maintain unit tests",
        policy.as_ref(),
        Some(v_config),
        4,
        cancel,
    )
    .await
    .unwrap();

    // Verifier MUST reject the run because baseline test_beta was deleted
    assert_eq!(output.stop_reason, StopReason::RepairExhausted);

    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error, "Verifier allowed deleted test!");
            assert!(
                content.contains("AUTOMATED VERIFICATION FAILED - REPAIR REQUIRED"),
                "Expected verification failure in tool output: {}",
                content
            );
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_g2_verifier_fails_on_ignored_test() {
    let workspace = setup_test_workspace("g2_ignored_test");
    let policy = Arc::new(DefaultPolicy::new(&workspace));
    let runner = Arc::new(ScopedRunner::new(&workspace).with_bwrap(true));
    let cancel = CancellationToken::new();

    fs::create_dir_all(workspace.join("src")).unwrap();

    fs::write(
        workspace.join("Cargo.toml"),
        r#"[package]
name = "ignored_test_pkg"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();

    // Pristine baseline has two tests: test_alpha and test_beta
    fs::write(
        workspace.join("src/lib.rs"),
        r#"pub fn add(a: i64, b: i64) -> i64 { a + b }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_alpha() {
        assert_eq!(add(1, 1), 2);
    }
    #[test]
    fn test_beta() {
        assert_eq!(add(2, 2), 4);
    }
}
"#,
    )
    .unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(WriteFileTool::new(&workspace)),
        Box::new(RunCommandTool::new(runner, policy.clone(), cancel.clone())),
    ];

    // Model proposes an edit that marks test_beta with #[ignore]
    let turn1_call = Message::assistant(
        Some("Marking test_beta as #[ignore]...".to_string()),
        vec![ToolCall {
            id: "call_ignore_test".to_string(),
            name: "write_file".to_string(),
            arguments: json!({
                "path": "src/lib.rs",
                "content": "pub fn add(a: i64, b: i64) -> i64 { a + b }\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn test_alpha() {\n        assert_eq!(add(1, 1), 2);\n    }\n    #[test]\n    #[ignore]\n    fn test_beta() {\n        assert_eq!(add(2, 2), 4);\n    }\n}\n"
            }),
        }],
    );

    let model = MockModel::with_responses(vec![turn1_call]);

    let v_config = VerificationConfig {
        enabled: true,
        command: "cargo".to_string(),
        args: vec!["test".to_string(), "--offline".to_string()],
        max_repairs: 1,
    };

    let output = run_agent_with_verification(
        &model,
        &tools,
        "Maintain unit tests",
        policy.as_ref(),
        Some(v_config),
        4,
        cancel,
    )
    .await
    .unwrap();

    // Verifier MUST reject because test_beta was marked ignored (requires 0 ignored)
    assert_eq!(output.stop_reason, StopReason::RepairExhausted);

    match &output.history[3] {
        Message::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error, "Verifier allowed ignored test!");
            assert!(
                content.contains("AUTOMATED VERIFICATION FAILED - REPAIR REQUIRED"),
                "Expected verification failure in tool output: {}",
                content
            );
        }
        other => panic!("Expected ToolResult at index 3, got: {:?}", other),
    }

    let _ = fs::remove_dir_all(&workspace);
}
