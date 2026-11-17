use crate::exec::ScopedRunner;
use crate::model::BoxFuture;
use crate::types::ToolDefinition;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Trait defining an executable agent tool.
pub trait Tool: Send + Sync {
    /// Unique name of the tool (e.g. "read_file").
    fn name(&self) -> &str;

    /// Natural-language description of tool functionality.
    fn description(&self) -> &str;

    /// JSON schema describing accepted parameters.
    fn parameters_schema(&self) -> Value;

    /// Full tool definition for registration with model APIs.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }

    /// Executes the tool asynchronously with validated JSON arguments.
    fn run<'a>(&'a self, args: Value) -> BoxFuture<'a, Result<String, String>>;
}

/// Reads text file contents within the workspace jail.
pub struct ReadFileTool {
    workspace_root: PathBuf,
}

impl ReadFileTool {
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        Self {
            workspace_root: workspace_root.as_ref().to_path_buf(),
        }
    }
}

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the full UTF-8 text content of a file within the workspace jail."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path within the workspace"
                }
            },
            "required": ["path"]
        })
    }

    fn run<'a>(&'a self, args: Value) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async move {
            let path_str = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'path'".to_string())?;

            let resolved = self.workspace_root.join(path_str);
            match tokio::fs::read_to_string(&resolved).await {
                Ok(content) => Ok(content),
                Err(err) => Err(format!("Failed to read file '{}': {}", path_str, err)),
            }
        })
    }
}

/// Atomically writes content to a file in the workspace jail and returns a unified diff.
pub struct WriteFileTool {
    workspace_root: PathBuf,
}

impl WriteFileTool {
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        Self {
            workspace_root: workspace_root.as_ref().to_path_buf(),
        }
    }
}

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Atomically writes content to a file within the workspace jail and returns a unified diff."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute target path within the workspace"
                },
                "content": {
                    "type": "string",
                    "description": "Full new UTF-8 content to write"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn run<'a>(&'a self, args: Value) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async move {
            let path_str = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'path'".to_string())?;

            let new_content = args
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'content'".to_string())?;

            let target_path = self.workspace_root.join(path_str);

            if let Some(parent) = target_path.parent()
                && !parent.exists()
            {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("Failed to create parent directories: {}", e))?;
            }

            // Read existing content if present for unified diff
            let old_content = tokio::fs::read_to_string(&target_path).await.ok();

            // Atomic write: write to temp file in same directory, then rename
            let parent_dir = target_path.parent().unwrap_or_else(|| Path::new("."));
            let temp_filename = format!(
                ".tmp.{}_{}",
                target_path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or("file"),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            );
            let temp_path = parent_dir.join(temp_filename);

            tokio::fs::write(&temp_path, new_content.as_bytes())
                .await
                .map_err(|e| format!("Failed to write temporary file: {}", e))?;

            tokio::fs::rename(&temp_path, &target_path)
                .await
                .map_err(|e| format!("Failed to rename temporary file into place: {}", e))?;

            // Generate unified diff
            let diff = generate_unified_diff(path_str, old_content.as_deref(), new_content);
            Ok(diff)
        })
    }
}

/// Lists files and directories within a target directory in the workspace jail.
pub struct ListDirTool {
    workspace_root: PathBuf,
}

impl ListDirTool {
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        Self {
            workspace_root: workspace_root.as_ref().to_path_buf(),
        }
    }
}

impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "Lists entries in a directory within the workspace jail."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to workspace root if empty)"
                }
            }
        })
    }

    fn run<'a>(&'a self, args: Value) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async move {
            let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

            let target = self.workspace_root.join(path_str);
            let mut read_dir = tokio::fs::read_dir(&target)
                .await
                .map_err(|e| format!("Failed to list directory '{}': {}", path_str, e))?;

            let mut entries = Vec::new();
            while let Ok(Some(entry)) = read_dir.next_entry().await {
                let name = entry.file_name().to_string_lossy().into_owned();
                let file_type = match entry.file_type().await {
                    Ok(ft) if ft.is_dir() => "[dir]",
                    Ok(ft) if ft.is_symlink() => "[symlink]",
                    _ => "[file]",
                };
                entries.push(format!("{} {}", file_type, name));
            }
            entries.sort();

            Ok(entries.join("\n"))
        })
    }
}

/// Executes a shell command using `ScopedRunner` within resource and jail limits.
pub struct RunCommandTool {
    runner: Arc<ScopedRunner>,
    cancel_token: CancellationToken,
}

impl RunCommandTool {
    pub fn new(runner: Arc<ScopedRunner>, cancel_token: CancellationToken) -> Self {
        Self {
            runner,
            cancel_token,
        }
    }
}

impl Tool for RunCommandTool {
    fn name(&self) -> &str {
        "run_command"
    }

    fn description(&self) -> &str {
        "Executes a command inside the bounded ScopedRunner environment."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Executable program or command to run (e.g., 'cargo', 'ls')"
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Command-line arguments"
                }
            },
            "required": ["cmd"]
        })
    }

    fn run<'a>(&'a self, args: Value) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async move {
            let cmd = args
                .get("cmd")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'cmd'".to_string())?;

            let empty_vec = Vec::new();
            let cmd_args: Vec<String> = args
                .get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(ToString::to_string))
                        .collect()
                })
                .unwrap_or(empty_vec);

            match self.runner.run(cmd, &cmd_args, &self.cancel_token).await {
                Ok(out) => {
                    let mut report = format!(
                        "Exit Code: {}\nDuration: {:?}\nTruncated: {}\n--- STDOUT ---\n{}\n--- STDERR ---\n{}",
                        out.exit_code
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "Killed by signal".to_string()),
                        out.duration,
                        out.truncated,
                        out.stdout,
                        out.stderr
                    );
                    if out.truncated {
                        report.push_str(
                            "\n[WARNING: Command output exceeded maximum buffer and was truncated]",
                        );
                    }
                    Ok(report)
                }
                Err(err) => Err(format!("Execution failed: {}", err)),
            }
        })
    }
}

/// Generates a unified diff comparing old and new file content.
fn generate_unified_diff(path: &str, old: Option<&str>, new: &str) -> String {
    let mut out = String::new();
    match old {
        None => {
            out.push_str(&format!("--- /dev/null\n+++ b/{}\n", path));
            let lines: Vec<&str> = new.lines().collect();
            out.push_str(&format!("@@ -0,0 +1,{} @@\n", lines.len()));
            for line in lines {
                out.push('+');
                out.push_str(line);
                out.push('\n');
            }
        }
        Some(old_str) => {
            out.push_str(&format!("--- a/{}\n+++ b/{}\n", path, path));
            let old_lines: Vec<&str> = old_str.lines().collect();
            let new_lines: Vec<&str> = new.lines().collect();

            out.push_str(&format!(
                "@@ -1,{} +1,{} @@\n",
                old_lines.len(),
                new_lines.len()
            ));

            // Simple line diff
            let max = old_lines.len().max(new_lines.len());
            for i in 0..max {
                match (old_lines.get(i), new_lines.get(i)) {
                    (Some(o), Some(n)) if o == n => {
                        out.push(' ');
                        out.push_str(o);
                        out.push('\n');
                    }
                    (Some(o), Some(n)) => {
                        out.push('-');
                        out.push_str(o);
                        out.push('\n');
                        out.push('+');
                        out.push_str(n);
                        out.push('\n');
                    }
                    (Some(o), None) => {
                        out.push('-');
                        out.push_str(o);
                        out.push('\n');
                    }
                    (None, Some(n)) => {
                        out.push('+');
                        out.push_str(n);
                        out.push('\n');
                    }
                    (None, None) => {}
                }
            }
        }
    }
    out
}
