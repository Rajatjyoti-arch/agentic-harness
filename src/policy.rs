use crate::tool::Tool;
use crate::types::ToolCall;
use std::path::{Component, Path, PathBuf};

/// Authorized command execution token with strictly private fields.
/// The ONLY way to obtain an `AuthorizedCall` is via successful evaluation by a `Policy`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedCall {
    exe: String,
    args: Vec<String>,
}

impl AuthorizedCall {
    /// Private constructor, accessible ONLY within `src/policy.rs` during policy validation.
    fn new(exe: String, args: Vec<String>) -> Self {
        Self { exe, args }
    }

    /// Executable binary to invoke.
    pub fn exe(&self) -> &str {
        &self.exe
    }

    /// Arguments to supply to the executable.
    pub fn args(&self) -> &[String] {
        &self.args
    }
}

/// The verdict produced by a security policy gate before tool execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// Tool invocation is allowed to proceed (for non-command tools: read_file, write_file, list_dir, domain tools).
    Allow,
    /// Command execution is authorized; carries the unforgeable `AuthorizedCall` token.
    AllowCommand(AuthorizedCall),
    /// Tool invocation is strictly blocked with a specific violation rationale.
    Deny(String),
    /// Tool invocation requires human confirmation before proceeding.
    RequireApproval(String),
}

/// Abstract security policy evaluated before any tool executes.
pub trait Policy: Send + Sync {
    /// Evaluates the tool call against configured security boundaries.
    fn evaluate(&self, call: &ToolCall, tool: &dyn Tool) -> PolicyDecision;
}

/// Permitted commands in the fail-closed allowlist.
/// Restricted strictly to `cargo` and `rustc`. `cat`, `ls`, and `git` are removed.
pub const ALLOWED_EXECUTABLES: &[&str] = &["cargo", "rustc"];

/// Disallowed interpreters and script runners that provide arbitrary command execution surfaces.
pub const BLOCKED_INTERPRETERS: &[&str] = &[
    "sh", "bash", "zsh", "csh", "tcsh", "dash", "ksh", "python", "python3", "python2", "node",
    "perl", "ruby", "php", "env", "xargs", "find",
];

/// Checks whether `bwrap` (bubblewrap sandbox utility) is present on system PATH.
pub fn is_bwrap_available() -> bool {
    if Path::new("/usr/bin/bwrap").exists() {
        return true;
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(':') {
            if Path::new(dir).join("bwrap").exists() {
                return true;
            }
        }
    }
    false
}

/// Production fail-closed security policy enforcing workspace filesystem jailing,
/// strict command allowlists, write confinement to `src/`, and containerized bubblewrap confinement.
#[derive(Debug, Clone)]
pub struct DefaultPolicy {
    workspace_root: PathBuf,
    canonical_workspace_root: PathBuf,
    enforce_bwrap: bool,
}

impl DefaultPolicy {
    /// Creates a new `DefaultPolicy` anchored at `workspace_root`.
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        let root = workspace_root.as_ref().to_path_buf();
        let canonical = root
            .canonicalize()
            .unwrap_or_else(|_| normalize_path(&root));
        Self {
            workspace_root: root,
            canonical_workspace_root: canonical,
            enforce_bwrap: true,
        }
    }

    /// Configures whether bubblewrap isolation must be enforced for sensitive commands.
    pub fn with_enforce_bwrap(mut self, enforce: bool) -> Self {
        self.enforce_bwrap = enforce;
        self
    }

    /// Verifies that a target path remains strictly inside the workspace jail,
    /// does not access protected secrets/metadata, and escapes no symlinks.
    pub fn validate_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let trimmed = raw_path.trim();
        if trimmed.is_empty() {
            return Err("Path cannot be empty".to_string());
        }

        // 1. Check for home directory expansion and raw protected directory names
        if trimmed.starts_with('~') || trimmed.contains("/.ssh") || trimmed.contains(".ssh/") {
            return Err(
                "Access to user home or .ssh credentials is strictly forbidden".to_string(),
            );
        }

        let input_path = Path::new(trimmed);

        // 2. Resolve relative paths against workspace_root
        let resolved = if input_path.is_absolute() {
            input_path.to_path_buf()
        } else {
            self.workspace_root.join(input_path)
        };

        // 3. Reject access to protected internal directories
        for component in resolved.components() {
            let comp_str = component.as_os_str().to_string_lossy();
            if comp_str == ".git" {
                return Err("Access to .git metadata directory is denied by policy".to_string());
            }
            if comp_str == ".env" || comp_str.starts_with(".env.") {
                return Err("Access to .env secret files is denied by policy".to_string());
            }
        }

        // 4. Lexical containment check
        let normalized = normalize_path(&resolved);
        if !normalized.starts_with(&self.canonical_workspace_root)
            && !normalized.starts_with(&self.workspace_root)
        {
            return Err(format!(
                "Path traversal denied: path '{}' escapes workspace jail '{}'",
                raw_path,
                self.workspace_root.display()
            ));
        }

        // 5. Canonicalization and symlink escape verification
        if normalized.exists() {
            match normalized.canonicalize() {
                Ok(canonical) => {
                    if !canonical.starts_with(&self.canonical_workspace_root) {
                        return Err(format!(
                            "Symlink escape denied: target '{}' resolves outside workspace jail",
                            raw_path
                        ));
                    }
                }
                Err(err) => {
                    return Err(format!(
                        "Failed to canonicalize path '{}': {}",
                        raw_path, err
                    ));
                }
            }
        } else {
            // Path does not exist yet (e.g. for write_file): verify closest existing parent
            let mut curr = normalized.as_path();
            while let Some(parent) = curr.parent() {
                if parent.exists() {
                    if let Ok(canonical_parent) = parent.canonicalize()
                        && !canonical_parent.starts_with(&self.canonical_workspace_root)
                    {
                        return Err(format!(
                            "Parent symlink escape denied for target '{}'",
                            raw_path
                        ));
                    }
                    break;
                }
                curr = parent;
            }
        }

        Ok(normalized)
    }

    /// Evaluates executable and arguments against the fail-closed allowlist policy.
    /// Returns an unforgeable `AuthorizedCall` token on success.
    pub fn validate_command(&self, cmd: &str, args: &[String]) -> Result<AuthorizedCall, String> {
        let trimmed_cmd = cmd.trim();
        if trimmed_cmd.is_empty() {
            return Err("Command cannot be empty".to_string());
        }

        // Extract base binary name (e.g., "/usr/bin/cargo" -> "cargo")
        let base_name = Path::new(trimmed_cmd)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(trimmed_cmd);

        // 1. Interpreters are NEVER auto-allowed: deny arbitrary scripting surface
        if BLOCKED_INTERPRETERS.contains(&base_name) {
            return Err(format!(
                "Execution of interpreter '{}' is blocked by policy: arbitrary scripting is prohibited",
                base_name
            ));
        }

        // 2. Strict executable allowlist check (cargo and rustc only)
        if !ALLOWED_EXECUTABLES.contains(&base_name) {
            return Err(format!(
                "Executable '{}' is not on the policy allowlist: only {:?} are permitted. Use typed tools (read_file, write_file, list_dir) for file access.",
                base_name, ALLOWED_EXECUTABLES
            ));
        }

        // 3. Confinement enforcement: if cargo or rustc are invoked without bwrap, fail closed
        if self.enforce_bwrap && !is_bwrap_available() {
            return Err(
                "Execution denied: bubblewrap (bwrap) container isolation is not available on PATH"
                    .to_string(),
            );
        }

        Ok(AuthorizedCall::new(trimmed_cmd.to_string(), args.to_vec()))
    }
}

impl Policy for DefaultPolicy {
    fn evaluate(&self, call: &ToolCall, _tool: &dyn Tool) -> PolicyDecision {
        match call.name.as_str() {
            "read_file" | "list_dir" => {
                if let Some(path_val) = call.arguments.get("path") {
                    if let Some(path_str) = path_val.as_str() {
                        match self.validate_path(path_str) {
                            Ok(_) => PolicyDecision::Allow,
                            Err(reason) => PolicyDecision::Deny(reason),
                        }
                    } else {
                        PolicyDecision::Deny("Argument 'path' must be a valid string".to_string())
                    }
                } else if call.name == "list_dir" {
                    // list_dir defaults to "."
                    PolicyDecision::Allow
                } else {
                    PolicyDecision::Deny("Missing required argument 'path'".to_string())
                }
            }
            "write_file" => {
                if let Some(path_val) = call.arguments.get("path") {
                    if let Some(path_str) = path_val.as_str() {
                        match self.validate_path(path_str) {
                            Ok(normalized) => {
                                // Enforce verifier integrity: agent may write ONLY inside 'src/'
                                let allowed_src = self.workspace_root.join("src");
                                let allowed_src_canonical =
                                    self.canonical_workspace_root.join("src");
                                if !normalized.starts_with(&allowed_src)
                                    && !normalized.starts_with(&allowed_src_canonical)
                                {
                                    PolicyDecision::Deny(
                                        "Write access denied: agent is restricted to writing only within the 'src/' directory to preserve verifier and workspace integrity".to_string(),
                                    )
                                } else {
                                    PolicyDecision::Allow
                                }
                            }
                            Err(reason) => PolicyDecision::Deny(reason),
                        }
                    } else {
                        PolicyDecision::Deny("Argument 'path' must be a valid string".to_string())
                    }
                } else {
                    PolicyDecision::Deny("Missing required argument 'path'".to_string())
                }
            }
            "run_command" => {
                let cmd = call
                    .arguments
                    .get("cmd")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let empty_vec = Vec::new();
                let args_vec: Vec<String> = call
                    .arguments
                    .get("args")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(ToString::to_string))
                            .collect()
                    })
                    .unwrap_or(empty_vec);

                match self.validate_command(cmd, &args_vec) {
                    Ok(auth_call) => PolicyDecision::AllowCommand(auth_call),
                    Err(reason) => PolicyDecision::Deny(reason),
                }
            }
            _ => PolicyDecision::Allow,
        }
    }
}

/// Normalizes lexical components without requiring the path to exist on disk.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                components.pop();
            }
            c => components.push(c),
        }
    }
    components.into_iter().collect()
}
