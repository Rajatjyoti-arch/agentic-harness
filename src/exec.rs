use crate::policy::is_bwrap_available;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// Errors arising during process lifecycle management.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Command execution timed out after {0:?}")]
    Timeout(Duration),

    #[error("Command execution was cancelled")]
    Cancelled,

    #[error("Process I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Output captured from a bounded process execution.
#[derive(Debug, Clone)]
pub struct ExecutionOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub truncated: bool,
    pub duration: Duration,
}

/// Bounded subprocess runner enforcing Linux resource limits, environment sanitization,
/// output caps, optional containerized bubblewrap confinement, and process-group termination.
pub struct ScopedRunner {
    workspace_root: PathBuf,
    default_timeout: Duration,
    max_output_bytes: usize,
    use_bwrap: bool,
}

impl ScopedRunner {
    /// Creates a new `ScopedRunner` locked to `workspace_root`.
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        Self {
            workspace_root: workspace_root.as_ref().to_path_buf(),
            default_timeout: Duration::from_secs(30),
            max_output_bytes: 512 * 1024, // 512 KB
            use_bwrap: is_bwrap_available(),
        }
    }

    /// Configures the wall-clock execution timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Configures the maximum stdout/stderr capture buffer size in bytes.
    pub fn with_max_output_bytes(mut self, max_bytes: usize) -> Self {
        self.max_output_bytes = max_bytes;
        self
    }

    /// Configures whether bubblewrap isolation is enabled.
    pub fn with_bwrap(mut self, enabled: bool) -> Self {
        self.use_bwrap = enabled;
        self
    }

    /// Executes a command within the sandboxed environment with cooperative cancellation.
    ///
    /// Never invokes a shell parser: invokes `Command::new(program).args(args)` directly.
    pub async fn run(
        &self,
        program: &str,
        args: &[String],
        cancel_token: &CancellationToken,
    ) -> Result<ExecutionOutput, ExecutionError> {
        // If bubblewrap is available and enabled, construct the sandboxed container command
        let (exec_program, exec_args) = if self.use_bwrap && is_bwrap_available() {
            let mut bwrap_args = vec![
                "--ro-bind".to_string(),
                "/".to_string(),
                "/".to_string(),
                "--tmpfs".to_string(),
                "/tmp".to_string(),
                "--proc".to_string(),
                "/proc".to_string(),
                "--dev".to_string(),
                "/dev".to_string(),
                "--bind".to_string(),
                self.workspace_root.to_string_lossy().into_owned(),
                self.workspace_root.to_string_lossy().into_owned(),
                "--unshare-all".to_string(),
                "--die-with-parent".to_string(),
                "--chdir".to_string(),
                self.workspace_root.to_string_lossy().into_owned(),
            ];

            // Mask sensitive host directories like ~/.ssh with empty tmpfs mounts
            if let Ok(home) = std::env::var("HOME") {
                let ssh_dir = Path::new(&home).join(".ssh");
                if ssh_dir.exists() {
                    bwrap_args.push("--tmpfs".to_string());
                    bwrap_args.push(ssh_dir.to_string_lossy().into_owned());
                }
            }

            bwrap_args.push("--".to_string());
            bwrap_args.push(program.to_string());
            bwrap_args.extend_from_slice(args);
            ("bwrap".to_string(), bwrap_args)
        } else {
            (program.to_string(), args.to_vec())
        };

        let mut cmd = Command::new(&exec_program);
        cmd.args(&exec_args);

        // 1. Working directory locked to workspace root
        cmd.current_dir(&self.workspace_root);

        // 2. Clear all environment variables and whitelist safe primitives only
        cmd.env_clear();
        if let Ok(path) = std::env::var("PATH") {
            cmd.env("PATH", path);
        }
        cmd.env("HOME", "/tmp");
        cmd.env("LANG", "C.UTF-8");
        cmd.env("TERM", "xterm-256color");

        // 3. Process piping & kill-on-drop guard
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.kill_on_drop(true);

        // 4. Unix process isolation via pre_exec (async-signal-safe calls only)
        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                // Form a dedicated process group where pgid == child pid
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }

                // RLIMIT_AS (Virtual Address Space): strict 2 GiB
                let as_limit = libc::rlimit {
                    rlim_cur: 2 * 1024 * 1024 * 1024,
                    rlim_max: 2 * 1024 * 1024 * 1024,
                };
                libc::setrlimit(libc::RLIMIT_AS, &as_limit);

                // RLIMIT_CPU: soft = 60s, hard = 61s (kernel fires SIGKILL on hard limit)
                let cpu_limit = libc::rlimit {
                    rlim_cur: 60,
                    rlim_max: 61,
                };
                libc::setrlimit(libc::RLIMIT_CPU, &cpu_limit);

                // RLIMIT_NPROC: Never raise inherited limit. In Linux, RLIMIT_NPROC is
                // an aggregate count per real UID across all user sessions and desktop apps.
                // We read inherited limit and never increase it.
                let mut inherited_nproc = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_NPROC, &mut inherited_nproc) == 0 {
                    libc::setrlimit(libc::RLIMIT_NPROC, &inherited_nproc);
                }

                // RLIMIT_FSIZE (Maximum generated file size): 64 MB
                let fsize_limit = libc::rlimit {
                    rlim_cur: 64 * 1024 * 1024,
                    rlim_max: 64 * 1024 * 1024,
                };
                libc::setrlimit(libc::RLIMIT_FSIZE, &fsize_limit);

                Ok(())
            });
        }

        let start_time = Instant::now();
        let mut child = cmd.spawn()?;

        #[cfg(unix)]
        let child_pid = child
            .id()
            .ok_or_else(|| std::io::Error::other("Failed to retrieve child process ID"))?
            as i32;

        let stdout_pipe = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("Failed to capture stdout stream"))?;
        let stderr_pipe = child
            .stderr
            .take()
            .ok_or_else(|| std::io::Error::other("Failed to capture stderr stream"))?;

        let max_bytes = self.max_output_bytes;

        // Asynchronously read bounded streams
        let stdout_handle = tokio::spawn(async move {
            let mut buf = Vec::new();
            let _ = stdout_pipe
                .take((max_bytes + 1) as u64)
                .read_to_end(&mut buf)
                .await;
            buf
        });

        let stderr_handle = tokio::spawn(async move {
            let mut buf = Vec::new();
            let _ = stderr_pipe
                .take((max_bytes + 1) as u64)
                .read_to_end(&mut buf)
                .await;
            buf
        });

        let timeout = self.default_timeout;

        // Monitor execution with timeout and cancellation token
        tokio::select! {
            exit_res = child.wait() => {
                let status = exit_res?;
                let duration = start_time.elapsed();

                let mut stdout_raw = stdout_handle.await.unwrap_or_default();
                let mut stderr_raw = stderr_handle.await.unwrap_or_default();

                let mut truncated = false;
                if stdout_raw.len() > max_bytes {
                    truncated = true;
                    stdout_raw.truncate(max_bytes);
                }
                if stderr_raw.len() > max_bytes {
                    truncated = true;
                    stderr_raw.truncate(max_bytes);
                }

                Ok(ExecutionOutput {
                    stdout: String::from_utf8_lossy(&stdout_raw).into_owned(),
                    stderr: String::from_utf8_lossy(&stderr_raw).into_owned(),
                    exit_code: status.code(),
                    truncated,
                    duration,
                })
            }
            _ = tokio::time::sleep(timeout) => {
                // Timeout: Kill entire process group using negative PID
                #[cfg(unix)]
                unsafe {
                    libc::kill(-child_pid, libc::SIGKILL);
                }
                let _ = child.wait().await; // Reap child zombie
                Err(ExecutionError::Timeout(timeout))
            }
            _ = cancel_token.cancelled() => {
                // Cancellation: Kill entire process group
                #[cfg(unix)]
                unsafe {
                    libc::kill(-child_pid, libc::SIGKILL);
                }
                let _ = child.wait().await;
                Err(ExecutionError::Cancelled)
            }
        }
    }
}
