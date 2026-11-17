use crate::policy::{AuthorizedCall, is_bwrap_available};
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
/// output caps, minimal containerized bubblewrap confinement, and process-group termination.
pub struct ScopedRunner {
    workspace_root: PathBuf,
    default_timeout: Duration,
    max_output_bytes: usize,
    use_bwrap: bool,
    rlimit_as: u64,
}

impl ScopedRunner {
    /// Creates a new `ScopedRunner` locked to `workspace_root`.
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        Self {
            workspace_root: workspace_root.as_ref().to_path_buf(),
            default_timeout: Duration::from_secs(30),
            max_output_bytes: 512 * 1024, // 512 KB
            use_bwrap: is_bwrap_available(),
            rlimit_as: 4 * 1024 * 1024 * 1024, // 4 GiB default virtual address space ceiling
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

    /// Configures the maximum virtual address space limit (RLIMIT_AS) in bytes.
    pub fn with_rlimit_as(mut self, limit_bytes: u64) -> Self {
        self.rlimit_as = limit_bytes;
        self
    }

    /// Executes an authorized command within the sandboxed environment with cooperative cancellation.
    ///
    /// Accepts ONLY `&AuthorizedCall`, strictly guaranteeing that no unverified or raw commands
    /// can be dispatched without traversing the policy gate.
    pub async fn execute(
        &self,
        call: &AuthorizedCall,
        cancel_token: &CancellationToken,
    ) -> Result<ExecutionOutput, ExecutionError> {
        self.execute_internal(call.exe(), call.args(), cancel_token)
            .await
    }

    /// Internal execution engine. Never exposed outside this module.
    async fn execute_internal(
        &self,
        program: &str,
        args: &[String],
        cancel_token: &CancellationToken,
    ) -> Result<ExecutionOutput, ExecutionError> {
        // If workspace contains Cargo.toml but no Cargo.lock, pre-generate Cargo.lock
        // under sandboxed confinement so read-only workspace_root doesn't block cargo
        // from reading dependencies or creating lockfile.
        let cargo_toml = self.workspace_root.join("Cargo.toml");
        let cargo_lock = self.workspace_root.join("Cargo.lock");
        if cargo_toml.exists() && !cargo_lock.exists() {
            // Pre-create empty lockfile on host so bubblewrap can mount it read-write
            let _ = std::fs::File::create(&cargo_lock);

            if self.use_bwrap && is_bwrap_available() {
                let mut bwrap_lock_args = Vec::new();
                bwrap_lock_args.push("--new-session".to_string());
                bwrap_lock_args.extend([
                    "--ro-bind".to_string(),
                    "/usr".to_string(),
                    "/usr".to_string(),
                ]);
                for path_str in &["/bin", "/lib", "/lib64"] {
                    let p = Path::new(path_str);
                    if let Ok(meta) = std::fs::symlink_metadata(p) {
                        if meta.file_type().is_symlink() {
                            if let Ok(target) = std::fs::read_link(p) {
                                bwrap_lock_args.extend([
                                    "--symlink".to_string(),
                                    target.to_string_lossy().to_string(),
                                    path_str.to_string(),
                                ]);
                            }
                        } else if meta.is_dir() {
                            bwrap_lock_args.extend([
                                "--ro-bind".to_string(),
                                path_str.to_string(),
                                path_str.to_string(),
                            ]);
                        }
                    }
                }
                bwrap_lock_args.extend(["--dir".to_string(), "/etc".to_string()]);
                for etc_file in &[
                    "/etc/ld.so.cache",
                    "/etc/ld.so.conf",
                    "/etc/ld.so.conf.d",
                    "/etc/localtime",
                ] {
                    if Path::new(etc_file).exists() {
                        bwrap_lock_args.extend([
                            "--ro-bind".to_string(),
                            etc_file.to_string(),
                            etc_file.to_string(),
                        ]);
                    }
                }
                bwrap_lock_args.extend([
                    "--proc".to_string(),
                    "/proc".to_string(),
                    "--dev".to_string(),
                    "/dev".to_string(),
                    "--dir".to_string(),
                    "/dev/shm".to_string(),
                    "--size".to_string(),
                    "67108864".to_string(),
                    "--tmpfs".to_string(),
                    "/dev/shm".to_string(),
                    "--size".to_string(),
                    "1073741824".to_string(),
                    "--tmpfs".to_string(),
                    "/tmp".to_string(),
                    "--ro-bind".to_string(),
                    self.workspace_root.to_string_lossy().into_owned(),
                    self.workspace_root.to_string_lossy().into_owned(),
                    "--bind".to_string(),
                    cargo_lock.to_string_lossy().into_owned(),
                    cargo_lock.to_string_lossy().into_owned(),
                    "--chdir".to_string(),
                    self.workspace_root.to_string_lossy().into_owned(),
                    "--unshare-all".to_string(),
                    "--die-with-parent".to_string(),
                    "--".to_string(),
                    "cargo".to_string(),
                    "generate-lockfile".to_string(),
                    "--offline".to_string(),
                ]);
                let _ = std::process::Command::new("bwrap")
                    .args(&bwrap_lock_args)
                    .output();
            } else {
                let _ = std::process::Command::new("cargo")
                    .arg("generate-lockfile")
                    .arg("--offline")
                    .current_dir(&self.workspace_root)
                    .output();
            }
        }

        // If bubblewrap is available and enabled, construct the sandboxed container command
        let (exec_program, exec_args) = if self.use_bwrap && is_bwrap_available() {
            let mut bwrap_args = Vec::new();

            // Prevent TIOCSTI ioctl terminal injection
            bwrap_args.push("--new-session".to_string());

            // Minimal host layout: /usr read-only
            bwrap_args.extend([
                "--ro-bind".to_string(),
                "/usr".to_string(),
                "/usr".to_string(),
            ]);

            // Symlinks and system directories for /bin, /lib, /lib64
            for path_str in &["/bin", "/lib", "/lib64"] {
                let p = Path::new(path_str);
                if let Ok(meta) = std::fs::symlink_metadata(p) {
                    if meta.file_type().is_symlink() {
                        if let Ok(target) = std::fs::read_link(p) {
                            bwrap_args.extend([
                                "--symlink".to_string(),
                                target.to_string_lossy().to_string(),
                                path_str.to_string(),
                            ]);
                        }
                    } else if meta.is_dir() {
                        bwrap_args.extend([
                            "--ro-bind".to_string(),
                            path_str.to_string(),
                            path_str.to_string(),
                        ]);
                    }
                }
            }

            // Narrow /etc to dynamic linker files and timezone on Arch Linux:
            // /etc/ld.so.cache, /etc/ld.so.conf, /etc/ld.so.conf.d, and /etc/localtime.
            // Host files like /etc/passwd, /etc/shadow, /etc/machine-id, /etc/hostname are NOT bound.
            bwrap_args.extend(["--dir".to_string(), "/etc".to_string()]);
            for etc_file in &[
                "/etc/ld.so.cache",
                "/etc/ld.so.conf",
                "/etc/ld.so.conf.d",
                "/etc/localtime",
            ] {
                let p = Path::new(etc_file);
                if p.exists() {
                    bwrap_args.extend([
                        "--ro-bind".to_string(),
                        etc_file.to_string(),
                        etc_file.to_string(),
                    ]);
                }
            }

            // Standard runtime mounts: proc, dev, bounded /dev/shm (64 MiB), bounded /tmp (1 GiB)
            bwrap_args.extend([
                "--proc".to_string(),
                "/proc".to_string(),
                "--dev".to_string(),
                "/dev".to_string(),
                "--dir".to_string(),
                "/dev/shm".to_string(),
                "--size".to_string(),
                "67108864".to_string(),
                "--tmpfs".to_string(),
                "/dev/shm".to_string(),
                "--size".to_string(),
                "1073741824".to_string(),
                "--tmpfs".to_string(),
                "/tmp".to_string(),
            ]);

            // Mask $HOME completely with a 64 MiB bounded tmpfs to prevent secret laundering and resource exhaustion
            if let Ok(home) = std::env::var("HOME") {
                let home_path = PathBuf::from(&home);
                if let Some(parent) = home_path.parent() {
                    bwrap_args.extend(["--dir".to_string(), parent.to_string_lossy().into_owned()]);
                }
                bwrap_args.extend([
                    "--dir".to_string(),
                    home_path.to_string_lossy().into_owned(),
                    "--size".to_string(),
                    "67108864".to_string(),
                    "--tmpfs".to_string(),
                    home_path.to_string_lossy().into_owned(),
                ]);
            }

            // Workspace confinement:
            // Mount workspace root strictly read-only. This protects Cargo.toml,
            // Cargo.lock, tests/, .cargo/, build.rs from being tampered with by
            // model code at test time.
            bwrap_args.extend([
                "--ro-bind".to_string(),
                self.workspace_root.to_string_lossy().into_owned(),
                self.workspace_root.to_string_lossy().into_owned(),
            ]);

            // Overlay src/ as read-write (model mutation zone)
            let src_dir = self.workspace_root.join("src");
            if src_dir.is_dir() {
                bwrap_args.extend([
                    "--bind".to_string(),
                    src_dir.to_string_lossy().into_owned(),
                    src_dir.to_string_lossy().into_owned(),
                ]);
            }

            // Target directory is NOT bound into the container.
            // CARGO_TARGET_DIR is redirected to /tmp/target in ephemeral tmpfs,
            // preventing build-cache poisoning or persistent artifact planting.

            bwrap_args.extend([
                "--chdir".to_string(),
                self.workspace_root.to_string_lossy().into_owned(),
                "--unshare-all".to_string(),
                "--die-with-parent".to_string(),
            ]);

            bwrap_args.push("--".to_string());
            bwrap_args.push(program.to_string());
            bwrap_args.extend_from_slice(args);
            eprintln!("[bwrap argv] bwrap {}", bwrap_args.join(" "));
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
        if let Ok(home) = std::env::var("HOME") {
            cmd.env("HOME", home);
        }
        cmd.env("RUST_BACKTRACE", "1");
        cmd.env("CARGO_TARGET_DIR", "/tmp/target");

        // 3. Process Group Leader & POSIX Resource Limits via pre_exec
        let rlimit_as = self.rlimit_as;
        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(move || {
                // Isolate child into its own process group to prevent orphan leaks
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }

                // RLIMIT_CPU: Soft = 60s, Hard = 61s
                let cpu_limit = libc::rlimit {
                    rlim_cur: 60,
                    rlim_max: 61,
                };
                if libc::setrlimit(libc::RLIMIT_CPU, &cpu_limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }

                // RLIMIT_AS: Configurable virtual address space ceiling (default 4 GiB)
                let as_limit = libc::rlimit {
                    rlim_cur: rlimit_as,
                    rlim_max: rlimit_as,
                };
                let _ = libc::setrlimit(libc::RLIMIT_AS, &as_limit);

                // RLIMIT_FSIZE: 64 MB maximum generated file size
                let fsize_limit = libc::rlimit {
                    rlim_cur: 64 * 1024 * 1024,
                    rlim_max: 64 * 1024 * 1024,
                };
                let _ = libc::setrlimit(libc::RLIMIT_FSIZE, &fsize_limit);

                // RLIMIT_NPROC: Read inherited process limit and never increase it
                let mut current_nproc = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_NPROC, &mut current_nproc) == 0 {
                    let _ = libc::setrlimit(libc::RLIMIT_NPROC, &current_nproc);
                }

                Ok(())
            });
        }

        // Configure piped stdout and stderr for streaming truncation
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let start_time = Instant::now();
        let mut child = cmd.spawn()?;

        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();

        let max_bytes = self.max_output_bytes;

        // Async task to drain stdout up to max_output_bytes + 1 to detect truncation
        let stdout_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut stream) = stdout_pipe {
                let mut take = (&mut stream).take((max_bytes + 1) as u64);
                let _ = take.read_to_end(&mut buf).await;
            }
            buf
        });

        // Async task to drain stderr up to max_output_bytes + 1 to detect truncation
        let stderr_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut stream) = stderr_pipe {
                let mut take = (&mut stream).take((max_bytes + 1) as u64);
                let _ = take.read_to_end(&mut buf).await;
            }
            buf
        });

        // Get process ID for pgid targeting
        let child_pid = child.id().map(|id| id as i32);

        // Wait with wall-clock timeout and cooperative cancellation token
        let timeout_duration = self.default_timeout;
        let wait_result = tokio::select! {
            res = child.wait() => Ok(res?),
            _ = tokio::time::sleep(timeout_duration) => Err(ExecutionError::Timeout(timeout_duration)),
            _ = cancel_token.cancelled() => Err(ExecutionError::Cancelled),
        };

        match wait_result {
            Ok(status) => {
                let duration = start_time.elapsed();
                let mut stdout_raw = stdout_task.await.unwrap_or_default();
                let mut stderr_raw = stderr_task.await.unwrap_or_default();

                let mut truncated = false;
                if stdout_raw.len() > max_bytes {
                    stdout_raw.truncate(max_bytes);
                    truncated = true;
                }
                if stderr_raw.len() > max_bytes {
                    stderr_raw.truncate(max_bytes);
                    truncated = true;
                }

                Ok(ExecutionOutput {
                    stdout: String::from_utf8_lossy(&stdout_raw).into_owned(),
                    stderr: String::from_utf8_lossy(&stderr_raw).into_owned(),
                    exit_code: status.code(),
                    truncated,
                    duration,
                })
            }
            Err(err) => {
                // If timed out or cancelled, aggressively kill entire process group
                if let Some(pid) = child_pid {
                    #[cfg(unix)]
                    unsafe {
                        libc::kill(-pid, libc::SIGKILL);
                        libc::kill(pid, libc::SIGKILL);
                    }
                }
                // Reap the child to prevent zombies
                let _ = child.wait().await;
                let _ = stdout_task.await;
                let _ = stderr_task.await;

                Err(err)
            }
        }
    }
}
