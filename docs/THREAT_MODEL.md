# Threat Model & Security Architecture (`agentic-harness`)

## 1. Threat Profile
The agent harness is designed to execute LLM-proposed actions against a local code repository. Language models frequently hallucinate destructive command arguments, attempt unauthorized file traversals, or invoke shell interpreters with arbitrary sub-shells.

The goal of this security architecture is to provide **fail-closed, defense-in-depth isolation** where untrusted model outputs cannot escape declared workspace boundaries, corrupt the host system, or spawn unmonitored zombie processes.

---

## 2. Security Boundaries: What IS Confined

### A. Command & Execution Surfaces (`exec.rs`, `policy.rs`)
1. **Direct `execve` Dispatch (Zero Shell Parsing):**
   - Commands are never invoked via `sh -c "..."` string concatenation.
   - Programs and argument vectors are passed directly to `tokio::process::Command::new(exe).args(vec)` translating directly into `execve(2)`. This eliminates shell injection, sub-shell interpolation, and metacharacter expansion vulnerabilities.
2. **Fail-Closed Executable Allowlist:**
   - Interpreters (`sh`, `bash`, `zsh`, `python`, `node`, `perl`, `ruby`, `env`, etc.) are explicitly blocked.
   - Only audited tools on the allowlist (`cargo`, `rustc`, `git`, `ls`, `cat`) are permitted.
   - For `git`, only read-only inspection subcommands (`status`, `diff`, `log`, `rev-parse`, `show`, `branch`) are allowed.
3. **Containerized Linux Namespace Isolation (when `bwrap` is active):**
   - Full container namespace unsharing (`--unshare-all`: mount, IPC, PID, network, UTS, cgroups).
   - Root filesystem is mounted read-only (`--ro-bind / /`).
   - Only the designated `workspace_root` is bind-mounted read-write (`--bind workspace workspace`).
   - Private user credential directories (such as `~/.ssh`) are masked with empty tmpfs mounts.
   - Loopback and external network access are completely unshared (zero network exfiltration surface).

### B. Filesystem Jail & Atomic Operations (`tool.rs`, `policy.rs`)
1. **Workspace Boundary Enforcement:**
   - All file tools (`read_file`, `write_file`, `list_dir`) strictly validate paths against canonical `workspace_root`.
   - Lexical `..` path traversals escaping the jail are rejected before touching disk.
   - Symlinks resolving outside the jail are verified via `canonicalize()` and rejected.
2. **Protected File Exclusions:**
   - Access to `.git/` internals, `.env` secret files, and `~/.ssh` directories is strictly blocked by policy.
3. **Atomic Writes with Unified Diff:**
   - File edits write to a temporary file (`.tmp.<name>_<nanos>`) in the target directory and are swapped into place via atomic `rename(2)`.
   - A line-by-line unified diff is returned to provide transparent auditability of all file modifications.

### C. Process Tree Control & Resource Limits (`exec.rs`)
1. **Orphan-Proof Process Group Cleanup:**
   - `pre_exec` sets `setpgid(0, 0)` so the spawned process becomes its own process group leader.
   - On timeout or cancellation (`CancellationToken`), `libc::kill(-pgid, SIGKILL)` terminates the entire process tree simultaneously.
   - The child is reaped with `child.wait().await`, preventing zombie accumulation.
2. **POSIX Resource Limits (`setrlimit`):**
   - **`RLIMIT_AS` (Virtual Address Space):** Strict 2 GiB ceiling per process.
   - **`RLIMIT_CPU`:** Soft limit = 60s, Hard limit = 61s (kernel automatically sends `SIGKILL` on hard limit).
   - **`RLIMIT_FSIZE`:** 64 MB maximum generated file size.
   - **`RLIMIT_NPROC`:** Reads inherited user process limits and never increases them (preventing privilege expansion while avoiding `EAGAIN` fork failure).
3. **I/O Stream Caps:**
   - Standard output and standard error are capped at 512 KB via `tokio::io::AsyncReadExt::take`. Truncation flags are explicitly recorded in execution output.

---

## 3. What is NOT Confined (When `bwrap` is Absent)

> [!WARNING]
> Without Linux user namespace support or `bwrap`, the runner falls back to unconfined execution under the calling user's UID.

If bubblewrap (`bwrap`) is absent:
1. **Host Read Access:** Subprocesses run under the operator's UID and could read non-protected files accessible to that user on the host filesystem.
2. **Network Access:** Subprocesses retain the host network interface and could transmit outbound packets.
3. **Arbitrary Build Script Execution:** Running `cargo test` executes arbitrary procedural macros and build scripts (`build.rs`). Without `bwrap`, malicious crates or compromised dependencies run unconfined code on the host.

**Fail-Closed Policy Guard:**
`DefaultPolicy` enforces `enforce_bwrap = true`. If `bwrap` is not detected on PATH, invoking `cargo` or interpreters is automatically **Denied**, preventing silent fallback to unconfined host execution.
