# Threat Model & Security Architecture (`agentic-harness`)

## 1. Threat Profile
The agent harness executes LLM-proposed actions against a local code repository. Language models frequently hallucinate destructive command arguments, attempt unauthorized file traversals, invoke shell interpreters with arbitrary sub-shells, or perform reward hacking by rewriting tests to artificially pass verification.

The goal of this security architecture is to provide **fail-closed, type-enforced, kernel-isolated confinement** where untrusted model outputs cannot escape declared workspace boundaries, access host secrets, execute arbitrary commands, or tamper with test verification.

---

## 2. Security Architecture & Invariants

### A. Type-Enforced Execution Choke Point (`AuthorizedCall`)
- **Unrepresentable Policy Bypass:**
  - `AuthorizedCall` has strictly private fields (`exe: String, args: Vec<String>`) with no public constructor.
  - The ONLY mechanism to construct an `AuthorizedCall` is via successful evaluation by a `Policy` (`validate_command` / `evaluate`).
  - `ScopedRunner::execute(&self, call: &AuthorizedCall, cancel: &CancellationToken)` accepts ONLY `&AuthorizedCall`. All raw, untrusted execution entry points (`ScopedRunner::run`) have been deleted.
  - Verification hooks call the identical `Policy::evaluate` method with zero exceptions. If verification is misconfigured with an interpreter (e.g. `sh`), it is denied at the gate, producing zero execution.

### B. Shrunk Allowlist & Tool Boundary
- **Command Allowlist:** Strictly limited to `["cargo", "rustc"]`.
- **Elimination of `cat`, `ls`, `git`:**
  - `cat` and `ls` circumvent path jailing because argument paths are not verified like file-tool paths.
  - `git` has a broad attack surface: `git diff` can execute arbitrary external diff/pager drivers from repo config, and commands like `git checkout` or `git branch -D` mutate repository state.
  - All file reads, writes, and directory listings MUST traverse the typed, path-jailed tools (`read_file`, `write_file`, `list_dir`).
- **Interpreters Prohibited:** All shells and script interpreters (`sh`, `bash`, `zsh`, `python*`, `node`, `perl`, `ruby`, `env`) are blocked by policy.

### C. Verifier Integrity & Reward Hacking Prevention
- **Confinement of Agent Mutations to `src/`:**
  - The agent's `write_file` tool is restricted strictly to writing within `<workspace>/src/`.
  - Attempts to target `tests/`, `Cargo.toml`, or verification scripts outside `src/` are rejected with `PolicyDecision::Deny`.
  - The test suite and verification logic remain completely immutable to the agent, eliminating the possibility of self-repair cheating (e.g. replacing failing assertions with `assert!(true)` or `exit 0`).

### D. Minimal Sandbox Mounts & Secret Isolation
When Bubblewrap (`bwrap`) is active:
- **No Global Root Bind:** Dropped `--ro-bind / /`. The sandbox mounts ONLY what is strictly required:
  - `--ro-bind /usr /usr`
  - Symlinks / ro-binds for `/bin`, `/lib`, `/lib64`
  - `--ro-bind /etc /etc`
  - `--proc /proc --dev /dev --tmpfs /tmp`
- **Masking of `$HOME` with tmpfs:**
  - The user's home directory is masked by mounting an empty `--tmpfs` over `$HOME`. Host credentials (`~/.ssh`, `~/.aws`, `~/.config`, `.env`) do not exist inside the sandbox.
  - Secret laundering (reading a host secret inside the sandbox and persisting it into the workspace) is physically impossible because the host filesystem outside `/usr` and `/etc` is not mounted.
- **Isolated Workspace Mount:**
  - Only the target workspace directory is bind-mounted read-write (`--bind <workspace> <workspace>`).
- **Namespace Unsharing:**
  - `--unshare-all` unshares mount, IPC, PID, network, UTS, and cgroup namespaces.
  - Zero network exfiltration surface (network namespace is disconnected).
  - `--die-with-parent` guarantees container cleanup if the parent harness process terminates.

### E. Process Tree Lifecycle & Resource Limits
- **Process Group Termination:**
  - In `pre_exec`, `setpgid(0, 0)` isolates the child into a dedicated process group.
  - On timeout or `CancellationToken` cancellation, `libc::kill(-pgid, SIGKILL)` terminates the entire process group simultaneously, preventing orphan sleep processes.
  - Reaping via `child.wait().await` prevents zombie processes.
- **POSIX Resource Limits (`setrlimit`):**
  - `RLIMIT_AS`: 2 GiB virtual address space ceiling per process.
  - `RLIMIT_CPU`: 60s soft limit, 61s hard limit (`SIGKILL`).
  - `RLIMIT_FSIZE`: 64 MB maximum generated file size.
  - `RLIMIT_NPROC`: Inherited from the calling user and never raised (preventing privilege escalation while avoiding Linux `EAGAIN` fork failures).
- **Stream Caps:**
  - Standard output and standard error are capped at 512 KB via `tokio::io::AsyncReadExt::take`.

---

## 3. Empirical Verification Evidence & Confinement Boundaries

### A. Zero-Dependency Fixture Verification
- **Verified Environment:** Confinement and self-repair loops were empirically verified using a zero-dependency Cargo fixture crate (`reconciliation_fixture`) under offline compilation (`cargo test --offline`).
- **`RLIMIT_AS` (2 GiB Ceiling):** Verified sufficient for compiling and running unit tests on zero-dependency fixtures without triggering out-of-memory aborts.
- **`CARGO_HOME` on tmpfs:** When `$HOME` is masked by an empty tmpfs, Cargo operates with `$HOME/.cargo` in temporary memory, avoiding access to the host user's personal Cargo configuration or registry cache.

### B. Confirmed Confinement Boundaries & Non-Supported Workloads
1. **No External Registry Dependencies:** Because network namespaces are unshared (`--unshare-all`) and `$HOME/.cargo` is an empty tmpfs, the sandboxed runner cannot fetch or download external crates from `crates.io`. Projects requiring third-party crates must have dependencies pre-vendored or pre-built before entering the sandbox.
2. **No Path Dependencies Outside Workspace:** Path dependencies resolving outside the declared workspace jail (e.g. `../agentic-harness` from a consumer crate) do not exist inside the container because only `<workspace>` is bind-mounted. All source code and local dependencies must reside within the workspace tree.
3. **Memory Limits for Large Real-World Crates:** Empirical evidence for `RLIMIT_AS = 2 GiB` is strictly limited to small fixture crates. Complex procedural macros and heavy linking trees (e.g. `tokio`, `serde_derive`, `wasm-bindgen`) may require larger virtual address space limits or pre-warmed target caches.

---

## 4. Failure Modes Without Bubblewrap

If `bwrap` is not installed on the host system:
- `DefaultPolicy` enforces `enforce_bwrap = true`. Any invocation of `cargo` or `rustc` fails closed with:
  `"Execution denied: bubblewrap (bwrap) container isolation is not available on PATH"`.
- The runner NEVER silently degrades to unconfined host execution.
