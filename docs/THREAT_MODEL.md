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
- **Read-Only Workspace Root in Sandbox:**
  - In bubblewrap, `<workspace_root>` is mounted `--ro-bind` at the kernel level.
  - Only `<workspace_root>/src` is mounted `--bind` (read-write).
  - Crucial project files (`Cargo.toml`, `Cargo.lock`, `tests/`, `.cargo/config.toml`, `build.rs`) cannot be altered by test-time code execution; write syscalls fail with `EROFS` (Read-only file system).
- **Libtest Integrity & Per-Test-Name Matching:**
  - The verifier does not rely on process exit codes alone.
  - It parses `running N tests`, verifies each individual `test <name> ... ok` line appears with unique test names, and requires the summary `test result: ok. N passed; 0 failed` with $N == \text{expected}$.
  - Any premature process termination (e.g. `std::process::exit(0)`) or forged summary lines without matching test names are rejected as verification failures.

### D. Minimal Sandbox Mounts & Secret Isolation
When Bubblewrap (`bwrap`) is active:
- **Terminal Injection Prevention (`--new-session`):**
  - Isolates the sandboxed process in a new session, preventing `TIOCSTI` ioctl terminal input injection into the host shell.
- **Narrowed `/etc` Mounts:**
  - Rather than binding entire `/etc`, only dynamic linker and timezone files are bound:
    - `/etc/ld.so.cache`, `/etc/ld.so.conf`, `/etc/ld.so.conf.d`
    - `/etc/localtime`
  - Sensitive files (`/etc/passwd`, `/etc/shadow`, `/etc/machine-id`, `/etc/hostname`) are completely unmapped.
- **No Global Root Bind:** Dropped `--ro-bind / /`. The sandbox mounts ONLY what is strictly required:
  - `--ro-bind /usr /usr`
  - Symlinks / ro-binds for `/bin`, `/lib`, `/lib64`
  - `--dir /etc` with specific linker/timezone file overlays
  - `--proc /proc --dev /dev --size 1073741824 --tmpfs /tmp`
- **Masking of `$HOME` with tmpfs:**
  - The user's home directory is masked by mounting an empty `--tmpfs` over `$HOME`. Host credentials (`~/.ssh`, `~/.aws`, `~/.config`, `.env`) do not exist inside the sandbox.
  - Secret laundering (reading a host secret inside the sandbox and persisting it into the workspace) is physically impossible because the host filesystem outside `/usr` and `/etc` is not mounted.
- **Isolated Workspace Mount & Persistence Channel Removal:**
  - The workspace root is mounted `--ro-bind`, with only `src/` mounted read-write.
  - `target/` is NOT mounted from the host workspace; `CARGO_TARGET_DIR` is redirected to `/tmp/target` in ephemeral tmpfs. This removed a persistence channel, preventing build-cache poisoning or persistent artifact planting across invocations.
- **Bounded `/tmp`, `$HOME`, and `/dev/shm` tmpfs:**
  - Bubblewrap mounts `/tmp` with `--size 1073741824 --tmpfs /tmp` (1 GiB ceiling), preventing disk or memory exhaustion from runaway sandboxed writes.
  - Bubblewrap mounts `$HOME` with `--size 67108864 --tmpfs $HOME` (64 MiB ceiling).
  - Bubblewrap mounts `/dev/shm` with `--dir /dev/shm --size 67108864 --tmpfs /dev/shm` (64 MiB ceiling).
  - Bubblewrap's root `/` and `/dev` base mount are created internally by bwrap without user-specified size options; all user-accessible writable locations (`/tmp`, `$HOME`, `/dev/shm`, and `<workspace>/src`) are strictly size-capped or write-confined.
- **Single Named Exception: Sandboxed Lockfile Generation (`src/exec.rs:89-173`):**
  - If a workspace contains `Cargo.toml` without a `Cargo.lock`, `ScopedRunner::execute_internal` pre-creates an empty `Cargo.lock` on the host and runs `cargo generate-lockfile --offline` in a dedicated bubblewrap jail with `--bind Cargo.lock` before executing user code.
  - This is an internal prerequisite required by `--locked` when `<workspace_root>` is mounted read-only. It only works for dependency-free crates because `--offline` forbids network access.
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
  - `RLIMIT_AS`: 4 GiB virtual address space ceiling per process (default, configurable per runner). RLIMIT_RSS is not enforced by the Linux kernel; a real RSS cap needs cgroup v2 memory.max; not implemented.
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

### C. Stated Residual Risk: In-Process Test Runner Address Space Sharing
- **Vulnerability Surface:** Standard `cargo test` executes the test harness and the tested library inside the same Linux process address space.
- **Identified Residual Attack:** While `println!` is captured by libtest and forged summaries without matching test records are rejected by the harness parser, an intentionally adversarial payload could theoretically use direct low-level syscalls (`libc::write(1, ...)`) to forge exact per-test lines (`test <name> ... ok`) and summary lines (`test result: ok. N passed; 0 failed`) before calling `exit(0)`.
- **Production Defense:** For hostile multi-tenant environments where the agent is explicitly adversarial rather than faulty, test execution must be isolated out-of-process (e.g., using `cargo nextest` or JUnit/JSON structured output streams captured across dedicated IPC pipes rather than parsing stdout).

---

## 4. Failure Modes Without Bubblewrap

If `bwrap` is not installed on the host system:
- `DefaultPolicy` enforces `enforce_bwrap = true`. Any invocation of `cargo` or `rustc` fails closed with:
  `"Execution denied: bubblewrap (bwrap) container isolation is not available on PATH"`.
- The runner NEVER silently degrades to unconfined host execution.
