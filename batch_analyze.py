"""
Batch repository deep analysis — checkpoint/resume + retry + concurrency.

Features:
  - Checkpoint: survive Ctrl+C, power loss, any interruption
  - Retry: transient errors (503, timeout, connection) auto-retry with backoff
  - Concurrent: N repos analyzed in parallel (controlled by --workers)
  - Dedup: duplicate repos filtered by path

Usage:
    python batch_analyze.py                     # Run with default 2 workers
    python batch_analyze.py --workers 4         # 4 concurrent repos
    python batch_analyze.py --status            # Show progress
    python batch_analyze.py --reset <name>      # Reset a repo to pending
    python batch_analyze.py --reset-failed      # Reset all failed repos
    python batch_analyze.py --force <name>      # Re-run a specific repo (ignores checkpoint)
"""

import json
import os
import re
import sys
import subprocess
import shutil
import time
import random
import threading
import queue
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path("D:/work/ant/ALH")
PFT_DEEP_DIR = PROJECT_ROOT / "pft-deep"
CHECKPOINT_FILE = PFT_DEEP_DIR / "checkpoint.json"
ANALYSIS_OUTPUT = PFT_DEEP_DIR / "analysis_output"
BATCH_LOG_DIR = PFT_DEEP_DIR / "logs" / "batch"
PROCESS_IDLE_TIMEOUT = int(os.getenv("PFT_DEEP_IDLE_TIMEOUT", "2700"))
PROCESS_MAX_RUNTIME = int(os.getenv("PFT_DEEP_MAX_RUNTIME", "14400"))

# Retry settings
MAX_RETRIES = 10                         # Max retry attempts per repo
RETRY_BASE_DELAY = 15                    # Base delay seconds (exponential backoff)
RETRY_MAX_DELAY = 600                    # Max delay cap
RETRYABLE_PATTERNS = [
    r"503", r"502", r"504",
    r"Server Error", r"Service Unavailable", r"Too Many Requests",
    r"timeout", r"timed out", r"Time-out",
    r"Connection.*error", r"connection.*refused", r"connection.*reset",
    r"Rate limit", r"rate.limit",
    r"429", r"Too Many Requests", r"Concurrency limit exceeded",
    r"Temporary failure", r"transient",
    r"Overloaded", r"overloaded",
    r"IDLE_TIMEOUT", r"MAX_RUNTIME",
]

# Non-retryable — permanent failures
FATAL_PATTERNS = [
    r"ModuleNotFoundError", r"ImportError", r"SyntaxError",
    r"FileNotFoundError", r"No such file",
    r"Permission denied", r"Access denied",
    r"invalid.*argument", r"not found",
]

CHECKPOINT_LOCK = threading.Lock()


def safe_print(message=""):
    """Print without letting console encoding errors kill a batch run."""
    text = str(message)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        print(safe_text)


# All repos to analyze: (relative_path, display_name, stage_name)
# Deduplication: same rel_path → only one entry
SEEN_PATHS = set()
REPOS = []
RAW_REPOS = [
    # Stage 2
    ("stages/stage-2/repos/gpt-researcher", "gpt-researcher", "stage-2"),
    ("stages/stage-2/repos/open_deep_research", "open_deep_research", "stage-2"),
    ("stages/stage-2/repos/storm", "storm", "stage-2"),
    ("stages/stage-2/repos/khoj", "khoj", "stage-2"),
    ("stages/stage-2/repos/onyx", "onyx", "stage-2"),
    ("stages/stage-2/repos/anything-llm", "anything-llm", "stage-2"),
    ("stages/stage-2/repos/ragflow", "ragflow", "stage-2"),
    ("stages/stage-2/repos/mem0", "mem0", "stage-2"),
    ("stages/stage-2/repos/letta", "letta", "stage-2"),
    # Stage 3
    ("stages/stage-3/repos/learn-claude-code", "learn-claude-code", "stage-3"),
    ("stages/stage-3/repos/claw0", "claw0", "stage-3"),
    ("stages/stage-3/repos/hello-agents", "hello-agents", "stage-3"),
    ("stages/stage-3/repos/openclaw", "openclaw", "stage-3"),
    ("stages/stage-3/repos/hermes-agent", "hermes-agent", "stage-3"),
    ("stages/stage-3/repos/CyberClaw", "CyberClaw", "stage-3"),
    ("stages/stage-3/repos/langgraph", "langgraph", "stage-3"),
    # Stage 4
    ("stages/stage-4/repos/openai-agents-python", "openai-agents-python", "stage-4"),
    # Stage 6
    ("stages/stage-6/repos/browser-use", "browser-use", "stage-6"),
    ("stages/stage-6/repos/UI-TARS-desktop", "UI-TARS-desktop", "stage-6"),
    # Resources
    ("resources/codex", "codex", "resources"),
    ("resources/opencode", "opencode", "resources"),
    ("resources/OpenHands", "OpenHands", "resources"),
    ("resources/SWE-agent", "SWE-agent", "resources"),
    ("resources/pi", "pi", "resources"),
    ("resources/deer-flow", "deer-flow", "resources"),
    ("resources/smolagents", "smolagents", "resources"),
    ("resources/HelloAgents", "HelloAgents", "resources"),
    ("resources/GenAI_Agents", "GenAI_Agents", "resources"),
    ("resources/agents-towards-production", "agents-towards-production", "resources"),
    ("resources/Qwen-Agent", "Qwen-Agent", "resources"),
    ("resources/ai-agents-for-beginners", "ai-agents-for-beginners", "resources"),
    # Legacy
    ("legacy/autogen", "autogen", "legacy"),
]

for path, name, stage in RAW_REPOS:
    normalized = str(PROJECT_ROOT / path).lower().replace("\\", "/")
    if normalized not in SEEN_PATHS:
        SEEN_PATHS.add(normalized)
        REPOS.append((path, name, stage))
    else:
        safe_print(f"DEDUP: skipped duplicate path {path}")


# ── Checkpoint ─────────────────────────────────────────────

def _load_checkpoint_raw():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint_raw(data):
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, CHECKPOINT_FILE)


def init_checkpoint(reset_running=True):
    """Initialize checkpoint, preserving done entries."""
    with CHECKPOINT_LOCK:
        existing = _load_checkpoint_raw()
        new_cp = {}
        for rel_path, name, stage in REPOS:
            abs_path = PROJECT_ROOT / rel_path
            if not abs_path.is_dir():
                new_cp[name] = {
                    "rel_path": rel_path, "stage": stage,
                    "status": "skipped", "reason": "Directory not found",
                    "retries": 0, "updated": datetime.now().isoformat(),
                }
                continue

            if name in existing and existing[name].get("status") == "done":
                new_cp[name] = existing[name]
            elif name in existing and existing[name].get("status") == "running":
                if reset_running:
                    # A new batch process cannot own an old running marker.
                    new_cp[name] = {**existing[name], "status": "pending",
                                    "reason": "Reset from stale running state",
                                    "updated": datetime.now().isoformat()}
                else:
                    new_cp[name] = existing[name]
            else:
                prev = existing.get(name, {})
                new_cp[name] = {
                    "rel_path": rel_path, "stage": stage,
                    "status": "pending", "reason": prev.get("reason", ""),
                    "retries": prev.get("retries", 0),
                    "updated": datetime.now().isoformat(),
                }

        _save_checkpoint_raw(new_cp)
        return new_cp


def update_checkpoint(name, **kwargs):
    """Thread-safe checkpoint update for a single entry."""
    with CHECKPOINT_LOCK:
        cp = _load_checkpoint_raw()
        if name in cp:
            cp[name].update(kwargs)
            cp[name]["updated"] = datetime.now().isoformat()
            _save_checkpoint_raw(cp)


def get_checkpoint():
    with CHECKPOINT_LOCK:
        return _load_checkpoint_raw()


# ── Error Classification ───────────────────────────────────

def is_transient_error(output_text):
    """Check if error output indicates a transient (retryable) failure."""
    text = output_text.lower() if output_text else ""
    for pat in FATAL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False
    for pat in RETRYABLE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    # Default: classify non-zero exit with unknown output as retryable (safety)
    return True


# ── Analysis Runner ────────────────────────────────────────

def artifact_root(name):
    """Return the generated deep-analysis document root, if present."""
    output_dir = ANALYSIS_OUTPUT / name
    nested_dir = output_dir / name
    if (nested_dir / "index.md").is_file():
        return nested_dir
    if (output_dir / "index.md").is_file():
        return output_dir
    return None


def validate_analysis_output(name):
    """Validate that pft-deep produced a usable document set for one repo."""
    root = artifact_root(name)
    if root is None:
        return False, "missing index.md"

    index_path = root / "index.md"
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, "index.md is not valid UTF-8"

    md_files = list(root.rglob("*.md"))
    if len(index_text.strip()) < 500:
        return False, "index.md is too small"
    if len(md_files) < 4:
        return False, f"too few markdown files ({len(md_files)})"
    return True, f"{len(md_files)} markdown files"


def _terminate_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def _reader_thread(stream, out_queue):
    try:
        for line in stream:
            out_queue.put(line)
    finally:
        out_queue.put(None)


def run_analysis_once(name, rel_path, stage):
    """Run pft-deep once. Returns (success: bool, output_text: str)."""
    repo_dir = PROJECT_ROOT / rel_path
    output_dir = ANALYSIS_OUTPUT / name

    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    timestamp = datetime.now().strftime("%H:%M:%S")
    safe_print(f"\n{'='*60}")
    safe_print(f" [{timestamp}] START: {name}  ({stage})")
    safe_print(f" Source: {repo_dir}")
    safe_print(f"{'='*60}")

    cmd = [
        sys.executable, str(PFT_DEEP_DIR / "main.py"),
        "--dir", str(repo_dir),
        "--deep",
        "--output", str(output_dir),
        "-n", name,
        "--language", "Chinese",
        "--max-abstractions", "8",
    ]

    start_time = time.time()
    last_output_time = start_time
    all_output = []
    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_log = BATCH_LOG_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}.log"

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["LLM_PARALLEL_WORKERS"] = os.getenv("BATCH_LLM_PARALLEL_WORKERS", "1")
        env.setdefault("LLM_GLOBAL_MAX_CONCURRENCY", "2")
        env.setdefault("LLM_SLOT_STALE_SECONDS", "7200")

        process = subprocess.Popen(
            cmd,
            cwd=str(PFT_DEEP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env,
        )

        out_queue = queue.Queue()
        reader = threading.Thread(
            target=_reader_thread,
            args=(process.stdout, out_queue),
            daemon=True,
        )
        reader.start()

        with open(run_log, "w", encoding="utf-8") as log_handle:
            while True:
                try:
                    line = out_queue.get(timeout=1)
                except queue.Empty:
                    now = time.time()
                    if process.poll() is not None:
                        break
                    if now - last_output_time > PROCESS_IDLE_TIMEOUT:
                        msg = (
                            f"IDLE_TIMEOUT: no child output for "
                            f"{PROCESS_IDLE_TIMEOUT}s; terminating process"
                        )
                        safe_print(f"  [{name}] {msg}")
                        all_output.append(msg)
                        log_handle.write(msg + "\n")
                        _terminate_process(process)
                        break
                    if now - start_time > PROCESS_MAX_RUNTIME:
                        msg = (
                            f"MAX_RUNTIME: exceeded {PROCESS_MAX_RUNTIME}s; "
                            f"terminating process"
                        )
                        safe_print(f"  [{name}] {msg}")
                        all_output.append(msg)
                        log_handle.write(msg + "\n")
                        _terminate_process(process)
                        break
                    continue

                if line is None:
                    break

                last_output_time = time.time()
                line = line.rstrip()
                log_handle.write(line + "\n")
                log_handle.flush()
                # Truncate very long lines for readability
                if len(line) > 200:
                    display = line[:200] + "..."
                else:
                    display = line
                safe_print(f"  [{name}] {display}")
                all_output.append(line)

        process.wait()
        elapsed = time.time() - start_time
        output_text = "\n".join(all_output)

        if process.returncode == 0:
            valid, reason = validate_analysis_output(name)
            if valid:
                safe_print(f"  [{name}] OK ({elapsed:.0f}s, {reason})")
                return True, output_text
            safe_print(f"  [{name}] FAIL (invalid output: {reason}, {elapsed:.0f}s)")
            return False, output_text + f"\nINVALID_OUTPUT: {reason}"
        else:
            safe_print(f"  [{name}] FAIL (exit={process.returncode}, {elapsed:.0f}s)")
            return False, output_text

    except Exception as e:
        elapsed = time.time() - start_time
        output_text = "\n".join(all_output) + f"\nEXCEPTION: {e}"
        safe_print(f"  [{name}] EXCEPTION after {elapsed:.0f}s: {e}")
        return False, output_text


def run_analysis_with_retry(name, rel_path, stage):
    """Run analysis with up to MAX_RETRIES retries on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        ok, output = run_analysis_once(name, rel_path, stage)

        if ok:
            return True, output, attempt

        if not is_transient_error(output):
            safe_print(f"  [{name}] FATAL error (non-retryable), giving up.")
            return False, output, attempt

        if attempt < MAX_RETRIES:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            jitter = random.uniform(0, delay * 0.3)
            wait = delay + jitter
            safe_print(f"  [{name}] Transient error, retry {attempt}/{MAX_RETRIES} in {wait:.0f}s...")
            time.sleep(wait)
        else:
            safe_print(f"  [{name}] All {MAX_RETRIES} retries exhausted.")

    return False, output, MAX_RETRIES


def copy_to_stage(name, stage):
    """Copy analysis output to the stage directory."""
    stage_dir = PROJECT_ROOT / "stages" / stage / "analysis"
    stage_dir.mkdir(parents=True, exist_ok=True)

    src_dir = artifact_root(name)
    if src_dir is None:
        safe_print(f"  [{name}] WARNING: usable output dir not found")
        return False

    dest_dir = stage_dir / f"{name}-deep-analysis"
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.copytree(str(src_dir), str(dest_dir))

    index_src = src_dir / "index.md"
    index_dest = stage_dir / f"{name}-deep-analysis.md"
    shutil.copy2(str(index_src), str(index_dest))
    safe_print(f"  [{name}] Copied document set: {dest_dir}")
    safe_print(f"  [{name}] Copied entrypoint: {index_dest}")

    return True


# ── Worker (one repo per thread) ───────────────────────────

def process_one_repo(name, entry):
    """Worker: process a single repo. Called from thread pool."""
    safe_print(f"\n[WORKER] Picked up: {name}")

    update_checkpoint(name, status="running")

    try:
        success, output, attempts = run_analysis_with_retry(
            name, entry["rel_path"], entry["stage"]
        )
    except KeyboardInterrupt:
        update_checkpoint(name, status="pending", reason="Interrupted")
        raise
    except Exception as e:
        update_checkpoint(name, status="error", reason=str(e)[:200])
        safe_print(f"  [{name}] UNEXPECTED ERROR: {e}")
        return False

    if success:
        if copy_to_stage(name, entry["stage"]):
            update_checkpoint(name, status="done", reason="", retries=attempts - 1)
            safe_print(f"  [{name}] COMPLETED (attempts={attempts})")
            return True
        update_checkpoint(
            name,
            status="error",
            reason="Failed to copy verified analysis output",
            retries=attempts - 1,
        )
        return False
    else:
        update_checkpoint(name, status="error",
                          reason=f"Failed after {attempts} attempts",
                          retries=attempts - 1)
        return False


# ── Status Display ─────────────────────────────────────────

def show_status(checkpoint=None):
    if checkpoint is None:
        checkpoint = get_checkpoint()

    stats = {"pending": 0, "done": 0, "error": 0, "running": 0, "skipped": 0}
    for entry in checkpoint.values():
        s = entry.get("status", "pending")
        stats[s] = stats.get(s, 0) + 1

    total = sum(stats.values())
    done = stats.get("done", 0)
    pending = stats.get("pending", 0)
    error = stats.get("error", 0)
    skipped = stats.get("skipped", 0)
    running = stats.get("running", 0)

    progress = f"{done}/{total}" if total > 0 else "0/0"
    pct = f"({100*done//total}%)" if total > 0 else ""

    safe_print(f"\n{'='*60}")
    safe_print(f"  BATCH ANALYSIS: {progress} {pct}")
    safe_print(f"{'='*60}")
    safe_print(f"  [DONE]  {done}")
    safe_print(f"  [RUN.]  {running}")
    safe_print(f"  [TODO]  {pending}")
    safe_print(f"  [FAIL]  {error}")
    safe_print(f"  [SKIP]  {skipped}")
    safe_print(f"{'='*60}\n")

    for name in sorted(checkpoint.keys()):
        entry = checkpoint[name]
        s = entry.get("status", "?")
        icon = {"done": "DONE", "pending": "TODO", "error": "FAIL",
                "running": "RUN.", "skipped": "SKIP"}.get(s, "????")
        stage = entry.get("stage", "?")
        retries = entry.get("retries", 0)
        retry_str = f" (retries={retries})" if retries > 0 else ""
        reason = f" -- {entry['reason']}" if entry.get("reason") else ""
        safe_print(f"  [{icon}] [{stage}] {name}{retry_str}{reason}")


# ── Main ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch repo deep analysis — checkpoint, retry, concurrency")
    parser.add_argument("--status", action="store_true",
                        help="Show progress and exit")
    parser.add_argument("--reset", metavar="NAME",
                        help="Reset one repo to pending")
    parser.add_argument("--reset-failed", action="store_true",
                        help="Reset all failed repos to pending")
    parser.add_argument("--force", metavar="NAME",
                        help="Re-run one repo ignoring its checkpoint state")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of concurrent repos (default: 2)")
    parser.add_argument("--one", action="store_true",
                        help="Process exactly one pending repo and exit")
    args = parser.parse_args()

    if args.status:
        checkpoint = get_checkpoint()
        if not checkpoint:
            checkpoint = init_checkpoint(reset_running=False)
        show_status(checkpoint)
        return 0

    # Init
    checkpoint = init_checkpoint()

    if args.reset:
        if args.reset in checkpoint:
            update_checkpoint(args.reset, status="pending", reason="Manual reset", retries=0)
            safe_print(f"Reset: {args.reset} -> pending")
        else:
            safe_print(f"Not found: {args.reset}")
        return 0

    if args.reset_failed:
        count = 0
        for name, entry in checkpoint.items():
            if entry.get("status") == "error":
                update_checkpoint(name, status="pending", reason="Reset via --reset-failed", retries=0)
                count += 1
        safe_print(f"Reset {count} failed repos to pending")
        return 0

    if args.force:
        name = args.force
        if name not in checkpoint:
            safe_print(f"Not found: {name}")
            return 1
        entry = checkpoint[name]
        safe_print(f"Force re-running: {name}")
        process_one_repo(name, entry)
        show_status()
        return 0

    # ── Concurrent batch mode ──
    pending = [(n, e) for n, e in checkpoint.items() if e.get("status") == "pending"]

    if not pending:
        safe_print("All done! Nothing pending.")
        show_status(checkpoint)
        return 0

    if args.one:
        name, entry = pending[0]
        process_one_repo(name, entry)
        show_status()
        return 0

    workers = args.workers
    safe_print(f"\nProcessing {len(pending)} repos with {workers} workers")
    safe_print(f"Retry config: max={MAX_RETRIES}, base_delay={RETRY_BASE_DELAY}s, max_delay={RETRY_MAX_DELAY}s")
    safe_print(f"Press Ctrl+C to stop gracefully (in-progress repos will be reset to pending).\n")

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            submitted = 0

            # Submit up to `workers` tasks at a time
            while submitted < len(pending):
                # Refresh checkpoint each round to pick up resets
                current_cp = get_checkpoint()
                still_pending = [(n, current_cp[n]) for n, _ in pending
                                 if current_cp.get(n, {}).get("status") == "pending"]

                if not still_pending and not futures:
                    break

                # Submit new work if slots available
                while len(futures) < workers and still_pending:
                    name, entry = still_pending.pop(0)
                    if name not in futures:
                        future = executor.submit(process_one_repo, name, entry)
                        futures[name] = future

                # Wait for at least one to complete
                if futures:
                    done_futures = []
                    for name, future in list(futures.items()):
                        if future.done():
                            try:
                                future.result()
                            except KeyboardInterrupt:
                                raise
                            except Exception as e:
                                safe_print(f"  [{name}] Worker crashed: {e}")
                            done_futures.append(name)

                    for name in done_futures:
                        del futures[name]
                        submitted += 1

                    # If nothing done yet, wait for next
                    if not done_futures:
                        try:
                            for future in as_completed(list(futures.values()), timeout=5):
                                for name, f in list(futures.items()):
                                    if f == future:
                                        try:
                                            future.result()
                                        except KeyboardInterrupt:
                                            raise
                                        except Exception:
                                            pass
                                        del futures[name]
                                        submitted += 1
                                        break
                                break  # Only process one completion per iteration
                        except TimeoutError:
                            pass  # No completion within 5s, loop again
                        except KeyboardInterrupt:
                            raise

                else:
                    break

            # Wait for remaining futures
            for name, future in list(futures.items()):
                try:
                    future.result()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass
                del futures[name]
                submitted += 1

    except KeyboardInterrupt:
        safe_print("\n\n*** Interrupted. Resetting running repos to pending... ***")
        cp = get_checkpoint()
        for name, entry in cp.items():
            if entry.get("status") == "running":
                update_checkpoint(name, status="pending", reason="Interrupted by user")
        safe_print("All running repos reset to pending. Run again to resume.\n")

    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
