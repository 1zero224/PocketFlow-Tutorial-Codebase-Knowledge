from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

from google import genai
import os
import logging
import json
import requests
import threading
import time
import random
import tempfile
import ctypes
from datetime import datetime

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"
_cache_lock = threading.RLock()
_telemetry_lock = threading.Lock()


def _positive_int_env(name, default):
    try:
        parsed = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float_env(name, default):
    try:
        parsed = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _llm_slot_dir():
    configured = os.getenv("LLM_LOCK_DIR")
    if configured:
        return configured
    return os.path.join(tempfile.gettempdir(), "pft_deep_llm_slots")


def _cleanup_stale_llm_slots(lock_dir, stale_after):
    now = time.time()
    try:
        names = os.listdir(lock_dir)
    except FileNotFoundError:
        return
    for name in names:
        if not name.endswith(".lock"):
            continue
        path = os.path.join(lock_dir, name)
        try:
            owner = _read_llm_slot_owner(path)
            owner_pid = _parse_llm_slot_pid(owner)
            if (
                (owner_pid is not None and not _pid_is_alive(owner_pid))
                or now - os.path.getmtime(path) > stale_after
            ):
                os.unlink(path)
        except OSError:
            pass


def _read_llm_slot_owner(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _parse_llm_slot_pid(owner):
    try:
        return int(str(owner).split(":", 1)[0])
    except (TypeError, ValueError):
        return None


def _pid_is_alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        process = kernel32.OpenProcess(0x1000, False, pid)
        if process:
            kernel32.CloseHandle(process)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _global_llm_slot():
    """Limit concurrent LLM HTTP calls across batch worker processes."""
    max_slots = _positive_int_env("LLM_GLOBAL_MAX_CONCURRENCY", 2)
    if max_slots <= 0:
        yield
        return

    lock_dir = _llm_slot_dir()
    os.makedirs(lock_dir, exist_ok=True)
    wait_timeout = _positive_float_env("LLM_SLOT_WAIT_TIMEOUT", 7200)
    stale_after = _positive_float_env("LLM_SLOT_STALE_SECONDS", 7200)
    owner = f"{os.getpid()}:{threading.get_ident()}:{random.getrandbits(32):08x}"
    started = time.time()
    acquired = None
    next_cleanup = 0
    next_notice = started + 30

    while acquired is None:
        now = time.time()
        if now >= next_cleanup:
            _cleanup_stale_llm_slots(lock_dir, stale_after)
            next_cleanup = now + 30

        for slot in range(max_slots):
            candidate = os.path.join(lock_dir, f"slot-{slot}.lock")
            try:
                fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(owner)
            acquired = candidate
            break

        if acquired is None:
            if time.time() - started > wait_timeout:
                raise TimeoutError(
                    f"Timed out waiting for LLM concurrency slot after {wait_timeout:.0f}s"
                )
            if now >= next_notice:
                print(
                    f"  [LLM] waiting for global slot "
                    f"({max_slots} concurrent calls allowed)...",
                    flush=True,
                )
                next_notice = now + 60
            time.sleep(0.5 + random.uniform(0, 0.5))

    try:
        yield
    finally:
        try:
            os.unlink(acquired)
        except OSError:
            pass

# ── Token usage tracking ──────────────────────────────────────────────


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0


_usage_accumulator = UsageRecord()
_usage_lock = threading.Lock()


def _accumulate_usage(prompt_tokens: int, completion_tokens: int) -> None:
    with _usage_lock:
        _usage_accumulator.prompt_tokens += prompt_tokens
        _usage_accumulator.completion_tokens += completion_tokens


def get_usage_summary() -> dict:
    """Return accumulated usage snapshot as a dict."""
    with _usage_lock:
        pt = _usage_accumulator.prompt_tokens
        ct = _usage_accumulator.completion_tokens
        return {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        }


def reset_usage() -> None:
    with _usage_lock:
        _usage_accumulator.prompt_tokens = 0
        _usage_accumulator.completion_tokens = 0


def load_cache():
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except:
        logger.warning(f"Failed to load cache.")
    return {}


def save_cache(cache):
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except:
        logger.warning(f"Failed to save cache")


def _telemetry_path():
    configured = os.getenv("LLM_TELEMETRY_FILE")
    if configured:
        return configured
    return os.path.join(
        log_directory, f"llm_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
    )


def _telemetry_enabled():
    value = os.getenv("LLM_TELEMETRY", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _model_for_provider(provider):
    if provider == "GEMINI":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    if provider:
        return os.getenv(f"{provider}_MODEL")
    return None


def _cached_response(prompt):
    with _cache_lock:
        cache = load_cache()
        return cache.get(prompt)


def _save_cached_response(prompt, response_text):
    with _cache_lock:
        cache = load_cache()
        cache[prompt] = response_text
        save_cache(cache)


def _dispatch_llm_call(prompt, provider):
    if provider == "GEMINI":
        return _call_llm_gemini(prompt)
    return _call_llm_provider(prompt)


def _record_llm_telemetry(
    *,
    stage,
    prompt,
    provider,
    model,
    started_at,
    duration_sec,
    cache_hit,
    success,
    metadata=None,
    error=None,
    prompt_tokens=None,
    completion_tokens=None,
):
    if not _telemetry_enabled():
        return

    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "stage": stage or "unspecified",
        "provider": provider,
        "model": model,
        "prompt_chars": len(prompt),
        "duration_sec": round(duration_sec, 3),
        "cache_hit": bool(cache_hit),
        "success": bool(success),
        "metadata": metadata or {},
    }
    if prompt_tokens is not None:
        event["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        event["completion_tokens"] = completion_tokens
    if error:
        event["error"] = str(error)

    try:
        path = _telemetry_path()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with _telemetry_lock:
            with open(path, "a", encoding="utf-8") as handle:
                json.dump(event, handle, ensure_ascii=False)
                handle.write("\n")
    except Exception as exc:
        logger.warning(f"Failed to write LLM telemetry: {exc}")


def _record_call_telemetry(
    stage,
    prompt,
    provider,
    model,
    started_at,
    start_time,
    cache_hit,
    success,
    metadata,
    error=None,
    prompt_tokens=None,
    completion_tokens=None,
):
    _record_llm_telemetry(
        stage=stage,
        prompt=prompt,
        provider=provider,
        model=model,
        started_at=started_at,
        duration_sec=time.perf_counter() - start_time,
        cache_hit=cache_hit,
        success=success,
        metadata=metadata,
        error=error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _call_context(prompt, stage, metadata):
    return {
        "prompt": prompt,
        "stage": stage,
        "metadata": metadata,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "start_time": time.perf_counter(),
    }


def _record_context_telemetry(context, provider, model, cache_hit, success, error=None, prompt_tokens=None, completion_tokens=None):
    _record_call_telemetry(
        context["stage"],
        context["prompt"],
        provider,
        model,
        context["started_at"],
        context["start_time"],
        cache_hit,
        success,
        context["metadata"],
        error=error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def get_llm_provider():
    provider = os.getenv("LLM_PROVIDER")
    if not provider and (os.getenv("GEMINI_PROJECT_ID") or os.getenv("GEMINI_API_KEY")):
        provider = "GEMINI"
    # if necessary, add ANTHROPIC/OPENAI
    return provider


def _call_llm_provider(prompt: str) -> tuple[str, int, int]:
    """
    Call an LLM provider based on environment variables.
    Environment variables:
    - LLM_PROVIDER: "OLLAMA" or "XAI"
    - <provider>_MODEL: Model name (e.g., OLLAMA_MODEL, XAI_MODEL)
    - <provider>_BASE_URL: Base URL without endpoint (e.g., OLLAMA_BASE_URL, XAI_BASE_URL)
    - <provider>_API_KEY: API key (e.g., OLLAMA_API_KEY, XAI_API_KEY; optional for providers that don't require it)
    The endpoint /v1/chat/completions will be appended to the base URL.
    """
    logger.info(f"PROMPT: {prompt}") # log the prompt

    # Read the provider from environment variable
    provider = os.environ.get("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM_PROVIDER environment variable is required")

    # Construct the names of the other environment variables
    model_var = f"{provider}_MODEL"
    base_url_var = f"{provider}_BASE_URL"
    api_key_var = f"{provider}_API_KEY"

    # Read the provider-specific variables
    model = os.environ.get(model_var)
    base_url = os.environ.get(base_url_var)
    api_key = os.environ.get(api_key_var, "")  # API key is optional, default to empty string

    # Validate required variables
    if not model:
        raise ValueError(f"{model_var} environment variable is required")
    if not base_url:
        raise ValueError(f"{base_url_var} environment variable is required")

    # Append the API endpoint path; configurable via LLM_API_ENDPOINT env var
    # Default: /v1/chat/completions. For OpenAI Responses API, set: /v1/responses
    endpoint = os.getenv("LLM_API_ENDPOINT", "/v1/chat/completions")
    url = f"{base_url.rstrip('/')}{endpoint}"

    # Detect Responses API format (OpenAI /v1/responses) vs Chat Completions
    is_responses_api = "/responses" in endpoint

    # Configure headers
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:  # Only add Authorization header if API key is provided
        headers["Authorization"] = f"Bearer {api_key}"

    # ── Build payload ──────────────────────────────────────────────────
    if is_responses_api:
        payload: dict = {
            "model": model,
            "input": prompt,
        }
        # Reasoning effort for OpenAI reasoning models
        reasoning_effort_env = os.getenv(f"{provider}_REASONING_EFFORT")
        if reasoning_effort_env:
            payload["reasoning"] = {"effort": reasoning_effort_env}
    else:
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        # DeepSeek V4 enables thinking mode by default; pass explicit params for clarity & control
        if provider == "DEEPSEEK":
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = "high"
            # temperature/top_p/presence_penalty/frequency_penalty are ignored in thinking mode

    timeout = float(os.getenv("LLM_HTTP_TIMEOUT", "3600"))

    try:
        with _global_llm_slot():
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)

        raw_text = response.text
        try:
            response_json = response.json()
            logger.info("RESPONSE:\n%s", json.dumps(response_json, indent=2))
        except ValueError as exc:
            if response.status_code >= 400:
                raise Exception(
                    f"HTTP error occurred: {response.status_code} {response.reason} "
                    f"for url: {url} (Body: {raw_text[:500]})"
                ) from exc
            raise Exception(
                f"Failed to parse response as JSON from {provider}. "
                f"Body: {raw_text[:500]}"
            ) from exc

        response.raise_for_status()

        if is_responses_api:
            # Parse Responses API format: output[type="message"].content[type="output_text"].text
            text_parts = []
            for item in response_json.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            text_parts.append(content.get("text", ""))
            text = "".join(text_parts)
            usage = response_json.get("usage", {})
            pt = usage.get("input_tokens", 0) or 0
            ct = usage.get("output_tokens", 0) or 0
        else:
            # Parse Chat Completions format
            usage = response_json.get("usage", {})
            pt = usage.get("prompt_tokens", 0) or 0
            ct = usage.get("completion_tokens", 0) or 0
            text = response_json["choices"][0]["message"]["content"]

        return text, pt, ct
    except requests.exceptions.HTTPError as e:
        error_message = f"HTTP error occurred: {e}"
        try:
            error_details = response.json().get("error", "No additional details")
            error_message += f" (Details: {error_details})"
        except:
            pass
        raise Exception(error_message)
    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to {provider} API. Check your network connection.")
    except requests.exceptions.Timeout:
        raise Exception(f"Request to {provider} API timed out.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred while making the request to {provider}: {e}")
    except ValueError:
        raise Exception(f"Failed to parse response as JSON from {provider}. The server might have returned an invalid response.")

def _is_transient_llm_error(exc: Exception) -> bool:
    """Check if an exception from LLM call is transient (retryable)."""
    msg = str(exc).lower()
    transient_markers = [
        "timeout", "timed out", "time-out",
        "connection", "connection refused", "connection reset",
        "server error", "service unavailable",
        "503", "502", "504",
        "429", "too many requests", "concurrency limit exceeded",
        "rate limit", "too many requests", "429",
        "overloaded", "temporary",
        "remote end closed", "incomplete read",
        "expecting value",       # Empty/broken JSON response
        "parse response as json", # Empty/broken JSON response
        "failed to parse",        # JSON parse failure
        "bad request",           # May indicate upstream transient issue
        "llm concurrency slot",
    ]
    for marker in transient_markers:
        if marker in msg:
            return True
    return False


# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
def call_llm(
    prompt: str,
    use_cache: bool = True,
    stage: str = None,
    metadata: dict = None,
) -> str:
    context = _call_context(prompt, stage, metadata)
    provider = None
    model = None

    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    try:
        provider = get_llm_provider()
        model = _model_for_provider(provider)

        # Check cache if enabled
        if use_cache:
            response_text = _cached_response(prompt)
            if response_text is not None:
                logger.info(f"RESPONSE: {response_text}")
                _record_context_telemetry(context, provider, model, True, True)
                return response_text

        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                dispatch_result = _dispatch_llm_call(prompt, provider)
                if isinstance(dispatch_result, tuple):
                    response_text, pt, ct = dispatch_result
                else:
                    response_text, pt, ct = dispatch_result, 0, 0

                # Log the response
                logger.info(f"RESPONSE: {response_text}")

                # The lock keeps concurrent batch workers from overwriting cache updates.
                if use_cache:
                    _save_cached_response(prompt, response_text)

                _accumulate_usage(pt, ct)
                _record_context_telemetry(context, provider, model, False, True,
                                          prompt_tokens=pt, completion_tokens=ct)
                return response_text

            except Exception as exc:
                last_error = exc
                if attempt < max_retries and _is_transient_llm_error(exc):
                    delay = min(10 * (2 ** (attempt - 1)), 120)  # 10s, 20s, 40s, 80s, cap 120s
                    jitter = random.uniform(0, delay * 0.3)
                    wait = delay + jitter
                    print(
                        f"  [LLM:{context['stage'] or 'unspecified'}] "
                        f"attempt {attempt}/{max_retries} failed; retrying in {wait:.1f}s: {exc}",
                        flush=True,
                    )
                    logger.warning(
                        f"LLM call attempt {attempt}/{max_retries} failed (transient), "
                        f"retrying in {wait:.1f}s: {exc}"
                    )
                    time.sleep(wait)
                else:
                    raise

        # All retries exhausted
        raise last_error

    except Exception as exc:
        _record_context_telemetry(context, provider, model, False, False, error=exc)
        raise


def _call_llm_gemini(prompt: str) -> tuple[str, int, int]:
    if os.getenv("GEMINI_PROJECT_ID"):
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GEMINI_PROJECT_ID"),
            location=os.getenv("GEMINI_LOCATION", "us-central1")
        )
    elif os.getenv("GEMINI_API_KEY"):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError("Either GEMINI_PROJECT_ID or GEMINI_API_KEY must be set in the environment")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    response = client.models.generate_content(
        model=model,
        contents=[prompt]
    )
    pt = ct = 0
    if response.usage_metadata:
        pt = response.usage_metadata.prompt_token_count or 0
        ct = response.usage_metadata.candidates_token_count or 0
    return response.text, pt, ct

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
