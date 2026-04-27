import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_PROMPT_CHARS = 12000
MAX_FALLBACK_CHARS = 3000
MAX_FALLBACK_LINES = 80
MAX_MISC_CHARS = 2200
MAX_MISC_LINES = 80


def normalize_path(path):
    return path.replace("\\", "/")


def sort_files_items(files_items):
    return sorted(files_items, key=lambda item: normalize_path(item[0]).lower())


def build_chunk_inventory(files_data):
    adapter_results = run_code_chunk_adapter(files_data)
    results_by_index = {item.get("file_index"): item for item in adapter_results}
    chunks = []

    for file_index, (path, content) in enumerate(files_data):
        result = results_by_index.get(file_index, {})
        mapped_chunks = []
        if not result.get("error") and result.get("chunks"):
            mapped_chunks = map_code_chunk_result(result)
            chunks.extend(mapped_chunks)

        if mapped_chunks:
            ranges = [chunk["line_range"] for chunk in mapped_chunks]
            chunks.extend(build_misc_chunks(file_index, path, content, ranges))
        else:
            chunks.extend(build_fallback_chunks(file_index, path, content))

    return chunks


def run_code_chunk_adapter(files_data):
    node_path = shutil.which("node")
    if not node_path:
        return [_adapter_error(i, path, "Node.js is not available") for i, (path, _) in enumerate(files_data)]

    adapter_path = Path(__file__).resolve().parents[1] / "tools" / "code_chunk_adapter.mjs"
    payload = {
        "files": [
            {"file_index": i, "filepath": path, "code": content}
            for i, (path, content) in enumerate(files_data)
        ]
    }

    try:
        proc = subprocess.run(
            [node_path, str(adapter_path)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        message = f"Unable to run code-chunk adapter: {exc}"
        return [_adapter_error(i, path, message) for i, (path, _) in enumerate(files_data)]

    if proc.returncode != 0:
        message = _adapter_failure_message(proc)
        return [_adapter_error(i, path, message) for i, (path, _) in enumerate(files_data)]

    try:
        decoded = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        message = f"code-chunk adapter returned invalid JSON: {exc}"
        return [_adapter_error(i, path, message) for i, (path, _) in enumerate(files_data)]

    if decoded.get("error"):
        message = decoded["error"]
        return [_adapter_error(i, path, message) for i, (path, _) in enumerate(files_data)]
    return decoded.get("results", [])


def map_code_chunk_result(result):
    file_index = result["file_index"]
    filepath = result["filepath"]
    mapped = []
    for chunk in result.get("chunks", []):
        line_range = _to_one_based_range(chunk.get("line_range"))
        chunk_kind = chunk.get("chunk_kind") or "entity"
        mapped.append(
            {
                "chunk_id": _chunk_id(file_index, filepath, chunk_kind, chunk.get("index", len(mapped))),
                "file_index": file_index,
                "filepath": filepath,
                "language": chunk.get("language") or "",
                "engine": "code-chunk",
                "chunk_kind": chunk_kind,
                "symbol_path": chunk.get("symbol_path") or "<module>",
                "signature": chunk.get("signature") or "",
                "line_range": line_range,
                "content": chunk.get("content") or "",
                "context_text": chunk.get("context_text") or chunk.get("content") or "",
                "parent_scope": chunk.get("parent_scope") or "",
                "related_imports": _string_list(chunk.get("related_imports")),
            }
        )
    return mapped


def build_misc_chunks(file_index, path, content, entity_ranges):
    lines = content.splitlines()
    uncovered = _uncovered_ranges(len(lines), entity_ranges)
    chunks = []
    misc_index = 0
    for start, end in uncovered:
        for sub_start, sub_end in _split_range(lines, start, end, MAX_MISC_LINES, MAX_MISC_CHARS):
            text = "\n".join(lines[sub_start - 1:sub_end]).strip()
            if not _has_signal(text):
                continue
            chunks.append(_local_chunk(file_index, path, "misc", misc_index, sub_start, sub_end, text))
            misc_index += 1
    return chunks


def build_fallback_chunks(file_index, path, content):
    lines = content.splitlines()
    if not lines and content:
        lines = [content]
    chunks = []
    for idx, (start, end) in enumerate(_split_range(lines, 1, len(lines), MAX_FALLBACK_LINES, MAX_FALLBACK_CHARS)):
        text = "\n".join(lines[start - 1:end]).strip()
        if text:
            chunks.append(_local_chunk(file_index, path, "fallback", idx, start, end, text))
    return chunks


def build_chunk_catalog(chunks):
    lines = []
    for chunk in chunks:
        rng = chunk["line_range"]
        location = f"{chunk['file_index']} # {chunk['filepath']}"
        short_context = _shorten(chunk.get("context_text") or chunk["content"], 220)
        lines.append(
            "\n".join(
                [
                    f"- chunk_id: {chunk['chunk_id']}",
                    f"  file: {location}",
                    f"  kind: {chunk['chunk_kind']}",
                    f"  symbol_path: {chunk['symbol_path']}",
                    f"  signature: {chunk.get('signature', '')}",
                    f"  lines: {rng['start']}-{rng['end']}",
                    f"  engine: {chunk['engine']}",
                    f"  short_context: {short_context}",
                ]
            )
        )
    return "\n".join(lines)


def pack_chunks_for_prompt(chunks, chunk_ids, max_chars=DEFAULT_PROMPT_CHARS):
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    batches = []
    current = []
    current_size = 0
    for chunk_id in chunk_ids:
        chunk = by_id.get(chunk_id)
        if not chunk:
            continue
        size = _prompt_size(chunk)
        if current and current_size + size > max_chars:
            batches.append(current)
            current = []
            current_size = 0
        current.append(chunk)
        current_size += size
    if current:
        batches.append(current)
    return batches


def extract_file_indices(chunks, chunk_ids):
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    indices = {by_id[chunk_id]["file_index"] for chunk_id in chunk_ids if chunk_id in by_id}
    return sorted(indices)


def format_chunks_for_prompt(chunks):
    blocks = []
    for chunk in chunks:
        rng = chunk["line_range"]
        header = (
            f"--- Chunk {chunk['chunk_id']} | file {chunk['file_index']} # {chunk['filepath']} "
            f"| {chunk['chunk_kind']} | lines {rng['start']}-{rng['end']} ---"
        )
        context = chunk.get("context_text") or ""
        content = chunk.get("content") or ""
        blocks.append(f"{header}\nContext:\n{context}\n\nCode:\n{content}")
    return "\n\n".join(blocks)


def deterministic_plan_batches(chunks, max_chars=DEFAULT_PROMPT_CHARS):
    batches = []
    current_ids = []
    current_size = 0
    current_name = None
    for chunk in chunks:
        size = _prompt_size(chunk)
        scope = chunk.get("parent_scope") or chunk.get("filepath")
        if current_ids and (current_size + size > max_chars or scope != current_name):
            batches.append(_batch(" / ".join(str(current_name).split("/")[-2:]), current_ids))
            current_ids = []
            current_size = 0
        current_name = scope
        current_ids.append(chunk["chunk_id"])
        current_size += size
    if current_ids:
        batches.append(_batch(" / ".join(str(current_name).split("/")[-2:]), current_ids))
    return batches


def _adapter_error(file_index, path, message):
    return {"file_index": file_index, "filepath": path, "error": message}


def _adapter_failure_message(proc):
    try:
        decoded = json.loads(proc.stdout)
        if decoded.get("error"):
            return decoded["error"]
    except json.JSONDecodeError:
        pass
    stderr = proc.stderr.strip()
    return stderr or "code-chunk adapter failed; run npm install and ensure Node.js is available"


def _to_one_based_range(line_range):
    if not isinstance(line_range, dict):
        return {"start": 1, "end": 1}
    start = int(line_range.get("start", 0)) + 1
    end = int(line_range.get("end", start - 1)) + 1
    return {"start": max(1, start), "end": max(start, end)}


def _chunk_id(file_index, filepath, kind, chunk_index):
    return f"{file_index:05d}:{normalize_path(filepath)}:{kind}:{chunk_index}"


def _local_chunk(file_index, path, kind, index, start, end, text):
    module = Path(path).stem or "<module>"
    return {
        "chunk_id": _chunk_id(file_index, path, kind, index),
        "file_index": file_index,
        "filepath": path,
        "language": _language_from_path(path),
        "engine": "local",
        "chunk_kind": kind,
        "symbol_path": f"{module}#{kind}:{index}" if kind != "misc" else f"{module}#misc:{index}",
        "signature": "",
        "line_range": {"start": start, "end": end},
        "content": text,
        "context_text": f"{kind} chunk from {path}, lines {start}-{end}",
        "parent_scope": module,
        "related_imports": [],
    }


def _uncovered_ranges(line_count, entity_ranges):
    covered = [False] * (line_count + 1)
    for line_range in entity_ranges:
        if not isinstance(line_range, dict):
            continue
        start = max(1, int(line_range.get("start", 1)))
        end = min(line_count, int(line_range.get("end", start)))
        for line_no in range(start, end + 1):
            covered[line_no] = True

    ranges = []
    start = None
    for line_no in range(1, line_count + 1):
        if not covered[line_no] and start is None:
            start = line_no
        if (covered[line_no] or line_no == line_count) and start is not None:
            end = line_no - 1 if covered[line_no] else line_no
            ranges.append((start, end))
            start = None
    return ranges


def _split_range(lines, start, end, max_lines, max_chars):
    if end < start:
        return []
    ranges = []
    cur_start = start
    cur_chars = 0
    for line_no in range(start, end + 1):
        cur_chars += len(lines[line_no - 1]) + 1
        too_many_lines = line_no - cur_start + 1 >= max_lines
        too_many_chars = cur_chars >= max_chars
        if too_many_lines or too_many_chars or line_no == end:
            ranges.append((cur_start, line_no))
            cur_start = line_no + 1
            cur_chars = 0
    return ranges


def _has_signal(text):
    meaningful = [line.strip() for line in text.splitlines() if line.strip()]
    if not meaningful:
        return False
    comment_prefixes = ("#", "//", "/*", "*", "--")
    return any(not line.startswith(comment_prefixes) for line in meaningful)


def _language_from_path(path):
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }.get(suffix, "")


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _shorten(text, limit):
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _prompt_size(chunk):
    return len(chunk.get("context_text") or "") + len(chunk.get("content") or "") + 200


def _batch(name, chunk_ids):
    return {
        "name": name or "Code Chunk Batch",
        "reason": "Deterministic fallback grouping by file and scope.",
        "chunk_ids": chunk_ids,
    }
