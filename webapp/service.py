from __future__ import annotations

import threading
import traceback
import uuid

import dotenv

dotenv.load_dotenv()
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from app_config import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_MAX_ABSTRACTIONS,
    DEFAULT_MAX_ABSTRACTIONS_MODE,
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TUTORIAL_LANGUAGE,
    build_shared_state,
)
from webapp.folder_picker import pick_windows_folder


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_patterns(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = value
    else:
        text = str(value).replace(",", "\n")
        parts = text.splitlines()
    cleaned = [item.strip() for item in parts if str(item).strip()]
    return cleaned or None


def _parse_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _parse_positive_int(value, default: int | None) -> int | None:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"参数必须是整数: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"参数必须是正整数: {value}")
    return parsed


def _parse_max_abstraction_num(value) -> int | str:
    if value in (None, ""):
        return DEFAULT_MAX_ABSTRACTIONS_MODE
    if isinstance(value, str) and value.strip().lower() == DEFAULT_MAX_ABSTRACTIONS_MODE:
        return DEFAULT_MAX_ABSTRACTIONS_MODE
    parsed = _parse_positive_int(value, DEFAULT_MAX_ABSTRACTIONS)
    return parsed or DEFAULT_MAX_ABSTRACTIONS


def _normalize_directory(path_text: str) -> str:
    if not path_text.strip():
        raise ValueError("请先选择一个本地仓库目录")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise ValueError(f"目录不存在: {path}")
    if not path.is_dir():
        raise ValueError(f"不是目录: {path}")
    return str(path.resolve())


@dataclass
class TaskConfig:
    source_path: str
    project_name: str | None
    output_dir: str
    language: str
    max_file_size: int
    max_abstraction_num: int | str
    use_cache: bool
    include_patterns: list[str] | None
    exclude_patterns: list[str] | None
    max_extraction_batches: int | None
    llm_extraction_concurrency: int | None


@dataclass
class ProgressInfo:
    current: int = 0
    total: int = 6
    name: str = ""

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return int(self.current * 100 / self.total)

    def to_dict(self) -> dict:
        return {"current": self.current, "total": self.total, "name": self.name, "percent": self.percent}


@dataclass
class TaskRecord:
    id: str
    config: TaskConfig
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    result_dir: str | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    progress: ProgressInfo = field(default_factory=ProgressInfo)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def append_log(self, line: str) -> None:
        cleaned = line.rstrip()
        if not cleaned:
            return
        with self._lock:
            self.logs.append(cleaned)
            if len(self.logs) > 200:
                self.logs[:] = self.logs[-200:]

    def update_progress(self, current: int, total: int, name: str) -> None:
        with self._lock:
            self.progress.current = current
            self.progress.total = total
            self.progress.name = name

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "result_dir": self.result_dir,
                "error": self.error,
                "logs": list(self.logs),
                "progress": self.progress.to_dict(),
                "config": asdict(self.config),
            }


class _TaskLogWriter:
    def __init__(self, task: TaskRecord):
        self.task = task
        self._buffer = ""

    def write(self, chunk: str) -> int:
        if not chunk:
            return 0
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.task.append_log(line)
        return len(chunk)

    def flush(self) -> None:
        if self._buffer:
            self.task.append_log(self._buffer)
            self._buffer = ""


class TaskQueueService:
    def __init__(
        self,
        *,
        flow_factory: Callable | None = None,
        task_runner: Callable[[TaskRecord, Callable], str | None] | None = None,
    ) -> None:
        self._flow_factory = flow_factory or _default_flow_factory
        self._task_runner = task_runner or self._run_task
        self._tasks: list[TaskRecord] = []
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._auto_run = False
        self._active_task_id: str | None = None

    def defaults(self) -> dict:
        return {
            "language": DEFAULT_TUTORIAL_LANGUAGE,
            "output_dir": DEFAULT_OUTPUT_DIR,
            "max_file_size": DEFAULT_MAX_FILE_SIZE,
            "max_abstraction_num": DEFAULT_MAX_ABSTRACTIONS_MODE,
            "include_patterns": "\n".join(sorted(DEFAULT_INCLUDE_PATTERNS)),
            "exclude_patterns": "\n".join(sorted(DEFAULT_EXCLUDE_PATTERNS)),
        }

    def snapshot(self) -> dict:
        with self._lock:
            tasks = [task.to_dict() for task in self._tasks]
            pending_count = sum(1 for task in self._tasks if task.status == "pending")
            completed_count = sum(1 for task in self._tasks if task.status == "completed")
            failed_count = sum(1 for task in self._tasks if task.status == "failed")
            return {
                "auto_run": self._auto_run,
                "active_task_id": self._active_task_id,
                "pending_count": pending_count,
                "completed_count": completed_count,
                "failed_count": failed_count,
                "tasks": tasks,
            }

    def pick_folder(self) -> dict:
        normalized = _normalize_directory(pick_windows_folder())
        return {
            "selected_path": normalized,
            "output_dir": str(Path(normalized) / DEFAULT_OUTPUT_DIR),
        }

    def add_task(self, payload: dict) -> dict:
        config = self._parse_task_config(payload)
        task = TaskRecord(id=uuid.uuid4().hex[:8], config=config)
        task.append_log(f"[{_now_iso()}] 已创建任务，等待执行。")

        with self._lock:
            self._tasks.append(task)
            if self._auto_run:
                self._ensure_worker_locked()

        return task.to_dict()

    def delete_task(self, task_id: str) -> dict:
        with self._lock:
            for index, task in enumerate(self._tasks):
                if task.id != task_id:
                    continue
                if task.status == "running":
                    raise ValueError("运行中的任务不能删除")
                deleted = self._tasks.pop(index)
                return deleted.to_dict()
        raise ValueError(f"任务不存在: {task_id}")

    def start_queue(self) -> dict:
        with self._lock:
            self._auto_run = True
            self._ensure_worker_locked()
        return self.snapshot()

    def _ensure_worker_locked(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if not self._auto_run:
                    self._active_task_id = None
                    return
                task = next((item for item in self._tasks if item.status == "pending"), None)
                if task is None:
                    self._active_task_id = None
                    return
                task.status = "running"
                task.started_at = _now_iso()
                task.append_log(f"[{task.started_at}] 开始执行。")
                self._active_task_id = task.id

            try:
                result_dir = self._task_runner(task, self._flow_factory)
            except Exception as exc:  # noqa: BLE001
                error_text = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                with self._lock:
                    task.status = "failed"
                    task.finished_at = _now_iso()
                    task.error = error_text
                    task.append_log(error_text)
            else:
                with self._lock:
                    task.status = "completed"
                    task.finished_at = _now_iso()
                    task.result_dir = result_dir
                    task.append_log(
                        f"[{task.finished_at}] 执行完成。输出目录: {result_dir or '未返回'}"
                    )
            finally:
                with self._lock:
                    if self._active_task_id == task.id:
                        self._active_task_id = None

    def _run_task(self, task: TaskRecord, flow_factory: Callable) -> str | None:
        writer = _TaskLogWriter(task)
        shared = build_shared_state(
            repo_url=None,
            local_dir=task.config.source_path,
            project_name=task.config.project_name,
            github_token=None,
            output_dir=task.config.output_dir,
            include_patterns=task.config.include_patterns,
            exclude_patterns=task.config.exclude_patterns,
            max_file_size=task.config.max_file_size,
            language=task.config.language,
            use_cache=task.config.use_cache,
            max_abstraction_num=task.config.max_abstraction_num,
            max_extraction_batches=task.config.max_extraction_batches,
            llm_extraction_concurrency=task.config.llm_extraction_concurrency,
        )

        with redirect_stdout(writer), redirect_stderr(writer):
            flow = flow_factory()

            # 遍历线性节点链，统计阶段总数
            stages = []
            curr = flow.start_node
            while curr:
                stages.append(curr)
                curr = curr.successors.get("default")
            total = len(stages)
            task.update_progress(0, total, "准备")

            # 包装每个节点的 _run 以报告进度
            for i, node in enumerate(stages):
                original_run = node._run
                stage_name = type(node).__name__

                def make_wrapper(idx, nd, orig, name, total_stages):
                    def wrapped(shared):
                        task.update_progress(idx, total_stages, f"▶ {name}")
                        task.append_log(f"[{_now_iso()}] ▶ 阶段 {idx + 1}/{total_stages}：{name}")
                        result = orig(shared)
                        task.update_progress(idx + 1, total_stages, f"✓ {name}")
                        return result
                    return wrapped

                node._run = make_wrapper(i, node, original_run, stage_name, total)

            flow.run(shared)
            from utils.call_llm import get_usage_summary  # noqa: PLC0415
            usage = get_usage_summary()
            print(
                f"Token: {usage['prompt_tokens']:,} 输入"
                f" + {usage['completion_tokens']:,} 输出"
                f" = {usage['total_tokens']:,} 总计"
            )

        writer.flush()
        final_output_dir = shared.get("final_output_dir")
        return str(final_output_dir) if final_output_dir else None

    def _parse_task_config(self, payload: dict) -> TaskConfig:
        source_path = _normalize_directory(str(payload.get("source_path", "")).strip())
        project_name = str(payload.get("project_name", "")).strip() or None
        raw_output_dir = str(payload.get("output_dir", "")).strip()
        output_dir = raw_output_dir or str(Path(source_path) / DEFAULT_OUTPUT_DIR)
        language = str(payload.get("language", DEFAULT_TUTORIAL_LANGUAGE)).strip() or DEFAULT_TUTORIAL_LANGUAGE

        return TaskConfig(
            source_path=source_path,
            project_name=project_name,
            output_dir=output_dir,
            language=language,
            max_file_size=_parse_positive_int(
                payload.get("max_file_size"),
                DEFAULT_MAX_FILE_SIZE,
            )
            or DEFAULT_MAX_FILE_SIZE,
            max_abstraction_num=_parse_max_abstraction_num(
                payload.get("max_abstraction_num")
            ),
            use_cache=_parse_bool(payload.get("use_cache"), True),
            include_patterns=_parse_patterns(payload.get("include_patterns")),
            exclude_patterns=_parse_patterns(payload.get("exclude_patterns")),
            max_extraction_batches=_parse_positive_int(
                payload.get("max_extraction_batches"),
                None,
            ),
            llm_extraction_concurrency=_parse_positive_int(
                payload.get("llm_extraction_concurrency"),
                None,
            ),
        )


def _default_flow_factory():
    from flow import create_tutorial_flow

    return create_tutorial_flow()
