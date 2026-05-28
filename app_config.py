from __future__ import annotations

from typing import Iterable

DEFAULT_INCLUDE_PATTERNS = {
    "*.py",
    "*.js",
    "*.jsx",
    "*.ts",
    "*.tsx",
    "*.go",
    "*.java",
    "*.pyi",
    "*.pyx",
    "*.c",
    "*.cc",
    "*.cpp",
    "*.h",
    "*.md",
    "*.rst",
    "*Dockerfile",
    "*Makefile",
    "*.yaml",
    "*.yml",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "assets/*",
    "data/*",
    "images/*",
    "public/*",
    "static/*",
    "temp/*",
    "*docs/*",
    "*venv/*",
    "*.venv/*",
    "*test*",
    "*tests/*",
    "*examples/*",
    "v1/*",
    "*dist/*",
    "*build/*",
    "*experimental/*",
    "*deprecated/*",
    "*misc/*",
    "*legacy/*",
    ".git/*",
    ".github/*",
    ".next/*",
    ".vscode/*",
    "*obj/*",
    "*bin/*",
    "*node_modules/*",
    "*.log",
}

DEFAULT_TUTORIAL_LANGUAGE = "Chinese"
DEFAULT_MAX_FILE_SIZE = 1048576
DEFAULT_MAX_ABSTRACTIONS = 10
DEFAULT_MAX_ABSTRACTIONS_MODE = "auto"
DEFAULT_OUTPUT_DIR = "pf_guide"


def normalize_patterns(
    patterns: Iterable[str] | None,
    defaults: set[str],
) -> set[str]:
    if patterns is None:
        return set(defaults)

    cleaned = {item.strip() for item in patterns if item and item.strip()}
    return cleaned or set(defaults)


def build_shared_state(
    *,
    repo_url: str | None,
    local_dir: str | None,
    project_name: str | None,
    github_token: str | None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    language: str = DEFAULT_TUTORIAL_LANGUAGE,
    use_cache: bool = True,
    max_abstraction_num: int | str = DEFAULT_MAX_ABSTRACTIONS_MODE,
    max_extraction_batches: int | None = None,
    llm_extraction_concurrency: int | None = None,
) -> dict:
    return {
        "repo_url": repo_url,
        "local_dir": local_dir,
        "project_name": project_name,
        "github_token": github_token,
        "output_dir": output_dir,
        "include_patterns": normalize_patterns(
            include_patterns,
            DEFAULT_INCLUDE_PATTERNS,
        ),
        "exclude_patterns": normalize_patterns(
            exclude_patterns,
            DEFAULT_EXCLUDE_PATTERNS,
        ),
        "max_file_size": max_file_size,
        "language": language,
        "use_cache": use_cache,
        "max_abstraction_num": max_abstraction_num,
        "max_extraction_batches": max_extraction_batches,
        "llm_extraction_concurrency": llm_extraction_concurrency,
        "files": [],
        "abstractions": [],
        "relationships": {},
        "chapter_order": [],
        "chapters": [],
        "final_output_dir": None,
    }
