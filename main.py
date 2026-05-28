import argparse
import os
import time

import dotenv

from app_config import (
    DEFAULT_MAX_ABSTRACTIONS,
    DEFAULT_MAX_ABSTRACTIONS_MODE,
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TUTORIAL_LANGUAGE,
    build_shared_state,
)
from flow import create_tutorial_flow
from utils.call_llm import get_usage_summary

dotenv.load_dotenv()


def _parse_max_abstractions(value: str) -> int | str:
    raw = str(value).strip()
    if not raw or raw.lower() == DEFAULT_MAX_ABSTRACTIONS_MODE:
        return DEFAULT_MAX_ABSTRACTIONS_MODE
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max-abstractions must be 'auto' or a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "max-abstractions must be a positive integer"
        )
    return parsed

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Generate a tutorial for a GitHub codebase or local directory.")

    # Create mutually exclusive group for source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--repo", help="URL of the public GitHub repository.")
    source_group.add_argument("--dir", help="Path to local directory.")

    parser.add_argument("-n", "--name", help="Project name (optional, derived from repo/directory if omitted).")
    parser.add_argument("-t", "--token", help="GitHub personal access token (optional, reads from GITHUB_TOKEN env var if not provided).")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory (default: ./pf_guide).")
    parser.add_argument("-i", "--include", nargs="+", help="Include file patterns (e.g. '*.py' '*.js'). Defaults to common code files if not specified.")
    parser.add_argument("-e", "--exclude", nargs="+", help="Exclude file patterns (e.g. 'tests/*' 'docs/*'). Defaults to test/build directories if not specified.")
    parser.add_argument("-s", "--max-size", type=int, default=DEFAULT_MAX_FILE_SIZE, help="Maximum file size in bytes (default: 1048576, about 1MB).")
    # Add language parameter for multi-language support
    parser.add_argument(
        "--language",
        default=DEFAULT_TUTORIAL_LANGUAGE,
        help=f"Language for the generated tutorial (default: {DEFAULT_TUTORIAL_LANGUAGE})",
    )
    # Add use_cache parameter to control LLM caching
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching (default: caching enabled)")
    # Add max_abstraction_num parameter to control the number of abstractions
    parser.add_argument(
        "--max-abstractions",
        type=_parse_max_abstractions,
        default=DEFAULT_MAX_ABSTRACTIONS_MODE,
        help=(
            "Maximum number of abstractions to identify, or 'auto' to let the "
            f"LLM estimate a suitable chapter count (default: {DEFAULT_MAX_ABSTRACTIONS_MODE})"
        ),
    )
    parser.add_argument(
        "--max-extraction-batches",
        type=int,
        default=None,
        help="Maximum LLM extraction batches during abstraction analysis (default: env LLM_MAX_EXTRACTION_BATCHES or 40)",
    )
    parser.add_argument(
        "--llm-extraction-concurrency",
        type=int,
        default=None,
        help="Concurrent LLM workers for abstraction extraction (default: env LLM_EXTRACTION_CONCURRENCY or 1)",
    )

    args = parser.parse_args()

    # Get GitHub token from argument or environment variable if using repo
    github_token = None
    if args.repo:
        github_token = args.token or os.environ.get('GITHUB_TOKEN')
        if not github_token:
            print("Warning: No GitHub token provided. You might hit rate limits for public repositories.")

    shared = build_shared_state(
        repo_url=args.repo,
        local_dir=args.dir,
        project_name=args.name,
        github_token=github_token,
        output_dir=args.output,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_file_size=args.max_size,
        language=args.language,
        use_cache=not args.no_cache,
        max_abstraction_num=args.max_abstractions,
        max_extraction_batches=args.max_extraction_batches,
        llm_extraction_concurrency=args.llm_extraction_concurrency,
    )

    # Display starting message with repository/directory and language
    print(f"Starting tutorial generation for: {args.repo or args.dir} in {args.language.capitalize()} language")
    print(f"LLM caching: {'Disabled' if args.no_cache else 'Enabled'}")

    # Create the flow instance
    tutorial_flow = create_tutorial_flow()

    # ── capture node chain & wrap with terminal progress ──
    if hasattr(tutorial_flow, "start_node") and tutorial_flow.start_node:
        stages = []
        curr = tutorial_flow.start_node
        while curr:
            stages.append(curr)
            curr = curr.successors.get("default")

        total = len(stages)
        max_w = max(len(type(n).__name__) for n in stages)

        for i, node in enumerate(stages):
            orig_run = node._run
            name = type(node).__name__
            label = name.ljust(max_w)

            def make_wrapper(idx, nd, orig, lbl, total_stages):
                def wrapped(shared):
                    t0 = time.time()
                    if total_stages:
                        fraction = idx / total_stages
                        bar_len = 20
                        filled = int(bar_len * fraction)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        pct = int(fraction * 100)
                    else:
                        bar, pct = "#" * 20, 100
                    print(f"\r  [{bar}] {pct:>3}%  [{idx}/{total_stages}] {lbl}  RUN ", end="", flush=True)
                    result = orig(shared)
                    elapsed = time.time() - t0
                    print(f"\r  [{bar}] {pct:>3}%  [{idx}/{total_stages}] {lbl}  OK   {elapsed:.1f}s")
                    return result
                return wrapped

            node._run = make_wrapper(i, node, orig_run, label, total)

    # Run the flow
    t_start = time.time()
    tutorial_flow.run(shared)
    t_elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {t_elapsed:.1f}s")

    # ── Token usage summary ──
    usage = get_usage_summary()
    print(
        f"  Tokens: {usage['prompt_tokens']:,} prompt + "
        f"{usage['completion_tokens']:,} completion = {usage['total_tokens']:,} total"
    )

if __name__ == "__main__":
    main()
