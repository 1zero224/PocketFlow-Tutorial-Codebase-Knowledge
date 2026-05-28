import unittest
import json
import os
import sys
import tempfile
import threading
import time
import types
from unittest.mock import patch

fake_pocketflow = types.ModuleType("pocketflow")


class FakeNode:
    def __init__(self, *args, **kwargs):
        self.cur_retry = 0


fake_pocketflow.Node = FakeNode
fake_pocketflow.BatchNode = FakeNode
sys.modules.setdefault("pocketflow", fake_pocketflow)

import nodes
import utils.call_llm as call_llm_module
from utils.semantic_chunks import (
    build_compact_chunk_catalog,
    build_fallback_chunks,
    build_misc_chunks,
    extract_file_indices,
    map_code_chunk_result,
    pack_chunks_for_prompt,
    run_code_chunk_adapter,
    sort_files_items,
)


class SemanticChunkTests(unittest.TestCase):
    def test_sorted_file_indices_are_stable(self):
        files = {
            "src/Zeta.py": "z",
            "README.md": "r",
            "src\\alpha.py": "a",
        }

        sorted_items = sort_files_items(files.items())

        self.assertEqual(
            [path for path, _ in sorted_items],
            ["README.md", "src\\alpha.py", "src/Zeta.py"],
        )

    def test_chunk_schema_maps_code_chunk_output(self):
        result = {
            "file_index": 2,
            "filepath": "src/router.ts",
            "chunks": [
                {
                    "index": 0,
                    "content": "export function registerRoutes() {}",
                    "context_text": "Function registerRoutes",
                    "line_range": {"start": 4, "end": 8},
                    "language": "typescript",
                    "chunk_kind": "entity",
                    "symbol_path": "Router > registerRoutes",
                    "signature": "function registerRoutes(): void",
                    "parent_scope": "Router",
                    "related_imports": ["express"],
                }
            ],
        }

        chunks = map_code_chunk_result(result)

        self.assertEqual(chunks[0]["chunk_id"], "00002:src/router.ts:entity:0")
        self.assertEqual(chunks[0]["file_index"], 2)
        self.assertEqual(chunks[0]["filepath"], "src/router.ts")
        self.assertEqual(chunks[0]["chunk_kind"], "entity")
        self.assertEqual(chunks[0]["symbol_path"], "Router > registerRoutes")
        self.assertEqual(chunks[0]["signature"], "function registerRoutes(): void")
        self.assertEqual(chunks[0]["line_range"], {"start": 5, "end": 9})
        self.assertEqual(chunks[0]["context_text"], "Function registerRoutes")

    def test_misc_chunks_cover_uncovered_glue(self):
        content = "\n".join(
            [
                "import express from 'express'",
                "const router = express.Router()",
                "",
                "function handler(req, res) {",
                "  res.send('ok')",
                "}",
                "",
                "router.get('/health', handler)",
                "export default router",
            ]
        )
        entity_ranges = [{"start": 4, "end": 6}]

        chunks = build_misc_chunks(0, "router.ts", content, entity_ranges)
        combined = "\n".join(chunk["content"] for chunk in chunks)

        self.assertIn("import express", combined)
        self.assertIn("router.get('/health', handler)", combined)
        self.assertTrue(all(chunk["chunk_kind"] == "misc" for chunk in chunks))

    def test_fallback_chunks_preserve_file_index(self):
        chunks = build_fallback_chunks(4, "config.yaml", "server:\n  port: 8080\n")

        self.assertEqual(chunks[0]["file_index"], 4)
        self.assertEqual(chunks[0]["filepath"], "config.yaml")
        self.assertEqual(chunks[0]["chunk_kind"], "fallback")
        self.assertIn("server:", chunks[0]["content"])

    @patch("utils.semantic_chunks.shutil.which", return_value="node")
    @patch("utils.semantic_chunks.subprocess.run")
    def test_run_code_chunk_adapter_decodes_utf8_bytes_on_windows(self, mock_run, _mock_which):
        class Completed:
            returncode = 0
            stdout = json.dumps(
                {
                    "results": [
                        {
                            "file_index": 0,
                            "filepath": "src/中文.py",
                            "chunks": [],
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode("utf-8")
            stderr = b""

        mock_run.return_value = Completed()

        results = run_code_chunk_adapter([("src/中文.py", "print('你好')")])

        self.assertEqual(results[0]["filepath"], "src/中文.py")

    @patch("utils.semantic_chunks.shutil.which", return_value="node")
    @patch("utils.semantic_chunks.subprocess.run")
    def test_run_code_chunk_adapter_handles_missing_stdout_after_decode_failure(self, mock_run, _mock_which):
        class Completed:
            returncode = 1
            stdout = None
            stderr = None

        mock_run.return_value = Completed()

        results = run_code_chunk_adapter([("src/app.py", "print('ok')")])

        self.assertIn("code-chunk adapter failed", results[0]["error"])

    def test_extract_file_indices_from_supporting_chunks(self):
        chunks = [
            {"chunk_id": "a", "file_index": 3},
            {"chunk_id": "b", "file_index": 1},
            {"chunk_id": "c", "file_index": 3},
        ]

        self.assertEqual(extract_file_indices(chunks, ["c", "a", "b"]), [1, 3])

    def test_pack_chunks_for_prompt_respects_budget(self):
        chunks = [
            {
                "chunk_id": f"c{i}",
                "context_text": "",
                "content": "x" * 400,
            }
            for i in range(4)
        ]

        batches = pack_chunks_for_prompt(chunks, ["c0", "c1", "c2", "c3"], max_chars=700)
        flattened = [chunk["chunk_id"] for batch in batches for chunk in batch]

        self.assertEqual(flattened, ["c0", "c1", "c2", "c3"])
        for batch in batches:
            self.assertLessEqual(sum(len(chunk["content"]) + 200 for chunk in batch), 700)

    @patch("nodes.call_llm")
    @patch("nodes.build_chunk_inventory")
    @patch("nodes.crawl_local_files")
    def test_fetch_repo_identify_contract_without_real_llm(
        self,
        mock_crawl_local_files,
        mock_build_chunk_inventory,
        mock_call_llm,
    ):
        mock_crawl_local_files.return_value = {
            "files": {
                "b.py": "def beta():\n    return 2\n",
                "a.py": "def alpha():\n    return 1\n",
            }
        }
        fake_chunks = [
            {
                "chunk_id": "00000:a.py:entity:0",
                "file_index": 0,
                "filepath": "a.py",
                "language": "python",
                "engine": "code-chunk",
                "chunk_kind": "entity",
                "symbol_path": "alpha",
                "signature": "def alpha()",
                "line_range": {"start": 1, "end": 2},
                "content": "def alpha():\n    return 1",
                "context_text": "Function alpha",
                "parent_scope": "",
                "related_imports": [],
            }
        ]
        mock_build_chunk_inventory.return_value = fake_chunks
        mock_call_llm.side_effect = [
            """```yaml
recommended_count: 4
min_count: 3
max_count: 6
reason: |
  Small repo with one core concept.
```""",
            """```yaml
abstractions:
  - name: |
      Alpha Layer
    reason: |
      Alpha function candidate.
    supporting_chunk_ids:
      - "00000:a.py:entity:0"
```""",
            """```yaml
abstractions:
  - name: |
      Alpha Function
    description: |
      Handles alpha behavior.
    file_indices:
      - 0 # a.py
    supporting_chunk_ids:
      - "00000:a.py:entity:0"
```""",
        ]
        shared = {
            "local_dir": "sample",
            "repo_url": None,
            "project_name": "sample",
            "include_patterns": ["*.py"],
            "exclude_patterns": [],
            "max_file_size": 10000,
            "language": "english",
            "use_cache": False,
            "max_abstraction_num": "auto",
        }

        fetch = nodes.FetchRepo()
        fetch.post(shared, None, fetch.exec(fetch.prep(shared)))
        identify = nodes.IdentifyAbstractions()
        identify.post(shared, None, identify.exec(identify.prep(shared)))

        self.assertEqual([path for path, _ in shared["files"]], ["a.py", "b.py"])
        self.assertEqual(
            [call.kwargs["stage"] for call in mock_call_llm.call_args_list],
            ["identify.estimate_budget", "identify.compact_plan", "identify.refine"],
        )
        self.assertEqual(
            shared["abstractions"],
            [
                {
                    "name": "Alpha Function",
                    "description": "Handles alpha behavior.",
                    "files": [0],
                }
            ],
        )
        self.assertNotIn("supporting_chunk_ids", shared["abstractions"][0])

    @patch("nodes.call_llm")
    def test_compact_plan_refines_only_selected_evidence_chunks(self, mock_call_llm):
        chunks = [
            {
                "chunk_id": "c1",
                "file_index": 0,
                "filepath": "models/backbone.py",
                "language": "python",
                "engine": "code-chunk",
                "chunk_kind": "entity",
                "symbol_path": "Backbone",
                "signature": "class Backbone",
                "line_range": {"start": 1, "end": 5},
                "content": "class Backbone:\n    pass",
                "context_text": "Backbone model",
                "parent_scope": "",
                "related_imports": [],
            },
            {
                "chunk_id": "c2",
                "file_index": 1,
                "filepath": "unused/debug.py",
                "language": "python",
                "engine": "code-chunk",
                "chunk_kind": "entity",
                "symbol_path": "DebugTool",
                "signature": "class DebugTool",
                "line_range": {"start": 1, "end": 5},
                "content": "class DebugTool:\n    pass",
                "context_text": "Debug utility",
                "parent_scope": "",
                "related_imports": [],
            },
            {
                "chunk_id": "c3",
                "file_index": 2,
                "filepath": "datasets/loader.py",
                "language": "python",
                "engine": "code-chunk",
                "chunk_kind": "entity",
                "symbol_path": "DatasetLoader",
                "signature": "class DatasetLoader",
                "line_range": {"start": 1, "end": 5},
                "content": "class DatasetLoader:\n    pass",
                "context_text": "Dataset loader",
                "parent_scope": "",
                "related_imports": [],
            },
        ]
        mock_call_llm.side_effect = [
            """```yaml
recommended_count: 5
min_count: 4
max_count: 6
reason: |
  A few distinct layers need separate chapters.
```""",
            """```yaml
abstractions:
  - name: |
      Model Backbone
    reason: |
      Core model code.
    supporting_chunk_ids:
      - c1
      - c3
```""",
            """```yaml
abstractions:
  - name: |
      Model Backbone
    description: |
      Explains how the backbone and dataset loader connect.
    file_indices:
      - 0 # models/backbone.py
      - 2 # datasets/loader.py
    supporting_chunk_ids:
      - c1
      - c3
```""",
        ]
        prep_res = {
            "chunk_inventory": chunks,
            "file_count": 3,
            "project_name": "vision",
            "language": "english",
            "use_cache": False,
            "max_abstraction_num": "auto",
            "max_extraction_batches": 40,
            "llm_extraction_concurrency": 1,
        }

        abstractions = nodes.IdentifyAbstractions().exec(prep_res)

        self.assertEqual(len(mock_call_llm.call_args_list), 3)
        self.assertEqual(
            [call.kwargs["stage"] for call in mock_call_llm.call_args_list],
            ["identify.estimate_budget", "identify.compact_plan", "identify.refine"],
        )
        refinement_prompt = mock_call_llm.call_args_list[2].args[0]
        self.assertIn("class Backbone", refinement_prompt)
        self.assertIn("class DatasetLoader", refinement_prompt)
        self.assertNotIn("class DebugTool", refinement_prompt)
        self.assertEqual(
            abstractions,
            [
                {
                    "name": "Model Backbone",
                    "description": "Explains how the backbone and dataset loader connect.",
                    "files": [0, 2],
                }
            ],
        )

    def test_compact_catalog_excludes_full_code_body(self):
        catalog = build_compact_chunk_catalog(
            [
                {
                    "chunk_id": "c1",
                    "file_index": 0,
                    "filepath": "app.py",
                    "chunk_kind": "entity",
                    "symbol_path": "main",
                    "signature": "def main()",
                    "line_range": {"start": 1, "end": 3},
                    "engine": "code-chunk",
                    "context_text": "Main application entrypoint",
                    "content": "def main():\n    secret_runtime_detail()\n",
                }
            ]
        )

        self.assertIn("def main()", catalog)
        self.assertIn("Main application entrypoint", catalog)
        self.assertNotIn("secret_runtime_detail", catalog)

    @patch("nodes.call_llm")
    def test_large_chunk_inventory_skips_global_planner_llm(self, mock_call_llm):
        mock_call_llm.return_value = "not: [valid"
        chunks = []
        for index in range(500):
            chunks.append(
                {
                    "chunk_id": f"{index:05d}:src/mod_{index}.py:entity:0",
                    "file_index": index,
                    "filepath": f"src/mod_{index}.py",
                    "language": "python",
                    "engine": "code-chunk",
                    "chunk_kind": "entity",
                    "symbol_path": f"symbol_{index}",
                    "signature": f"def symbol_{index}()",
                    "line_range": {"start": 1, "end": 5},
                    "content": "x" * 200,
                    "context_text": "y" * 500,
                    "parent_scope": "",
                    "related_imports": [],
                }
            )

        identify = nodes.IdentifyAbstractions()
        batches = identify._plan_batches("large", chunks, use_cache=False)

        self.assertGreater(len(batches), 1)
        self.assertLessEqual(len(batches), nodes.MAX_LLM_EXTRACTION_BATCHES)
        mock_call_llm.assert_not_called()

    @patch("nodes.build_chunk_inventory")
    def test_identify_prep_accepts_analysis_budget_controls(self, mock_build_chunk_inventory):
        mock_build_chunk_inventory.return_value = [
            {
                "chunk_id": "00000:a.py:entity:0",
                "file_index": 0,
                "filepath": "a.py",
                "language": "python",
                "engine": "code-chunk",
                "chunk_kind": "entity",
                "symbol_path": "alpha",
                "signature": "def alpha()",
                "line_range": {"start": 1, "end": 2},
                "content": "def alpha():\n    return 1",
                "context_text": "Function alpha",
                "parent_scope": "",
                "related_imports": [],
            }
        ]
        shared = {
            "files": [("a.py", "def alpha():\n    return 1\n")],
            "project_name": "sample",
            "language": "english",
            "use_cache": False,
            "max_abstraction_num": 8,
            "max_extraction_batches": 7,
            "llm_extraction_concurrency": 3,
        }

        prep_res = nodes.IdentifyAbstractions().prep(shared)

        self.assertEqual(prep_res["max_extraction_batches"], 7)
        self.assertEqual(prep_res["llm_extraction_concurrency"], 3)

    @patch("nodes.call_llm")
    def test_manual_max_abstractions_skips_auto_budget_stage(self, mock_call_llm):
        chunks = [
            {
                "chunk_id": "c1",
                "file_index": 0,
                "filepath": "core.py",
                "language": "python",
                "engine": "code-chunk",
                "chunk_kind": "entity",
                "symbol_path": "Core",
                "signature": "class Core",
                "line_range": {"start": 1, "end": 5},
                "content": "class Core:\n    pass",
                "context_text": "Core class",
                "parent_scope": "",
                "related_imports": [],
            }
        ]
        mock_call_llm.side_effect = [
            """```yaml
abstractions:
  - name: |
      Core Layer
    reason: |
      Main concept.
    supporting_chunk_ids:
      - c1
```""",
            """```yaml
abstractions:
  - name: |
      Core Layer
    description: |
      Explains the core layer.
    file_indices:
      - 0 # core.py
    supporting_chunk_ids:
      - c1
```""",
        ]
        prep_res = {
            "chunk_inventory": chunks,
            "file_count": 1,
            "project_name": "coreproj",
            "language": "english",
            "use_cache": False,
            "max_abstraction_num": 4,
            "requested_max_abstraction_num": 4,
            "max_extraction_batches": 40,
            "llm_extraction_concurrency": 1,
        }

        abstractions = nodes.IdentifyAbstractions().exec(prep_res)

        self.assertEqual(
            [call.kwargs["stage"] for call in mock_call_llm.call_args_list],
            ["identify.compact_plan", "identify.refine"],
        )
        self.assertEqual(
            abstractions,
            [
                {
                    "name": "Core Layer",
                    "description": "Explains the core layer.",
                    "files": [0],
                }
            ],
        )

    @patch("nodes.call_llm")
    def test_auto_budget_is_clamped_before_planning(self, mock_call_llm):
        chunks = [
            {
                "chunk_id": "c1",
                "file_index": 0,
                "filepath": "entry.py",
                "language": "python",
                "engine": "code-chunk",
                "chunk_kind": "entity",
                "symbol_path": "entry",
                "signature": "def entry()",
                "line_range": {"start": 1, "end": 2},
                "content": "def entry():\n    pass",
                "context_text": "Entry function",
                "parent_scope": "",
                "related_imports": [],
            }
        ]
        mock_call_llm.side_effect = [
            """```yaml
recommended_count: 20
min_count: 8
max_count: 20
reason: |
  Overestimates on purpose.
```""",
            """```yaml
abstractions:
  - name: |
      Entry Layer
    reason: |
      Entry point.
    supporting_chunk_ids:
      - c1
```""",
            """```yaml
abstractions:
  - name: |
      Entry Layer
    description: |
      Explains the entry point.
    file_indices:
      - 0 # entry.py
    supporting_chunk_ids:
      - c1
```""",
        ]
        prep_res = {
            "chunk_inventory": chunks,
            "file_count": 1,
            "project_name": "entryproj",
            "language": "english",
            "use_cache": False,
            "max_abstraction_num": "auto",
            "requested_max_abstraction_num": "auto",
            "max_extraction_batches": 40,
            "llm_extraction_concurrency": 1,
        }

        nodes.IdentifyAbstractions().exec(prep_res)

        self.assertEqual(
            mock_call_llm.call_args_list[1].kwargs["metadata"]["max_abstraction_num"],
            12,
        )

    def test_parallel_extraction_preserves_batch_order(self):
        class ParallelProbeIdentify(nodes.IdentifyAbstractions):
            def _plan_batches(self, project_name, chunk_inventory, use_cache, max_extraction_batches=None):
                return [
                    {"name": "first", "reason": "", "chunk_ids": ["c1"]},
                    {"name": "second", "reason": "", "chunk_ids": ["c2"]},
                ]

            def _extract_batch_abstractions(self, project_name, batch, chunks, language, use_cache, metadata=None):
                if batch["name"] == "first":
                    if not second_started.wait(0.5):
                        raise AssertionError("expected second batch to start before first returns")
                    time.sleep(0.01)
                else:
                    second_started.set()
                return [
                    {
                        "name": batch["name"],
                        "description": f"{batch['name']} description",
                        "file_indices": [chunks[0]["file_index"]],
                        "supporting_chunk_ids": [chunks[0]["chunk_id"]],
                    }
                ]

        second_started = threading.Event()
        chunks = [
            {
                "chunk_id": "c1",
                "file_index": 0,
                "filepath": "a.py",
                "chunk_kind": "entity",
                "line_range": {"start": 1, "end": 1},
                "content": "a",
                "context_text": "a",
                "symbol_path": "a",
                "engine": "code-chunk",
            },
            {
                "chunk_id": "c2",
                "file_index": 1,
                "filepath": "b.py",
                "chunk_kind": "entity",
                "line_range": {"start": 1, "end": 1},
                "content": "b",
                "context_text": "b",
                "symbol_path": "b",
                "engine": "code-chunk",
            },
        ]
        prep_res = {
            "chunk_inventory": chunks,
            "file_count": 2,
            "project_name": "sample",
            "language": "english",
            "use_cache": False,
            "max_abstraction_num": 10,
            "max_extraction_batches": 2,
            "llm_extraction_concurrency": 2,
        }
        jobs = [
            {
                "ordinal": 0,
                "batch": {"name": "first", "reason": "", "chunk_ids": ["c1"]},
                "chunks": [chunks[0]],
                "metadata": {},
            },
            {
                "ordinal": 1,
                "batch": {"name": "second", "reason": "", "chunk_ids": ["c2"]},
                "chunks": [chunks[1]],
                "metadata": {},
            },
        ]

        candidates = ParallelProbeIdentify()._extract_batch_jobs(
            jobs,
            prep_res["project_name"],
            prep_res["language"],
            prep_res["use_cache"],
            prep_res["llm_extraction_concurrency"],
        )
        abstractions = ParallelProbeIdentify()._validate_and_merge(
            candidates,
            prep_res["chunk_inventory"],
            prep_res["file_count"],
            prep_res["max_abstraction_num"],
        )

        self.assertEqual([item["name"] for item in abstractions], ["first", "second"])

    def test_call_llm_writes_jsonl_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry_file = os.path.join(tmpdir, "llm_metrics.jsonl")
            with patch.dict("os.environ", {"LLM_TELEMETRY_FILE": telemetry_file}, clear=False):
                with patch("utils.call_llm.get_llm_provider", return_value="XAI"):
                    with patch("utils.call_llm._call_llm_provider", return_value="ok"):
                        result = call_llm_module.call_llm(
                            "hello",
                            use_cache=False,
                            stage="identify.extract",
                            metadata={"batch_index": 1},
                        )

            self.assertEqual(result, "ok")
            with open(telemetry_file, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "identify.extract")
        self.assertEqual(rows[0]["metadata"], {"batch_index": 1})
        self.assertEqual(rows[0]["prompt_chars"], 5)
        self.assertFalse(rows[0]["cache_hit"])
        self.assertGreaterEqual(rows[0]["duration_sec"], 0)

    @patch.dict(
        "os.environ",
        {
            "LLM_PROVIDER": "XAI",
            "XAI_MODEL": "test-model",
            "XAI_BASE_URL": "https://example.test",
            "XAI_API_KEY": "secret",
            "LLM_HTTP_TIMEOUT": "17",
        },
        clear=False,
    )
    @patch("utils.call_llm.requests.post")
    def test_openai_compatible_provider_uses_timeout(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value.raise_for_status.return_value = None

        result, prompt_tokens, completion_tokens = call_llm_module._call_llm_provider("hello")

        self.assertEqual(result, "ok")
        self.assertEqual(prompt_tokens, 0)
        self.assertEqual(completion_tokens, 0)
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 17.0)


if __name__ == "__main__":
    unittest.main()
