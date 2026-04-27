import unittest
import sys
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
    build_fallback_chunks,
    build_misc_chunks,
    extract_file_indices,
    map_code_chunk_result,
    pack_chunks_for_prompt,
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
batches:
  - name: |
      Alpha Layer
    reason: |
      Alpha function batch.
    chunk_ids:
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
            "max_abstraction_num": 8,
        }

        fetch = nodes.FetchRepo()
        fetch.post(shared, None, fetch.exec(fetch.prep(shared)))
        identify = nodes.IdentifyAbstractions()
        identify.post(shared, None, identify.exec(identify.prep(shared)))

        self.assertEqual([path for path, _ in shared["files"]], ["a.py", "b.py"])
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

        result = call_llm_module._call_llm_provider("hello")

        self.assertEqual(result, "ok")
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 17.0)


if __name__ == "__main__":
    unittest.main()
