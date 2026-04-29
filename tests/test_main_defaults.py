import sys
import types
import unittest
from unittest.mock import patch


fake_pocketflow = types.ModuleType("pocketflow")


class FakeNode:
    def __init__(self, *args, **kwargs):
        self.cur_retry = 0

    def __rshift__(self, other):
        return other


class FakeFlow:
    def __init__(self, *args, **kwargs):
        pass


fake_pocketflow.Node = FakeNode
fake_pocketflow.BatchNode = FakeNode
fake_pocketflow.Flow = FakeFlow
sys.modules.setdefault("pocketflow", fake_pocketflow)

import main
import nodes


class MainDefaultsTests(unittest.TestCase):
    def test_cli_defaults_generated_tutorial_language_to_chinese(self):
        captured_shared = {}

        class CapturingFlow:
            def run(self, shared):
                captured_shared.update(shared)

        with patch.object(
            sys,
            "argv",
            ["main.py", "--dir", "sample"],
        ):
            with patch("main.create_tutorial_flow", return_value=CapturingFlow()):
                main.main()

        self.assertEqual(captured_shared["language"], "Chinese")

    @patch("nodes.build_chunk_inventory")
    def test_identify_abstractions_defaults_language_to_chinese(self, mock_build_chunk_inventory):
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
        }

        prep_res = nodes.IdentifyAbstractions().prep(shared)

        self.assertEqual(prep_res["language"], "Chinese")


if __name__ == "__main__":
    unittest.main()
