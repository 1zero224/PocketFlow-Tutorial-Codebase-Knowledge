import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app_config import DEFAULT_INCLUDE_PATTERNS, build_shared_state
from webapp.service import TaskQueueService


class AppConfigTests(unittest.TestCase):
    def test_build_shared_state_uses_default_patterns_when_empty(self):
        shared = build_shared_state(
            repo_url=None,
            local_dir="sample",
            project_name=None,
            github_token=None,
            include_patterns=[],
            exclude_patterns=[],
        )

        self.assertEqual(shared["include_patterns"], set(DEFAULT_INCLUDE_PATTERNS))
        self.assertTrue(shared["exclude_patterns"])


class TaskQueueServiceTests(unittest.TestCase):
    def test_add_task_normalizes_directory_and_patterns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TaskQueueService()
            task = service.add_task(
                {
                    "source_path": temp_dir,
                    "project_name": "demo",
                    "include_patterns": "*.py\n*.md",
                    "exclude_patterns": "tests/*\nnode_modules/*",
                }
            )

        self.assertEqual(task["config"]["source_path"], str(Path(temp_dir).resolve()))
        self.assertEqual(task["config"]["include_patterns"], ["*.py", "*.md"])
        self.assertEqual(
            task["config"]["exclude_patterns"],
            ["tests/*", "node_modules/*"],
        )
        self.assertEqual(
            task["config"]["output_dir"],
            str(Path(temp_dir).resolve() / "pf_guide"),
        )
        self.assertEqual(task["status"], "pending")

    def test_start_queue_runs_pending_task(self):
        calls = []

        def fake_runner(task, flow_factory):
            calls.append(task.id)
            task.append_log("runner called")
            return "pf_guide"

        with tempfile.TemporaryDirectory() as temp_dir:
            service = TaskQueueService(task_runner=fake_runner)
            service.add_task({"source_path": temp_dir})
            service.start_queue()

            if service._worker is not None:
                service._worker.join(timeout=2)

        snapshot = service.snapshot()
        self.assertEqual(len(calls), 1)
        self.assertEqual(snapshot["completed_count"], 1)
        self.assertEqual(snapshot["tasks"][0]["result_dir"], "pf_guide")

    def test_delete_task_removes_non_running_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TaskQueueService()
            created = service.add_task({"source_path": temp_dir})
            deleted = service.delete_task(created["id"])

        self.assertEqual(deleted["id"], created["id"])
        self.assertEqual(service.snapshot()["tasks"], [])

    @patch("webapp.service.pick_windows_folder")
    def test_pick_folder_returns_selected_path_and_default_output_dir(self, mock_pick_folder):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_pick_folder.return_value = temp_dir
            service = TaskQueueService()
            payload = service.pick_folder()

        self.assertEqual(payload["selected_path"], str(Path(temp_dir).resolve()))
        self.assertEqual(
            payload["output_dir"],
            str(Path(temp_dir).resolve() / "pf_guide"),
        )

    @patch("webapp.service.pick_windows_folder")
    def test_pick_folder_raises_on_cancel(self, mock_pick_folder):
        mock_pick_folder.side_effect = ValueError("已取消文件夹选择")
        service = TaskQueueService()
        with self.assertRaisesRegex(ValueError, "已取消文件夹选择"):
            service.pick_folder()


if __name__ == "__main__":
    unittest.main()
