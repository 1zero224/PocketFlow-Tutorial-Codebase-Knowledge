from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from webapp.service import TaskQueueService

WEB_ROOT = Path(__file__).resolve().parent / "static"


def create_handler(service: TaskQueueService):
    class AppHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                return self._send_json(service.snapshot())
            if parsed.path == "/api/defaults":
                return self._send_json(service.defaults())
            return super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
            except ValueError as exc:
                return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

            if parsed.path == "/api/tasks":
                try:
                    task = service.add_task(payload)
                except ValueError as exc:
                    return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return self._send_json(task, status=HTTPStatus.CREATED)

            if parsed.path == "/api/start":
                return self._send_json(service.start_queue())

            if parsed.path == "/api/pick-folder":
                try:
                    payload = service.pick_folder()
                except ValueError as exc:
                    return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return self._send_json(payload)

            return self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown API route")

        def do_DELETE(self):
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/tasks/"):
                return self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown API route")

            task_id = parsed.path.rsplit("/", 1)[-1]
            try:
                deleted = service.delete_task(task_id)
            except ValueError as exc:
                return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return self._send_json(deleted)

        def log_message(self, format, *args):  # noqa: A003
            return

        def _read_json_body(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("请求体必须是合法 JSON") from exc
            if not isinstance(data, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return data

        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: HTTPStatus, message: str):
            self._send_json({"error": message}, status=status)

    return AppHandler


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    service = TaskQueueService()
    handler = create_handler(service)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"仓库分析任务控制台已启动：{url}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动仓库分析任务控制台。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
