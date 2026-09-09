"""Loopback-only, stateless Web/API shell over the existing travel orchestrator."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from .model_output import ModelOutputError
from .orchestrator import TravelPlanningOrchestrator
from .runtime import SUPPORTED_EXTRACTOR_MODES, ExtractorRuntimeConfig, build_extractor

ASSETS = Path(__file__).parent / "web_assets"
MAX_BODY = 32768


def parse_request(payload: object) -> list[str]:
    if not isinstance(payload, dict) or set(payload) - {"message", "context"}:
        raise ValueError("Unexpected request fields")
    message, context = payload.get("message"), payload.get("context", [])
    if not isinstance(message, str) or not message.strip() or len(message) > 2000:
        raise ValueError("Message must contain 1 to 2000 characters")
    if not isinstance(context, list) or len(context) > 4:
        raise ValueError("At most four previous user messages are accepted")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 2000 for item in context):
        raise ValueError("Invalid context")
    turns = [item.strip() for item in context] + [message.strip()]
    if sum(map(len, turns)) > 6000:
        raise ValueError("Conversation is too long")
    return turns


class TravelWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, port: int, orchestrator: TravelPlanningOrchestrator, profile="rule", clock=date.today
    ):
        self.orchestrator = orchestrator
        self.profile = profile
        self.clock = clock
        self.inference_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), TravelHandler)


class TravelHandler(BaseHTTPRequestHandler):
    server: TravelWebServer
    server_version = "TravelDemo"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        # Never log user prompts, cookies or request bodies in the public demo.
        return

    def reply(self, status, data, content_type="application/json; charset=utf-8"):
        raw = json.dumps(data, ensure_ascii=False).encode() if not isinstance(data, bytes) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        self.end_headers()
        self.wfile.write(raw)

    def allowed_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self):
        if not self.allowed_host():
            self.reply(403, {"error": "HOST_REJECTED"})
            return
        route = urlsplit(self.path).path
        if route == "/api/status":
            self.reply(
                200,
                {
                    "status": "ready",
                    "profile": self.server.profile,
                    "today": self.server.clock().isoformat(),
                    "catalog": "demo_catalog_v1",
                    "external_writes": False,
                },
            )
            return
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if route not in files:
            self.reply(404, {"error": "NOT_FOUND"})
            return
        name, content_type = files[route]
        self.reply(200, (ASSETS / name).read_bytes(), content_type)

    def do_POST(self):
        if not self.allowed_host():
            self.reply(403, {"error": "HOST_REJECTED"})
            return
        origin = self.headers.get("Origin")
        if origin != "http://" + self.headers.get("Host", ""):
            self.reply(403, {"error": "ORIGIN_REJECTED"})
            return
        if urlsplit(self.path).path != "/api/plan":
            self.reply(404, {"error": "NOT_FOUND"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.reply(415, {"error": "JSON_REQUIRED"})
            return
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                raise ValueError("Invalid framing")
            size = int(lengths[0])
            if not 0 < size <= MAX_BODY:
                self.reply(413, {"error": "REQUEST_TOO_LARGE", "message": "请求过长，请重新开始。"})
                return
            turns = parse_request(json.loads(self.rfile.read(size)))
        except (ValueError, UnicodeError, TimeoutError, RecursionError):
            self.reply(
                400,
                {
                    "error": "INVALID_REQUEST",
                    "message": "请填写有效需求；最多补充四轮，总长度不超过6000字。",
                },
            )
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.reply(429, {"error": "BUSY", "message": "模型正在处理另一条请求，请稍后重试。"})
            return
        start = perf_counter()
        try:
            # Only user text is carried between requests. No browser-provided decision,
            # tool arguments, model paths or authority enters the orchestrator.
            message = "。补充：".join(turns)
            result = self.server.orchestrator.handle(message, self.server.clock())
            self.reply(
                200,
                {
                    "response": result.to_dict(),
                    "profile": self.server.profile,
                    "elapsed_ms": round((perf_counter() - start) * 1000, 2),
                },
            )
        except ModelOutputError:
            self.reply(
                422,
                {
                    "error": "MODEL_OUTPUT_INVALID",
                    "message": "模型输出未通过校验，未调用目录工具。请重新描述需求。",
                },
            )
        except ValueError:
            self.reply(
                400,
                {
                    "error": "INVALID_TRAVEL_INPUT",
                    "message": "日期或旅行信息无效，请检查后重新填写。",
                },
            )
        except Exception:  # noqa: BLE001 - mask backend details at the public HTTP boundary
            self.reply(
                503,
                {
                    "error": "RUNTIME_UNAVAILABLE",
                    "message": "本地模型或配置暂不可用。没有切换模型，也没有生成替代方案。",
                },
            )
        finally:
            self.server.inference_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8504)
    parser.add_argument("--extractor-mode", choices=SUPPORTED_EXTRACTOR_MODES, default="rule")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    root = Path(__file__).resolve().parents[2]
    config = replace(ExtractorRuntimeConfig.from_environment(root), mode=args.extractor_mode)
    try:
        server = TravelWebServer(
            args.port, TravelPlanningOrchestrator(build_extractor(config)), args.extractor_mode
        )
    except (ValueError, OSError, RuntimeError):
        parser.error("Cannot start this profile; check port, local model files and configuration.")
    print(
        f"Travel demo: http://127.0.0.1:{args.port}/ | selected profile={args.extractor_mode}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
