"""OllamaAgent / AnthropicAgent のHTTP契約に対する統合テスト。

実際のOllama/Anthropicサーバーには依存せず、標準ライブラリの http.server で
最小限のモックを立てて、json-v1契約(リクエスト形状・レスポンスのパース・
ウォームアップの失敗検知)がコード側で正しく実装されていることを検証する。
"""

import json
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import speed_arena as sa


def free_port() -> int:
    """未使用のTCPポート番号を1つ確保する(接続不能テスト用)。

    bind直後にcloseするため他プロセスとの完全な排他はできないが、返却前に
    connect_exで実際に接続不可であることを確認し、フレーク率を下げる。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is unexpectedly in use")
    finally:
        probe.close()
    return port


def _make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            state["requests"].append({
                "path": self.path,
                "headers": dict(self.headers.items()),
                "json": json.loads(body) if body else None,
            })
            response = state["responses"].pop(0) if state["responses"] else {}
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self._handle()

        def log_message(self, *_args):
            pass  # テスト出力を静かにする

    return Handler


class MockServer:
    """レスポンスを順番に返す最小限のJSON HTTPモックサーバー。"""

    def __init__(self, responses):
        self.state = {"requests": [], "responses": list(responses)}
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.httpd.shutdown()
        self.thread.join(timeout=5.0)
        self.httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def requests(self):
        return self.state["requests"]


class TestOllamaAgentHTTP(unittest.TestCase):
    def test_decide_sends_json_v1_contract(self):
        with MockServer([{"message": {"content": '{"action":"play","card":5,"pile":0}'}}]) as server:
            agent = sa.OllamaAgent("test-model", host=server.url, keep_alive=123.0)
            result = agent.decide(sa.WARMUP_SNAPSHOT)

        self.assertEqual(result, {"action": "play", "card": 5, "pile": 0})
        req = server.requests[0]["json"]
        self.assertEqual(req["model"], "test-model")
        self.assertEqual(req["stream"], False)
        self.assertEqual(req["think"], False)
        self.assertEqual(req["format"], sa.ACTION_SCHEMA)
        self.assertEqual(req["keep_alive"], 123.0)
        self.assertEqual(req["options"]["temperature"], sa.TEMPERATURE)
        self.assertEqual(req["options"]["num_predict"], sa.MAX_TOKENS)
        self.assertEqual(req["messages"][0]["role"], "system")
        self.assertEqual(req["messages"][1]["role"], "user")
        self.assertEqual(server.requests[0]["path"], "/api/chat")

    def test_decide_parses_pass(self):
        with MockServer([{"message": {"content": '{"action":"pass"}'}}]) as server:
            agent = sa.OllamaAgent("test-model", host=server.url)
            result = agent.decide(sa.WARMUP_SNAPSHOT)
        self.assertEqual(result, {"action": "pass"})

    def test_decide_flags_malformed_content_as_parse_error(self):
        with MockServer([{"message": {"content": "sure, here you go: play card 5"}}]) as server:
            agent = sa.OllamaAgent("test-model", host=server.url)
            result = agent.decide(sa.WARMUP_SNAPSHOT)
        self.assertTrue(result.get("parse_error"))
        self.assertEqual(result["action"], "pass")

    def test_decide_rejects_extra_key_response(self):
        with MockServer([{"message": {"content": '{"action":"pass","card":5}'}}]) as server:
            agent = sa.OllamaAgent("test-model", host=server.url)
            result = agent.decide(sa.WARMUP_SNAPSHOT)
        self.assertTrue(result.get("parse_error"))

    def test_warmup_ok_when_server_responds(self):
        with MockServer([{"message": {"content": '{"action":"pass"}'}}]) as server:
            agent = sa.OllamaAgent("test-model", host=server.url)
            result = agent.warmup()
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["duration"], 0.0)

    def test_warmup_failed_when_unreachable(self):
        agent = sa.OllamaAgent("test-model", host=f"http://127.0.0.1:{free_port()}", timeout=2.0)
        result = agent.warmup()
        self.assertEqual(result["status"], "failed")

    def test_warmup_gets_a_longer_timeout_than_a_move(self):
        calls = []

        class RecordingSession:
            def post(self, url, json=None, timeout=None):
                calls.append(timeout)
                raise sa.requests.RequestException("stop here")

        agent = sa.OllamaAgent("test-model", host="http://127.0.0.1:1",
                               timeout=60.0, warmup_timeout=900.0)
        agent.session = RecordingSession()
        agent.warmup()
        agent.decide(sa.WARMUP_SNAPSHOT)
        self.assertEqual(calls, [900.0, 60.0])

    def test_quantization_is_asked_once_when_not_given(self):
        with MockServer([{"details": {"quantization_level": "MXFP4"}}]) as server:
            agent = sa.OllamaAgent("test-model", host=server.url)
            self.assertEqual(agent.quantization, "MXFP4")
            self.assertEqual(agent.quantization, "MXFP4")
            self.assertEqual(len(server.requests), 1)
            self.assertEqual(server.requests[0]["path"], "/api/show")
            self.assertEqual(server.requests[0]["json"]["model"], "test-model")

    def test_quantization_given_explicitly_is_not_asked(self):
        with MockServer([{"details": {"quantization_level": "MXFP4"}}]) as server:
            agent = sa.OllamaAgent("test-model", host=server.url, quantization="Q4_K_M")
            self.assertEqual(agent.quantization, "Q4_K_M")
            self.assertEqual(server.requests, [])

    def test_quantization_falls_back_to_unknown_when_unreachable(self):
        agent = sa.OllamaAgent("test-model", host=f"http://127.0.0.1:{free_port()}", timeout=2.0)
        self.assertEqual(agent.quantization, "unknown")


class TestAnthropicAgentHTTP(unittest.TestCase):
    def test_decide_sends_json_v1_contract(self):
        response = {
            "content": [
                {"type": "tool_use", "name": "submit_action",
                 "input": {"action": "play", "card": 7, "pile": 1}},
            ],
        }
        with MockServer([response]) as server:
            agent = sa.AnthropicAgent("test-model", base_url=server.url)
            result = agent.decide(sa.WARMUP_SNAPSHOT)

        self.assertEqual(result, {"action": "play", "card": 7, "pile": 1})
        req = server.requests[0]["json"]
        self.assertEqual(req["model"], "test-model")
        self.assertEqual(req["max_tokens"], sa.MAX_TOKENS)
        self.assertEqual(req["temperature"], sa.TEMPERATURE)
        self.assertEqual(req["tools"], [sa.ACTION_TOOL])
        self.assertEqual(req["tool_choice"], {"type": "tool", "name": "submit_action"})
        self.assertEqual(server.requests[0]["path"], "/v1/messages")

    def test_decide_flags_missing_tool_use_as_parse_error(self):
        response = {"content": [{"type": "text", "text": "I'll pass this turn."}]}
        with MockServer([response]) as server:
            agent = sa.AnthropicAgent("test-model", base_url=server.url)
            result = agent.decide(sa.WARMUP_SNAPSHOT)
        self.assertTrue(result.get("parse_error"))
        self.assertEqual(result["action"], "pass")

    def test_decide_rejects_extra_key_in_tool_input(self):
        response = {
            "content": [
                {"type": "tool_use", "name": "submit_action",
                 "input": {"action": "pass", "card": 5}},
            ],
        }
        with MockServer([response]) as server:
            agent = sa.AnthropicAgent("test-model", base_url=server.url)
            result = agent.decide(sa.WARMUP_SNAPSHOT)
        self.assertTrue(result.get("parse_error"))

    def test_warmup_failed_when_unreachable(self):
        agent = sa.AnthropicAgent("test-model", base_url=f"http://127.0.0.1:{free_port()}")
        result = agent.warmup()
        self.assertEqual(result["status"], "failed")


class TestRunMatchWarmupFailureIsInvalid(unittest.TestCase):
    """#1: ウォームアップ失敗はpassの擬似応答で隠さず、無効試合として記録する。"""

    def test_unreachable_agent_makes_match_invalid(self):
        broken = sa.OllamaAgent("broken-model", host=f"http://127.0.0.1:{free_port()}", timeout=2.0)
        fine = sa.HeuristicAgent("fine-bot", latency=0.01)
        ms = sa.run_match(broken, fine, seed=1, max_duration=5.0)
        self.assertFalse(ms.valid)
        self.assertEqual(ms.end_reason, "warmup_failed")
        by_agent = {s["agent"]: s for s in ms.per_player}
        self.assertEqual(by_agent["broken-model"]["warmup_status"], "failed")
        self.assertEqual(by_agent["fine-bot"]["warmup_status"], "ok")


if __name__ == "__main__":
    unittest.main()
