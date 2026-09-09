"""HTTP-level proof of the demo, no model or external service required."""

import http.client
import json
import threading
from datetime import date

import pytest

from travel_itinerary.catalog import DemoTripCatalog
from travel_itinerary.model_output import ModelOutputError
from travel_itinerary.orchestrator import TravelPlanningOrchestrator
from travel_itinerary.web import TravelWebServer, parse_request


@pytest.fixture
def server():
    app = TravelWebServer(0, TravelPlanningOrchestrator(), clock=lambda: date(2026, 9, 8))
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    yield app
    app.shutdown()
    app.server_close()
    thread.join(timeout=2)


def request(server, method, path, body=None, headers=None):
    host = f"127.0.0.1:{server.server_port}"
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    selected = {"Origin": "http://" + host, "Content-Type": "application/json"}
    selected.update(headers or {})
    payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    connection.request(method, path, payload, selected)
    result = connection.getresponse()
    content = result.read()
    output = result.status, dict(result.getheaders()), content
    connection.close()
    return output


def test_product_page_assets_and_profile_are_served_locally(server):
    status, headers, page = request(server, "GET", "/")
    assert status == 200 and "旅行草案" in page.decode()
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert request(server, "GET", "/app.js")[0] == 200
    assert request(server, "GET", "/style.css")[0] == 200
    status, _, body = request(server, "GET", "/api/status")
    assert json.loads(body)["profile"] == "rule"
    assert json.loads(body)["external_writes"] is False


def test_web_clarify_then_tool_then_draft_and_no_cross_request_memory(server):
    first = "我想去杭州旅游。"
    response = json.loads(request(server, "POST", "/api/plan", {"message": first})[2])
    assert response["response"]["action"] == "clarify"
    followup = "从南京出发，10月1日，玩三天，两个人，预算5000元。"
    status, _, raw = request(server, "POST", "/api/plan", {"message": followup, "context": [first]})
    response = json.loads(raw)["response"]
    assert status == 200
    assert response["action"] == "final_plan"
    assert response["decision"]["tool_call"]["name"] == "search_trip_options"
    assert response["plan"]["catalog_source"] == "demo_catalog_v1"
    assert response["plan"]["estimated_total_cny"] == 3600
    isolated = json.loads(request(server, "POST", "/api/plan", {"message": "两个人"})[2])
    assert isolated["response"]["request"]["destination"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "x", "tool_call": {}},
        {"message": "x", "profile": "hf_adapter"},
        {"message": "x", "context": ["x"] * 5},
        {"message": "x", "context": [None]},
    ],
)
def test_web_rejects_overscope_and_malformed_requests(server, payload):
    assert request(server, "POST", "/api/plan", payload)[0] == 400


def test_web_origin_host_and_path_guards(server):
    assert (
        request(server, "POST", "/api/plan", {"message": "x"}, {"Origin": "https://evil.example"})[
            0
        ]
        == 403
    )
    assert request(server, "GET", "/api/status", headers={"Host": "evil.example"})[0] == 403
    assert request(server, "GET", "/../pyproject.toml")[0] == 404
    assert (
        request(server, "POST", "/api/plan", {"message": "x"}, {"Content-Type": "text/plain"})[0]
        == 415
    )


def test_web_bounded_body_and_busy_response(server):
    assert request(server, "POST", "/api/plan", {"message": "x" * 40000})[0] == 413
    with server.inference_lock:
        assert request(server, "POST", "/api/plan", {"message": "x"})[0] == 429


@pytest.mark.parametrize(
    "error,status",
    [
        (ModelOutputError("private-model-detail"), 422),
        (RuntimeError("private-runtime-detail"), 503),
    ],
)
def test_model_failure_never_falls_back_or_invokes_catalog(server, error, status):
    class BrokenExtractor:
        def decide(self, *args):
            raise error

    class SpyCatalog(DemoTripCatalog):
        calls = 0

        def search(self, request):
            self.calls += 1
            return super().search(request)

    catalog = SpyCatalog()
    server.orchestrator = TravelPlanningOrchestrator(BrokenExtractor(), catalog)
    actual, _, raw = request(server, "POST", "/api/plan", {"message": "x"})
    assert actual == status
    assert b"private-" not in raw
    assert catalog.calls == 0


def test_context_budget():
    with pytest.raises(ValueError):
        parse_request({"message": "x" * 2000, "context": ["x" * 2000] * 3})
