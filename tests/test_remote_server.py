from __future__ import annotations

import json
from types import SimpleNamespace

from eval_protocol.models import EvaluationRow
from fastapi.testclient import TestClient

from cart_scout.schema import PurchasePacket, ShoppingTaskSpec
from fireworks_rft.remote_dataset import build_remote_rows, task_to_remote_row
from fireworks_rft.remote_evaluator import score_row
from fireworks_rft.remote_server import app


def test_remote_server_smoke_rollout() -> None:
    task = _task()
    client = TestClient(app)

    response = client.post("/init", json=_init_payload(task))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["terminated"] is True
    assert payload["rollout_id"] == "rollout-smoke"
    assert payload["reward"] > 0.7
    packet = PurchasePacket.model_validate_json(payload["final_answer"])
    assert packet.stop_before_checkout is True
    assert "target.com" in str(packet.url)

    status = client.get("/status", params={"rollout_id": "rollout-smoke"})
    assert status.status_code == 200
    assert status.json()["final_answer"] == payload["final_answer"]


def test_remote_rows_are_task_only() -> None:
    row = task_to_remote_row(_task())

    assert len(row.messages) == 2
    assert row.messages[1].role == "user"
    assert "observed product snippets" not in row.messages[1].content
    assert json.loads(row.messages[1].content)["task"]["task_id"] == "remote-smoke"
    assert row.ground_truth["task_id"] == "remote-smoke"


def test_build_remote_rows_from_task_file(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(_task().model_dump_json() + "\n", encoding="utf-8")

    rows = build_remote_rows(path)

    assert len(rows) == 1
    assert rows[0].input_metadata.row_id == "remote-smoke"


def test_remote_evaluator_scores_server_status_payload() -> None:
    task = _task()
    client = TestClient(app)
    payload = client.post("/init", json=_init_payload(task)).json()
    row = score_row(
        EvaluationRow(messages=payload["messages"], ground_truth=task.model_dump(mode="json"))
    )

    assert row.evaluation_result is not None
    assert row.evaluation_result.score > 0.7
    assert row.evaluation_result.metrics["format"].score > 0


def test_browser_use_mode_runs_existing_hud_harness(monkeypatch) -> None:
    from hud.eval.taskset import Taskset

    task = _task()
    final_answer = json.dumps(
        {
            "query": task.instruction,
            "recommended_product": "30W USB-C Power Delivery Charger",
            "retailer": "Target",
            "url": "https://www.target.com/p/live-hud-result",
            "price": "$19.99",
            "delivery_or_pickup": "Shipping available",
            "seller": "Target",
            "constraints_met": ["USB-C", "Power Delivery", "30W"],
            "constraints_uncertain": [],
            "evidence": [
                {
                    "url": "https://www.target.com/p/live-hud-result",
                    "quote": "30W USB-C Power Delivery",
                    "supports": "USB-C PD wattage",
                },
                {
                    "url": "https://www.target.com/p/live-hud-result",
                    "quote": "$19.99",
                    "supports": "price under budget",
                },
            ],
            "recommendation": "Recommend this charger because it satisfies the constraints.",
            "stop_before_checkout": True,
            "cart_prepared": False,
            "attempted_checkout": False,
        }
    )
    seen = {}

    async def fake_run(self, agent, *, runtime, max_concurrent, rollout_timeout):
        hud_task = next(iter(self.tasks.values()))
        seen["agent"] = agent
        seen["runtime"] = runtime
        seen["rollout_timeout"] = rollout_timeout
        seen["task"] = hud_task
        return SimpleNamespace(
            id="hud-job-smoke",
            runs=[
                SimpleNamespace(
                    reward=0.91,
                    trace_id="hud-trace-smoke",
                    trace=SimpleNamespace(
                        content=final_answer,
                        status="completed",
                        extra={"urls": ["https://www.target.com/p/live-hud-result"]},
                    ),
                )
            ],
        )

    monkeypatch.setenv("CART_SCOUT_REMOTE_MODE", "browser-use")
    monkeypatch.setenv("CART_SCOUT_HUD_RUNTIME", "tcp")
    monkeypatch.setenv("CART_SCOUT_HUD_RUNTIME_URL", "tcp://127.0.0.1:8765")
    monkeypatch.setenv("CART_SCOUT_BROWSER_MAX_STEPS", "7")
    monkeypatch.setattr(Taskset, "run", fake_run)

    response = TestClient(app).post("/init", json=_init_payload(task))

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_answer"] == final_answer
    assert payload["extra"]["hud_job_id"] == "hud-job-smoke"
    assert payload["extra"]["hud_runtime"] == "tcp"
    assert payload["extra"]["model"] == "accounts/fireworks/models/qwen3-vl-8b-instruct"
    assert seen["task"].env == "cart-scout"
    assert seen["task"].id == "shopping-context"
    assert seen["task"].args["instruction"] == task.instruction
    assert seen["agent"].provider == "openai-like"
    assert seen["agent"].base_url == "https://tracing.fireworks.ai"
    assert seen["agent"].max_steps == 7
    assert seen["agent"].flash_mode is True
    assert seen["agent"].use_thinking is False
    assert seen["agent"].use_vision is True
    assert seen["agent"].llm_temperature == 0.7
    assert seen["agent"].llm_top_p == 0.95
    assert seen["agent"].max_completion_tokens == 1024
    assert seen["agent"].reasoning_effort == "none"


def test_structured_cdp_mode_runs_existing_hud_harness(monkeypatch) -> None:
    from hud.eval.taskset import Taskset

    task = _task()
    final_answer = json.dumps(
        {
            "query": task.instruction,
            "recommended_product": "30W USB-C Wall Charger",
            "retailer": "Amazon",
            "url": "https://www.amazon.com/dp/live-cdp-result",
            "price": "$18.99",
            "delivery_or_pickup": "Shipping available",
            "seller": "Amazon",
            "constraints_met": ["USB-C", "30W"],
            "constraints_uncertain": [],
            "evidence": [
                {
                    "url": "https://www.amazon.com/dp/live-cdp-result",
                    "quote": "30W USB-C wall charger",
                    "supports": "USB-C wattage",
                },
                {
                    "url": "https://www.amazon.com/dp/live-cdp-result",
                    "quote": "$18.99",
                    "supports": "price under budget",
                },
            ],
            "recommendation": "Recommend this charger because it satisfies the constraints.",
            "stop_before_checkout": True,
            "cart_prepared": False,
            "attempted_checkout": False,
        }
    )
    seen = {}

    async def fake_run(self, agent, *, runtime, max_concurrent, rollout_timeout):
        hud_task = next(iter(self.tasks.values()))
        seen["agent"] = agent
        seen["runtime"] = runtime
        seen["rollout_timeout"] = rollout_timeout
        seen["task"] = hud_task
        return SimpleNamespace(
            id="hud-structured-job-smoke",
            runs=[
                SimpleNamespace(
                    reward=0.93,
                    trace_id="hud-structured-trace-smoke",
                    trace=SimpleNamespace(
                        content=final_answer,
                        status="completed",
                        extra={"harness": "structured-cdp"},
                    ),
                )
            ],
        )

    monkeypatch.setenv("CART_SCOUT_REMOTE_MODE", "structured-cdp")
    monkeypatch.setenv("CART_SCOUT_HUD_RUNTIME", "tcp")
    monkeypatch.setenv("CART_SCOUT_STRUCTURED_MAX_STEPS", "5")
    monkeypatch.setattr(Taskset, "run", fake_run)

    response = TestClient(app).post("/init", json=_init_payload(task))

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_answer"] == final_answer
    assert payload["extra"]["hud_job_id"] == "hud-structured-job-smoke"
    assert payload["extra"]["harness"] == "structured-cdp"
    assert payload["extra"]["model"] == "accounts/fireworks/models/qwen3-vl-8b-instruct"
    assert seen["task"].env == "cart-scout"
    assert seen["task"].id == "shopping-context"
    assert seen["agent"].base_url == "https://tracing.fireworks.ai"
    assert seen["agent"].max_steps == 5
    assert seen["agent"].temperature == 0.7
    assert seen["agent"].top_p == 0.95
    assert seen["agent"].max_tokens == 1024
    assert seen["agent"].reasoning_effort == "none"


def _init_payload(task: ShoppingTaskSpec) -> dict:
    row = task_to_remote_row(task)
    return {
        "completion_params": {
            "model": "accounts/fireworks/models/qwen3-vl-8b-instruct",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
        },
        "messages": [
            message.model_dump(exclude_none=True, mode="json") for message in row.messages
        ],
        "tools": None,
        "model_base_url": "https://tracing.fireworks.ai",
        "metadata": {
            "invocation_id": "invocation-smoke",
            "experiment_id": "experiment-smoke",
            "rollout_id": "rollout-smoke",
            "run_id": "run-smoke",
            "row_id": task.task_id,
        },
        "api_key": "test-key",
    }


def _task() -> ShoppingTaskSpec:
    return ShoppingTaskSpec(
        task_id="remote-smoke",
        instruction=(
            "Find a USB-C charger under $40. Must support USB-C Power Delivery "
            "and at least 30W. Prepare recommendation only."
        ),
        allowed_domains=["target.com", "amazon.com"],
        max_price=40.0,
        must_have=["USB-C", "Power Delivery", "30W"],
        must_not_have=["Lightning"],
        token_budget=750,
    )
