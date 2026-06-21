import json
import os
from typing import Any

import requests
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import RemoteRolloutProcessor, evaluation_test
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig

from cart_scout.reward import score_grpo_packet
from cart_scout.schema import ShoppingTaskSpec
from fireworks_rft.remote_dataset import task_to_remote_row


REMOTE_BASE_URL = (
    os.getenv("EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL")
    or os.getenv("CART_SCOUT_REMOTE_BASE_URL")
    or "http://127.0.0.1:9000"
).rstrip("/")
DEFAULT_MODEL = os.getenv(
    "CART_SCOUT_REMOTE_MODEL",
    "accounts/fireworks/models/qwen3-vl-8b-instruct",
)


def score_row(row: EvaluationRow) -> EvaluationRow:
    answer = _assistant_answer(row)
    task = _task_spec(row.ground_truth)
    result = score_grpo_packet(answer, task)
    row.evaluation_result = EvaluateResult(
        score=result.score,
        is_score_valid=True,
        reason=json.dumps(
            {
                "task_id": task.task_id,
                "reasons": result.reasons,
                "breakdown": result.breakdown,
            },
            ensure_ascii=True,
        ),
        metrics={
            name: MetricResult(score=value, reason=name, is_score_valid=True)
            for name, value in result.breakdown.items()
        },
    )
    return row


def remote_status_output_loader(config: DataLoaderConfig) -> DynamicDataLoader:
    def load_remote_row() -> list[EvaluationRow]:
        response = requests.get(
            f"{REMOTE_BASE_URL}/status",
            params={"rollout_id": config.rollout_id},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        messages = payload.get("messages") or []
        task = payload.get("task")
        return [
            EvaluationRow(
                messages=messages,
                ground_truth=task,
            )
        ]

    return DynamicDataLoader(generators=[load_remote_row])


@evaluation_test(
    input_rows=[
        [
            task_to_remote_row(
                ShoppingTaskSpec(
                    task_id="remote-smoke-usb-c-charger",
                    instruction=(
                        "Find a USB-C charger under $40. Must support USB-C Power Delivery "
                        "and at least 30W. Prepare a recommendation only."
                    ),
                    allowed_domains=["target.com", "amazon.com"],
                    max_price=40.0,
                    must_have=["USB-C", "Power Delivery", "30W"],
                    must_not_have=["Lightning"],
                    token_budget=750,
                )
            )
        ]
    ],
    completion_params=[
        {
            "model": DEFAULT_MODEL,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
        }
    ],
    max_dataset_rows=1,
    passed_threshold=0.0,
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=REMOTE_BASE_URL,
        output_data_loader=remote_status_output_loader,
        poll_interval=1.0,
        timeout_seconds=120.0,
    ),
    mode="pointwise",
)
def test_cart_scout_remote_rollout(row: EvaluationRow) -> EvaluationRow:
    return score_row(row)


def _assistant_answer(row: EvaluationRow) -> str:
    for message in reversed(row.messages):
        if message.role == "assistant" and message.content:
            if isinstance(message.content, str):
                return message.content
            return json.dumps(message.content, ensure_ascii=True)
    raise ValueError("row has no assistant answer to score")


def _task_spec(value: Any) -> ShoppingTaskSpec:
    if isinstance(value, ShoppingTaskSpec):
        return value
    if not isinstance(value, dict):
        raise ValueError("ground_truth must be a ShoppingTaskSpec dict")
    return ShoppingTaskSpec.model_validate(value)
