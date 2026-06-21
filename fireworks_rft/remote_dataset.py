from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from eval_protocol.models import EvaluationRow, InputMetadata, Message

from cart_scout.data import load_task_specs
from cart_scout.schema import ShoppingTaskSpec


SYSTEM_PROMPT = """\
You are the CartScout shopping agent. Browse allowed public retailer pages, gather buying-critical
context, cite evidence, and stop before checkout. Return a final PurchasePacket JSON object.
"""


def build_remote_rows(tasks_path: str | Path, *, limit: int | None = None) -> list[EvaluationRow]:
    tasks = load_task_specs(tasks_path)
    if limit is not None:
        tasks = tasks[:limit]
    return [task_to_remote_row(task) for task in tasks]


def task_to_remote_row(task: ShoppingTaskSpec) -> EvaluationRow:
    return EvaluationRow(
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=json.dumps({"task": task.model_dump(mode="json")}, ensure_ascii=True),
            ),
        ],
        ground_truth=task.model_dump(mode="json"),
        input_metadata=InputMetadata(row_id=task.task_id),
    )


def write_jsonl(rows: Iterable[EvaluationRow], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_none=True) + "\n")
