from __future__ import annotations

import json
from pathlib import Path

from cart_scout.schema import PageSnapshot, ShoppingTaskSpec


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_task_specs(path: str | Path) -> list[ShoppingTaskSpec]:
    return [ShoppingTaskSpec.model_validate(row) for row in load_jsonl(path)]


def load_snapshots(path: str | Path) -> list[PageSnapshot]:
    return [PageSnapshot.model_validate(row) for row in load_jsonl(path)]
