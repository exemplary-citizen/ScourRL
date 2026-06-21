from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from cart_scout.schema import ShoppingTaskSpec
from cart_scout.task_bank import generate_task_specs, task_category


DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_DATASET_ID = "cart-scout-small-rl"


class EpisodePlan(BaseModel):
    episode_id: str
    group_id: str
    rollout_index: int
    split: str
    category: str | None = None
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    max_steps: int = 35
    task: ShoppingTaskSpec


class EpisodeRecord(EpisodePlan):
    status: str = "planned"
    reward: float | None = None
    answer: str | None = None
    breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def build_episode_plans(
    episode_count: int = 300,
    task_count: int | None = None,
    rollouts_per_task: int = 3,
    seed: int = 13,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_steps: int = 35,
    dataset_id: str = DEFAULT_DATASET_ID,
    eval_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> list[EpisodePlan]:
    """Build stable grouped episode plans.

    The default is 100 task groups x 3 rollouts = 300 episodes. Repeated rollouts
    for the same task are useful for small-scale preference or GRPO-style training.
    """
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    if rollouts_per_task < 1:
        raise ValueError("rollouts_per_task must be positive")
    if not 0 <= eval_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("eval_fraction and test_fraction must be in [0, 1)")
    if eval_fraction + test_fraction >= 1:
        raise ValueError("eval_fraction + test_fraction must be less than 1")

    if task_count is None:
        task_count = math.ceil(episode_count / rollouts_per_task)
    specs = generate_task_specs(task_count, seed=seed)

    plans: list[EpisodePlan] = []
    for group_index, spec in enumerate(specs):
        split = _split_for_group(group_index, len(specs), eval_fraction, test_fraction)
        for rollout_index in range(rollouts_per_task):
            if len(plans) >= episode_count:
                break
            plans.append(
                EpisodePlan(
                    episode_id=f"{dataset_id}-{len(plans) + 1:04d}",
                    group_id=spec.task_id,
                    rollout_index=rollout_index,
                    split=split,
                    category=task_category(spec.task_id),
                    provider=provider,
                    model=model,
                    max_steps=max_steps,
                    task=spec,
                )
            )
    return plans


def read_episode_plans(path: str | Path) -> list[EpisodePlan]:
    return [EpisodePlan.model_validate(row) for row in read_jsonl(path)]


def read_episode_records(path: str | Path) -> list[EpisodeRecord]:
    return [EpisodeRecord.model_validate(row) for row in read_jsonl(path)]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(data, sort_keys=True) + "\n")


def write_hud_taskset(path: str | Path, plans: list[EpisodePlan]) -> None:
    """Write a HUD taskset file that can be used for future evals.

    Each planned episode becomes one HUD task. The task slug is the episode id so
    repeated rollouts for the same group remain distinguishable in HUD results.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""Generated CartScout episode taskset. Do not edit by hand."""',
        "",
        "from env import env, shopping_context_task  # noqa: F401",
        "",
        "",
        "def _episode_task(episode_id, spec):",
        "    task = shopping_context_task(**spec)",
        "    task.slug = episode_id",
        "    return task",
        "",
        "",
        "tasks = [",
    ]
    for plan in plans:
        spec = plan.task.model_dump(mode="json")
        lines.extend(
            [
                f"    _episode_task({plan.episode_id!r}, {{",
                f"        'task_id': {spec['task_id']!r},",
                f"        'instruction': {spec['instruction']!r},",
                f"        'max_price': {spec['max_price']!r},",
                f"        'must_have': {spec['must_have']!r},",
                f"        'must_not_have': {spec['must_not_have']!r},",
                f"        'allowed_domains': {spec['allowed_domains']!r},",
                f"        'token_budget': {spec['token_budget']!r},",
                f"        'require_cart': {spec['require_cart']!r},",
                "    }),",
            ]
        )
    lines.append("]")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_training_prompt(plan: EpisodePlan) -> str:
    task = plan.task
    return "\n".join(
        [
            "Task plus observed browser context will be provided by the environment.",
            "Return PurchasePacket JSON only.",
            "",
            f"Instruction: {task.instruction}",
            f"Max price: ${task.max_price:.2f}",
            f"Must have: {task.must_have}",
            f"Must not have: {task.must_not_have}",
            f"Allowed domains: {task.allowed_domains}",
        ]
    )


def completed_records(records: Iterable[EpisodeRecord], min_reward: float = 0.0) -> list[EpisodeRecord]:
    return [record for record in records if record.answer and record.reward is not None and record.reward >= min_reward]


def _split_for_group(group_index: int, group_count: int, eval_fraction: float, test_fraction: float) -> str:
    eval_start = int(group_count * (1.0 - eval_fraction - test_fraction))
    test_start = int(group_count * (1.0 - test_fraction))
    if group_index >= test_start:
        return "test"
    if group_index >= eval_start:
        return "eval"
    return "train"
