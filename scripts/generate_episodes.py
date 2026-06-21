from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from cart_scout.episodes import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    EpisodeRecord,
    build_episode_plans,
    read_episode_plans,
    read_jsonl,
    write_hud_taskset,
    write_jsonl,
)
from cart_scout.reward import score_purchase_packet


DEFAULT_PLAN_PATH = Path("data/episodes/cart_scout_300_plan.jsonl")
DEFAULT_TASKSET_PATH = Path("data/episodes/generated_tasks.py")
DEFAULT_RECORDS_PATH = Path("data/episodes/cart_scout_300_records.jsonl")


def _plan(args: argparse.Namespace) -> None:
    plans = build_episode_plans(
        episode_count=args.episodes,
        task_count=args.tasks,
        rollouts_per_task=args.rollouts_per_task,
        seed=args.seed,
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        dataset_id=args.dataset_id,
        eval_fraction=args.eval_fraction,
        test_fraction=args.test_fraction,
    )
    write_jsonl(args.output, plans)
    if args.taskset:
        write_hud_taskset(args.taskset, plans)

    print(
        json.dumps(
            {
                "episodes": len(plans),
                "task_groups": len({plan.group_id for plan in plans}),
                "rollouts_per_task": args.rollouts_per_task,
                "provider": args.provider,
                "model": args.model,
                "output": str(args.output),
                "taskset": str(args.taskset) if args.taskset else None,
                "splits": _count_by(plans, "split"),
                "categories": _count_by(plans, "category"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _score_answers(args: argparse.Namespace) -> None:
    plans = read_episode_plans(args.plan)
    answer_rows = read_jsonl(args.answers)
    answer_by_id = {row["episode_id"]: row for row in answer_rows}

    records: list[EpisodeRecord] = []
    for plan in plans:
        row = answer_by_id.get(plan.episode_id)
        if not row:
            records.append(EpisodeRecord(**plan.model_dump(mode="json"), status="missing", error="answer missing"))
            continue
        answer = str(row.get("answer") or row.get("final_answer") or "")
        result = score_purchase_packet(answer, plan.task)
        records.append(
            EpisodeRecord(
                **plan.model_dump(mode="json"),
                status="completed",
                reward=result.score,
                answer=answer,
                breakdown=result.breakdown,
                reasons=result.reasons,
                trace={key: value for key, value in row.items() if key not in {"episode_id", "answer", "final_answer"}},
            )
        )
    write_jsonl(args.output, records)
    _print_record_summary(records, args.output)


async def _collect_async(args: argparse.Namespace) -> None:
    from hud.eval.runtime import HUDRuntime
    from hud.eval.taskset import Taskset

    try:
        from run_browser_use_eval import BrowserUseCDPAgent
    except ModuleNotFoundError:  # pragma: no cover - module execution fallback
        from scripts.run_browser_use_eval import BrowserUseCDPAgent

    plans = read_episode_plans(args.plan)
    if args.limit is not None:
        plans = plans[: args.limit]
    if not plans:
        raise SystemExit("no episode plans selected")

    provider = args.provider or plans[0].provider
    model = args.model or plans[0].model
    max_steps = args.max_steps or max(plan.max_steps for plan in plans)
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None

    taskset_path = args.taskset or DEFAULT_TASKSET_PATH
    write_hud_taskset(taskset_path, plans)
    taskset = Taskset.from_file(str(taskset_path)).filter([plan.episode_id for plan in plans])

    agent = BrowserUseCDPAgent(
        provider=provider,
        model=model,
        max_steps=max_steps,
        api_key=api_key,
        base_url=args.base_url,
    )
    job = await taskset.run(
        agent,
        runtime=HUDRuntime(run_timeout=args.runtime_timeout),
        max_concurrent=args.max_concurrent,
        rollout_timeout=args.rollout_timeout,
    )

    runs_by_slug = {run.slug: run for run in job.runs}
    records: list[EpisodeRecord] = []
    for plan in plans:
        run = runs_by_slug.get(plan.episode_id)
        if run is None:
            records.append(EpisodeRecord(**plan.model_dump(mode="json"), status="missing", error="HUD run missing"))
            continue
        trace = getattr(run, "trace", None)
        answer = getattr(trace, "content", None) if trace else None
        extra = dict(getattr(trace, "extra", {}) or {}) if trace else {}
        status = getattr(trace, "status", None) if trace else None
        breakdown: dict[str, float] = {}
        reasons: list[str] = []
        if answer:
            result = score_purchase_packet(answer, plan.task)
            breakdown = result.breakdown
            reasons = result.reasons
        records.append(
            EpisodeRecord(
                **plan.model_dump(mode="json"),
                status=status or "completed",
                reward=getattr(run, "reward", None),
                answer=answer,
                breakdown=breakdown,
                reasons=reasons,
                trace={
                    "job_id": getattr(job, "id", None),
                    "run_slug": getattr(run, "slug", None),
                    "harness": "browser-use-cdp",
                    "provider": provider,
                    "model": model,
                    **extra,
                },
            )
        )
    write_jsonl(args.output, records)
    _print_record_summary(records, args.output)
    print(f"job: https://hud.ai/jobs/{job.id}")


def _collect(args: argparse.Namespace) -> None:
    asyncio.run(_collect_async(args))


def _count_by(rows: list[Any], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = getattr(row, field) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _print_record_summary(records: list[EpisodeRecord], output: Path) -> None:
    rewards = [record.reward for record in records if record.reward is not None]
    summary = {
        "records": len(records),
        "completed": sum(1 for record in records if record.answer),
        "output": str(output),
        "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "statuses": _count_by(records, "status"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan, collect, and score CartScout RL episodes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create an episode plan JSONL and optional HUD taskset.")
    plan.add_argument("--episodes", type=int, default=300)
    plan.add_argument("--tasks", type=int, default=None, help="Task-group count. Defaults to ceil(episodes / rollouts-per-task).")
    plan.add_argument("--rollouts-per-task", type=int, default=3)
    plan.add_argument("--seed", type=int, default=13)
    plan.add_argument("--provider", default=DEFAULT_PROVIDER)
    plan.add_argument("--model", default=DEFAULT_MODEL)
    plan.add_argument("--max-steps", type=int, default=35)
    plan.add_argument("--dataset-id", default="cart-scout-small-rl")
    plan.add_argument("--eval-fraction", type=float, default=0.10)
    plan.add_argument("--test-fraction", type=float, default=0.10)
    plan.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH)
    plan.add_argument("--taskset", type=Path, default=DEFAULT_TASKSET_PATH, help="Write a HUD taskset file for later eval/collection.")
    plan.set_defaults(func=_plan)

    collect = subparsers.add_parser("collect", help="Run planned episodes through Browser Use over HUD CDP.")
    collect.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    collect.add_argument("--output", type=Path, default=DEFAULT_RECORDS_PATH)
    collect.add_argument("--taskset", type=Path, default=DEFAULT_TASKSET_PATH)
    collect.add_argument("--provider", default=None)
    collect.add_argument("--model", default=None)
    collect.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    collect.add_argument("--base-url", default=None)
    collect.add_argument("--max-steps", type=int, default=None)
    collect.add_argument("--limit", type=int, default=None)
    collect.add_argument("--max-concurrent", type=int, default=1)
    collect.add_argument("--runtime-timeout", type=float, default=1800.0)
    collect.add_argument("--rollout-timeout", type=float, default=1800.0)
    collect.set_defaults(func=_collect)

    score = subparsers.add_parser("score-answers", help="Score externally collected answers against a plan.")
    score.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    score.add_argument("--answers", type=Path, required=True, help="JSONL rows with episode_id and answer/final_answer.")
    score.add_argument("--output", type=Path, default=DEFAULT_RECORDS_PATH)
    score.set_defaults(func=_score_answers)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
