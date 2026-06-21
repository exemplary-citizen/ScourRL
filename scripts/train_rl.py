from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cart_scout.episodes import EpisodeRecord, read_episode_records, to_training_prompt, write_jsonl


DEFAULT_RECORDS_PATH = Path("data/episodes/cart_scout_300_records.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/training/cart_scout_small_rl")


def _prepare(args: argparse.Namespace) -> None:
    records = read_episode_records(args.records)
    train_records = [record for record in records if record.split == "train"]
    eval_records = [record for record in records if record.split == "eval"]
    test_records = [record for record in records if record.split == "test"]

    sft_rows = _sft_rows(train_records, min_reward=args.min_sft_reward)
    eval_rows = _sft_rows(eval_records, min_reward=args.min_eval_reward)
    preference_rows = _preference_rows(train_records, margin=args.preference_margin)
    reward_rows = _reward_rows(train_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sft_train.jsonl", sft_rows)
    write_jsonl(args.output_dir / "sft_eval.jsonl", eval_rows)
    write_jsonl(args.output_dir / "preference_pairs.jsonl", preference_rows)
    write_jsonl(args.output_dir / "reward_model_rows.jsonl", reward_rows)

    manifest = {
        "records": len(records),
        "splits": {
            "train": len(train_records),
            "eval": len(eval_records),
            "test": len(test_records),
        },
        "artifacts": {
            "sft_train": str(args.output_dir / "sft_train.jsonl"),
            "sft_eval": str(args.output_dir / "sft_eval.jsonl"),
            "preference_pairs": str(args.output_dir / "preference_pairs.jsonl"),
            "reward_model_rows": str(args.output_dir / "reward_model_rows.jsonl"),
        },
        "thresholds": {
            "min_sft_reward": args.min_sft_reward,
            "min_eval_reward": args.min_eval_reward,
            "preference_margin": args.preference_margin,
        },
        "counts": {
            "sft_train": len(sft_rows),
            "sft_eval": len(eval_rows),
            "preference_pairs": len(preference_rows),
            "reward_model_rows": len(reward_rows),
        },
        "reward_summary": _reward_summary(records),
        "recommended_loop": [
            "Collect grouped rollouts with scripts/generate_episodes.py collect.",
            "Run this prepare command to export SFT, preference, and reward rows.",
            "Fine-tune a policy using sft_train.jsonl as a warm start.",
            "Use preference_pairs.jsonl for DPO/IPO or grouped policy improvement.",
            "Re-run generated taskset evals and compare rewards on eval/test splits.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _sft_rows(records: list[EpisodeRecord], min_reward: float) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        if not record.answer or record.reward is None or record.reward < min_reward:
            continue
        rows.append(
            {
                "episode_id": record.episode_id,
                "group_id": record.group_id,
                "split": record.split,
                "reward": record.reward,
                "messages": [
                    {"role": "user", "content": to_training_prompt(record)},
                    {"role": "assistant", "content": record.answer},
                ],
                "ground_truth": _ground_truth(record),
            }
        )
    return rows


def _preference_rows(records: list[EpisodeRecord], margin: float) -> list[dict]:
    by_group: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        if record.answer and record.reward is not None:
            by_group[record.group_id].append(record)

    rows: list[dict] = []
    for group_id, group_records in sorted(by_group.items()):
        if len(group_records) < 2:
            continue
        ordered = sorted(group_records, key=lambda record: record.reward or 0.0)
        rejected = ordered[0]
        chosen = ordered[-1]
        if (chosen.reward or 0.0) - (rejected.reward or 0.0) < margin:
            continue
        rows.append(
            {
                "group_id": group_id,
                "chosen_episode_id": chosen.episode_id,
                "rejected_episode_id": rejected.episode_id,
                "chosen_reward": chosen.reward,
                "rejected_reward": rejected.reward,
                "prompt": to_training_prompt(chosen),
                "chosen": chosen.answer,
                "rejected": rejected.answer,
                "ground_truth": _ground_truth(chosen),
            }
        )
    return rows


def _reward_rows(records: list[EpisodeRecord]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        if not record.answer or record.reward is None:
            continue
        rows.append(
            {
                "episode_id": record.episode_id,
                "group_id": record.group_id,
                "split": record.split,
                "prompt": to_training_prompt(record),
                "completion": record.answer,
                "reward": record.reward,
                "breakdown": record.breakdown,
                "reasons": record.reasons,
                "ground_truth": _ground_truth(record),
            }
        )
    return rows


def _ground_truth(record: EpisodeRecord) -> dict:
    task = record.task
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "max_price": task.max_price,
        "must_have": task.must_have,
        "must_not_have": task.must_not_have,
        "allowed_domains": task.allowed_domains,
        "token_budget": task.token_budget,
        "require_cart": task.require_cart,
    }


def _reward_summary(records: list[EpisodeRecord]) -> dict:
    rewards = [record.reward for record in records if record.reward is not None]
    if not rewards:
        return {"count": 0, "avg": None, "min": None, "max": None}
    return {
        "count": len(rewards),
        "avg": round(sum(rewards) / len(rewards), 4),
        "min": min(rewards),
        "max": max(rewards),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CartScout episode records for small-scale RL training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Export SFT, preference, and reward-model artifacts.")
    prepare.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--min-sft-reward", type=float, default=0.75)
    prepare.add_argument("--min-eval-reward", type=float, default=0.75)
    prepare.add_argument("--preference-margin", type=float, default=0.10)
    prepare.set_defaults(func=_prepare)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
