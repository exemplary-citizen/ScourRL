from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from fireworks_rft.remote_dataset import build_remote_rows, write_jsonl


ROOT = Path(__file__).resolve().parent
MAIN_BASE_MODEL = "accounts/fireworks/models/qwen3-vl-32b-instruct"
POC_BASE_MODEL = "accounts/fireworks/models/qwen3-vl-8b-instruct"
DEFAULT_OUTPUT_MODEL = "cart-scout-qwen3-vl-32b-remote-grpo"
DEFAULT_DISPLAY_NAME = "CartScout Qwen3-VL-32B Remote Browser GRPO"
EVALUATOR_ENTRY = "fireworks_rft/remote_evaluator.py::test_cart_scout_remote_rollout"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build task-only CartScout remote-rollout rows and launch Fireworks RFT."
    )
    parser.add_argument("--train-tasks", default="data/shopping_train_1000.jsonl")
    parser.add_argument("--dataset-jsonl", default=".fireworks/remote_train.jsonl")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--base-model", default=MAIN_BASE_MODEL)
    parser.add_argument("--output-model", default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME)
    parser.add_argument(
        "--remote-base-url",
        default=os.getenv("EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL")
        or os.getenv("CART_SCOUT_REMOTE_BASE_URL"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", default="1e-4")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--temperature", default="0.7")
    parser.add_argument("--top-p", default="0.95")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--response-candidates-count", type=int, default=4)
    parser.add_argument(
        "--poc",
        action="store_true",
        help="Use a small proof-of-concept setup with Qwen3-VL-8B and 16 task rows.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Let eval-protocol run local validation. Requires the remote server to be reachable.",
    )
    args = parser.parse_args()
    _apply_poc_defaults(args)

    rows = build_remote_rows(ROOT / args.train_tasks, limit=args.train_limit)
    write_jsonl(rows, ROOT / args.dataset_jsonl)
    print(f"wrote {len(rows)} remote task rows to {args.dataset_jsonl}")

    if args.skip_train:
        return

    if not _has_fireworks_key():
        raise SystemExit(
            "FIREWORKS_API_KEY is required. Set it in the shell or in .env before launching RFT."
        )
    if not args.remote_base_url:
        raise SystemExit(
            "CART_SCOUT_REMOTE_BASE_URL or --remote-base-url is required. "
            "It must be a public URL where Fireworks can reach fireworks_rft.remote_server:app."
        )

    env = os.environ.copy()
    env["CART_SCOUT_REMOTE_BASE_URL"] = args.remote_base_url.rstrip("/")
    env["EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL"] = args.remote_base_url.rstrip("/")
    env["CART_SCOUT_REMOTE_MODEL"] = args.base_model

    command = [
        sys.executable,
        "-m",
        "eval_protocol",
        "create",
        "rft",
        "--base-model",
        args.base_model,
        "--dataset-jsonl",
        args.dataset_jsonl,
        "--eval-auto-carveout",
        "--output-model",
        args.output_model,
        "--display-name",
        args.display_name,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--lora-rank",
        str(args.lora_rank),
        "--chunk-size",
        str(args.chunk_size),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--response-candidates-count",
        str(args.response_candidates_count),
        "--ignore-docker",
        "--yes",
        "--force",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if not args.run_validation:
        command.append("--skip-validation")
    _run(command, env=env)


def _apply_poc_defaults(args: argparse.Namespace) -> None:
    if not args.poc:
        return
    if args.base_model == MAIN_BASE_MODEL:
        args.base_model = POC_BASE_MODEL
    if args.output_model == DEFAULT_OUTPUT_MODEL:
        args.output_model = "cart-scout-qwen3-vl-remote-poc-grpo"
    if args.display_name == DEFAULT_DISPLAY_NAME:
        args.display_name = "CartScout Qwen3-VL Remote POC GRPO"
    if args.train_limit is None:
        args.train_limit = 16
    if args.chunk_size == 50:
        args.chunk_size = 4
    if args.response_candidates_count == 4:
        args.response_candidates_count = 2
    if args.dataset_jsonl == ".fireworks/remote_train.jsonl":
        args.dataset_jsonl = ".fireworks/remote_poc_train.jsonl"


def _has_fireworks_key() -> bool:
    if os.getenv("FIREWORKS_API_KEY"):
        return True
    env_path = ROOT / ".env"
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "FIREWORKS_API_KEY" and value.strip():
            return True
    return False


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
