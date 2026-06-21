from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from eval_protocol import InitRequest, RolloutIdFilter, Status
from eval_protocol.models import Message
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from cart_scout.reward import RewardResult, score_grpo_packet
from cart_scout.schema import PurchasePacket, ShoppingTaskSpec


LOGGER = logging.getLogger("cart_scout.remote_rollout")
REMOTE_MODE_ENV = "CART_SCOUT_REMOTE_MODE"

app = FastAPI(title="CartScout Remote Browser RFT")


@dataclass
class RolloutRecord:
    rollout_id: str
    task: ShoppingTaskSpec
    final_answer: str
    reward: RewardResult
    messages: list[dict[str, Any]]
    terminated: bool = True
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutOutput:
    final_answer: str
    extra: dict[str, Any] = field(default_factory=dict)


_ROLLOUTS: dict[str, RolloutRecord] = {}


@app.post("/init")
async def init(request: InitRequest) -> dict[str, Any]:
    """Fireworks RemoteRolloutProcessor entrypoint.

    Fireworks posts one task rollout here. The production path should run the live
    HUD/CDP browser agent and return its final PurchasePacket. The default `stub`
    mode exists only for protocol smoke tests.
    """
    rollout_id = request.metadata.rollout_id
    logger = _rollout_logger(rollout_id)
    logger.info("starting CartScout remote rollout")

    try:
        task = task_from_messages(request.messages or [])
        rollout = await run_rollout(request, task)
        reward = score_grpo_packet(rollout.final_answer, task)
        messages = _messages_with_final_answer(request.messages or [], rollout.final_answer)
        record = RolloutRecord(
            rollout_id=rollout_id,
            task=task,
            final_answer=rollout.final_answer,
            reward=reward,
            messages=messages,
            extra={"mode": os.getenv(REMOTE_MODE_ENV, "stub"), **rollout.extra},
        )
        _ROLLOUTS[rollout_id] = record
        logger.info(
            "rollout completed",
            extra={"status": Status.rollout_finished()},
        )
        return _status_payload(record)
    except Exception as exc:
        logger.exception("rollout failed")
        fallback_task = _fallback_task()
        record = RolloutRecord(
            rollout_id=rollout_id,
            task=fallback_task,
            final_answer="",
            reward=RewardResult(0.0, {}, [str(exc)]),
            messages=[_message_to_dict(message) for message in request.messages or []],
            terminated=True,
            error=str(exc),
        )
        _ROLLOUTS[rollout_id] = record
        logger.error(
            "rollout errored",
            extra={"status": Status.rollout_error(str(exc))},
        )
        return JSONResponse(status_code=500, content=_status_payload(record))


@app.get("/status")
async def status(rollout_id: str) -> dict[str, Any]:
    record = _ROLLOUTS.get(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown rollout_id: {rollout_id}")
    return _status_payload(record)


async def run_rollout(request: InitRequest, task: ShoppingTaskSpec) -> RolloutOutput:
    mode = os.getenv(REMOTE_MODE_ENV, "stub")
    if mode == "stub":
        return RolloutOutput(final_answer=_stub_purchase_packet(task).model_dump_json())
    if mode == "structured-cdp":
        return await _run_hud_structured_cdp_rollout(request, task)
    if mode == "browser-use":
        return await _run_hud_browser_use_rollout(request, task)
    raise ValueError(f"unsupported {REMOTE_MODE_ENV}: {mode}")


async def _run_hud_structured_cdp_rollout(
    request: InitRequest,
    task: ShoppingTaskSpec,
) -> RolloutOutput:
    from hud.eval.taskset import Taskset

    from scripts.run_structured_cdp_eval import StructuredCDPAgent

    completion_params = request.completion_params or {}
    model = str(
        completion_params.get("model")
        or os.getenv("CART_SCOUT_REMOTE_MODEL")
        or "accounts/fireworks/models/qwen3-vl-32b-instruct"
    )
    max_steps = _env_int("CART_SCOUT_STRUCTURED_MAX_STEPS", 12)
    runtime_timeout = _env_float("CART_SCOUT_HUD_RUNTIME_TIMEOUT", 1800.0)
    rollout_timeout = _env_float("CART_SCOUT_HUD_ROLLOUT_TIMEOUT", runtime_timeout)
    runtime_kind = os.getenv("CART_SCOUT_HUD_RUNTIME", "hud")
    runtime_url = os.getenv("CART_SCOUT_HUD_RUNTIME_URL", "tcp://127.0.0.1:8765")
    model_base_url = _model_base_url(request)
    api_key = request.api_key or os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_LIKE_API_KEY")

    agent = StructuredCDPAgent(
        model=model,
        max_steps=max_steps,
        api_key=api_key,
        base_url=model_base_url,
        temperature=_float_or_none(completion_params.get("temperature")) or 0.0,
        top_p=_float_or_none(completion_params.get("top_p")),
        max_tokens=_int_or_none(
            completion_params.get("max_tokens")
            or completion_params.get("max_completion_tokens")
            or completion_params.get("max_output_tokens")
        ),
        reasoning_effort=os.getenv("CART_SCOUT_STRUCTURED_REASONING_EFFORT", "none"),
        trace_screenshots=_env_bool("CART_SCOUT_STRUCTURED_TRACE_SCREENSHOTS", False),
    )
    job = await Taskset("cart-scout-fireworks-structured-cdp", [_hud_task_from_spec(task)]).run(
        agent,
        runtime=_hud_runtime(runtime_kind, runtime_timeout, runtime_url),
        max_concurrent=1,
        rollout_timeout=rollout_timeout,
    )
    return _rollout_output_from_hud_job(
        job,
        harness="structured-cdp",
        runtime_kind=runtime_kind,
        model=model,
        max_steps=max_steps,
    )


async def _run_hud_browser_use_rollout(
    request: InitRequest,
    task: ShoppingTaskSpec,
) -> RolloutOutput:
    from hud.eval.taskset import Taskset

    from scripts.run_browser_use_eval import BrowserUseCDPAgent

    completion_params = request.completion_params or {}
    provider = os.getenv("CART_SCOUT_BROWSER_PROVIDER", "openai-like")
    model = str(
        completion_params.get("model")
        or os.getenv("CART_SCOUT_REMOTE_MODEL")
        or "accounts/fireworks/models/qwen3-vl-32b-instruct"
    )
    max_steps = _env_int("CART_SCOUT_BROWSER_MAX_STEPS", 35)
    runtime_timeout = _env_float("CART_SCOUT_HUD_RUNTIME_TIMEOUT", 1800.0)
    rollout_timeout = _env_float("CART_SCOUT_HUD_ROLLOUT_TIMEOUT", runtime_timeout)
    runtime_kind = os.getenv("CART_SCOUT_HUD_RUNTIME", "hud")
    runtime_url = os.getenv("CART_SCOUT_HUD_RUNTIME_URL", "tcp://127.0.0.1:8765")
    model_base_url = _model_base_url(request)
    api_key = request.api_key or os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_LIKE_API_KEY")
    hud_task = _hud_task_from_spec(task)
    agent = BrowserUseCDPAgent(
        provider=provider,
        model=model,
        max_steps=max_steps,
        api_key=api_key,
        base_url=model_base_url,
        llm_temperature=_float_or_none(completion_params.get("temperature")),
        llm_top_p=_float_or_none(completion_params.get("top_p")),
        max_completion_tokens=_int_or_none(
            completion_params.get("max_tokens")
            or completion_params.get("max_completion_tokens")
            or completion_params.get("max_output_tokens")
        ),
        reasoning_effort=os.getenv("CART_SCOUT_BROWSER_REASONING_EFFORT", "none"),
        use_vision=_env_bool("CART_SCOUT_BROWSER_USE_VISION", True),
        use_thinking=_env_bool("CART_SCOUT_BROWSER_USE_THINKING", False),
        flash_mode=_env_bool("CART_SCOUT_BROWSER_FLASH_MODE", True),
        trace_screenshots=_env_bool("CART_SCOUT_BROWSER_TRACE_SCREENSHOTS", False),
        generate_gif=_env_bool("CART_SCOUT_BROWSER_GENERATE_GIF", False),
        rfb_watch_interval=_env_float("CART_SCOUT_BROWSER_RFB_WATCH_INTERVAL", 0.0),
    )
    job = await Taskset("cart-scout-fireworks-remote", [hud_task]).run(
        agent,
        runtime=_hud_runtime(runtime_kind, runtime_timeout, runtime_url),
        max_concurrent=1,
        rollout_timeout=rollout_timeout,
    )
    output = _rollout_output_from_hud_job(
        job,
        harness="browser-use-cdp",
        runtime_kind=runtime_kind,
        model=model,
        max_steps=max_steps,
    )
    output.extra["provider"] = provider
    return output


def _model_base_url(request: InitRequest) -> str:
    return (
        request.model_base_url
        or os.getenv("CART_SCOUT_MODEL_BASE_URL")
        or os.getenv("OPENAI_LIKE_BASE_URL")
        or "https://api.fireworks.ai/inference/v1"
    )


def _hud_task_from_spec(task: ShoppingTaskSpec):
    from hud.eval.task import Task

    return Task(
        env="cart-scout",
        id="shopping-context",
        slug=task.task_id,
        args={
            "task_id": task.task_id,
            "instruction": task.instruction,
            "max_price": task.max_price,
            "must_have": task.must_have,
            "must_not_have": task.must_not_have,
            "allowed_domains": task.allowed_domains,
            "token_budget": task.token_budget,
            "require_cart": task.require_cart,
        },
    )


def _hud_runtime(runtime_kind: str, runtime_timeout: float, runtime_url: str):
    from hud.eval.runtime import HUDRuntime, Runtime as TCPRuntime

    if runtime_kind not in {"hud", "tcp"}:
        raise ValueError("CART_SCOUT_HUD_RUNTIME must be 'hud' or 'tcp'")
    return HUDRuntime(run_timeout=runtime_timeout) if runtime_kind == "hud" else TCPRuntime(runtime_url)


def _rollout_output_from_hud_job(
    job: Any,
    *,
    harness: str,
    runtime_kind: str,
    model: str,
    max_steps: int,
) -> RolloutOutput:
    if not job.runs:
        raise RuntimeError("HUD rollout returned no runs")
    run = job.runs[0]
    final_answer = run.trace.content or ""
    if not final_answer:
        raise RuntimeError("HUD rollout completed without a final PurchasePacket")
    return RolloutOutput(
        final_answer=final_answer,
        extra={
            "hud_job_id": job.id,
            "hud_reward": run.reward,
            "hud_status": run.trace.status,
            "hud_trace_id": run.trace_id,
            "hud_trace_extra": run.trace.extra,
            "hud_runtime": runtime_kind,
            "harness": harness,
            "model": model,
            "max_steps": max_steps,
        },
    )


def task_from_messages(messages: list[Message]) -> ShoppingTaskSpec:
    for message in reversed(messages):
        content = message.content
        if not content:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=True)
        data = _extract_json_object(content)
        if data is None:
            continue
        if "task" in data and isinstance(data["task"], dict):
            data = data["task"]
        try:
            return ShoppingTaskSpec.model_validate(data)
        except Exception:
            continue
    raise ValueError("no ShoppingTaskSpec JSON found in rollout messages")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _stub_purchase_packet(task: ShoppingTaskSpec) -> PurchasePacket:
    retailer_domain = task.allowed_domains[0]
    retailer = _retailer_name(retailer_domain)
    title = _title_from_task(task)
    price = max(1.0, round(task.max_price * 0.7, 2))
    url = f"https://www.{retailer_domain}/p/{_slug(task.task_id)}-stub"
    evidence_quote = f"{title} includes {' and '.join(task.must_have or ['requested item'])}"
    return PurchasePacket(
        query=task.instruction,
        recommended_product=title,
        retailer=retailer,
        url=url,
        price=f"${price:.2f}",
        delivery_or_pickup="Availability must be verified in live browser rollout",
        seller=retailer,
        constraints_met=list(task.must_have),
        constraints_uncertain=[],
        evidence=[
            {
                "url": url,
                "quote": evidence_quote,
                "supports": "required constraints",
            },
            {
                "url": url,
                "quote": f"Price ${price:.2f}, under ${task.max_price:g} budget",
                "supports": "price under budget",
            },
        ],
        recommendation=(
            f"Recommend this option because it matches the required constraints and is under "
            f"the ${task.max_price:g} budget."
        ),
        stop_before_checkout=True,
        cart_prepared=False,
        attempted_checkout=False,
    )


def _messages_with_final_answer(messages: list[Message], final_answer: str) -> list[dict[str, Any]]:
    return [
        *[_message_to_dict(message) for message in messages],
        {"role": "assistant", "content": final_answer},
    ]


def _message_to_dict(message: Message) -> dict[str, Any]:
    if hasattr(message, "dump_mdoel_for_chat_completion_request"):
        return message.dump_mdoel_for_chat_completion_request()
    return message.model_dump(exclude_none=True, mode="json")


def _status_payload(record: RolloutRecord) -> dict[str, Any]:
    return {
        "status": "error" if record.error else "success",
        "terminated": record.terminated,
        "rollout_id": record.rollout_id,
        "reward": record.reward.score,
        "breakdown": record.reward.breakdown,
        "reasons": record.reward.reasons,
        "final_answer": record.final_answer,
        "messages": record.messages,
        "task": record.task.model_dump(mode="json"),
        "error": record.error,
        "extra": record.extra,
    }


def _rollout_logger(rollout_id: str) -> logging.Logger:
    logger = logging.getLogger(f"cart_scout.remote_rollout.{rollout_id}")
    if not any(isinstance(filter_, RolloutIdFilter) for filter_ in logger.filters):
        logger.addFilter(RolloutIdFilter(rollout_id))
    return logger


def _fallback_task() -> ShoppingTaskSpec:
    return ShoppingTaskSpec(
        task_id="invalid-rollout",
        instruction="Invalid rollout",
        allowed_domains=["example.com"],
        max_price=1.0,
    )


def _title_from_task(task: ShoppingTaskSpec) -> str:
    if task.must_have:
        return " ".join(task.must_have[:3]).title()
    text = task.instruction.split(" under ", 1)[0]
    text = re.sub(r"^Find an? ", "", text, flags=re.IGNORECASE)
    return text[:80].strip(" .") or "Shopping Item"


def _retailer_name(domain: str) -> str:
    base = domain.split(".")[-2] if "." in domain else domain
    return {
        "amazon": "Amazon",
        "target": "Target",
        "walmart": "Walmart",
        "homedepot": "The Home Depot",
        "bestbuy": "Best Buy",
    }.get(base, base.replace("-", " ").title())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
