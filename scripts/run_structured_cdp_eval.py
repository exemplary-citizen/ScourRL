from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import mcp.types as mcp_types
from hud.agents.base import Agent
from hud.agents.types import AgentStep, ToolStep
from hud.eval.runtime import HUDRuntime
from hud.eval.runtime import Runtime as TCPRuntime
from hud.eval.taskset import Taskset
from hud.types import MCPToolCall, MCPToolResult, Step

from cart_scout.schema import ShoppingTaskSpec
from cart_scout.structured_cdp import (
    StructuredAction,
    build_controller_prompt,
    compact_text,
    observation_progress,
    parse_structured_action,
    retailer_search_url,
    shaping_reward,
)


LOGGER = logging.getLogger("cart_scout.structured_cdp")
CDP_PROTOCOL = "cdp/1.3"
INTERACTIVE_SELECTOR = "a, button, input, textarea, select, [role='button'], [contenteditable='true']"


@dataclass
class _RunStats:
    visited_urls: set[str] = field(default_factory=set)
    last_action_key: str | None = None
    repeated_actions: int = 0
    no_op_actions: int = 0
    unsafe_attempts: int = 0

    def note_action(self, action: StructuredAction) -> bool:
        action_key = json.dumps(action.model_dump(mode="json"), sort_keys=True)
        repeated = action_key == self.last_action_key
        if repeated:
            self.repeated_actions += 1
        self.last_action_key = action_key
        return repeated

    def note_result(self, action: StructuredAction, result: dict[str, Any], *, before_url: str) -> None:
        url = str(result.get("url") or before_url or "")
        if url:
            self.visited_urls.add(url)
        error = str(result.get("error") or "").lower()
        if "blocked unsafe" in error or "checkout" in error or "payment" in error or "sign in" in error:
            self.unsafe_attempts += 1
        if not result.get("ok"):
            self.no_op_actions += 1
        if action.action == "scroll":
            scroll = result.get("scroll") if isinstance(result.get("scroll"), dict) else {}
            if scroll.get("y") == 0 and str(action.args.get("direction", "down")).lower() == "up":
                self.no_op_actions += 1


@dataclass
class StructuredCDPAgent(Agent):
    model: str
    max_steps: int
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    trace_screenshots: bool = True

    async def __call__(self, run):
        from playwright.async_api import async_playwright

        cdp_url = _ws_to_http(run.client.binding(CDP_PROTOCOL).url)
        LOGGER.info("structured CDP harness attaching to %s", cdp_url)
        system_prompt = build_controller_prompt(run.prompt_text)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        task_spec = _task_spec_from_run(run)
        run_stats = _RunStats()
        shaping_rows: list[dict[str, Any]] = []

        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(cdp_url)
            try:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
                answer: str | None = None

                for step in range(1, self.max_steps + 1):
                    observation = await _observe_page(page)
                    progress_before = observation_progress(
                        observation,
                        task_spec,
                        unsafe_attempts=run_stats.unsafe_attempts,
                        repeated_actions=run_stats.repeated_actions,
                        no_op_actions=run_stats.no_op_actions,
                        visited_urls=len(run_stats.visited_urls),
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "step": step,
                                    "max_steps": self.max_steps,
                                    "observation": observation,
                                    "progress": _progress_payload(progress_before),
                                    "instruction": "Return exactly one action JSON object.",
                                },
                                ensure_ascii=True,
                            ),
                        }
                    )
                    raw = await _chat_completion(
                        _message_window(messages),
                        model=self.model,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        temperature=self.temperature,
                    )
                    parsed = parse_structured_action(raw)
                    action_payload = parsed.action.model_dump(mode="json") if parsed.action else None
                    run.record(
                        AgentStep(
                            content=json.dumps(
                                {
                                    "harness": "structured-cdp",
                                    "step": step,
                                    "observation": _observation_summary(observation),
                                    "progress": _progress_payload(progress_before),
                                    "raw_model_output": raw,
                                    "action": action_payload,
                                    "parse_error": parsed.error,
                                },
                                indent=2,
                                ensure_ascii=True,
                            ),
                            tool_calls=[
                                MCPToolCall(
                                    name=f"structured_cdp.{parsed.action.action if parsed.action else 'parse_error'}",
                                    arguments=action_payload or {"raw": raw, "error": parsed.error},
                                )
                            ],
                            done=False,
                        )
                    )
                    if parsed.action is None:
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "error": parsed.error,
                                        "instruction": "Your previous response was invalid. Return only one JSON action object.",
                                    },
                                    ensure_ascii=True,
                                ),
                            }
                        )
                        continue

                    messages.append({"role": "assistant", "content": raw})
                    repeated = run_stats.note_action(parsed.action)
                    result = await _execute_action(page, parsed.action, observation)
                    run_stats.note_result(parsed.action, result, before_url=str(observation.get("url") or ""))
                    next_observation = await _observe_page(page)
                    progress_after = observation_progress(
                        next_observation,
                        task_spec,
                        unsafe_attempts=run_stats.unsafe_attempts,
                        repeated_actions=run_stats.repeated_actions,
                        no_op_actions=run_stats.no_op_actions,
                        visited_urls=len(run_stats.visited_urls),
                    )
                    shape = shaping_reward(
                        step=step,
                        action_name=parsed.action.action,
                        previous=progress_before,
                        current=progress_after,
                    )
                    shape_payload = _shaping_payload(shape, repeated=repeated)
                    shaping_rows.append(shape_payload)
                    result["progress"] = {
                        "before": _progress_payload(progress_before),
                        "after": _progress_payload(progress_after),
                        "shaping": shape_payload,
                    }
                    await _record_tool_step(run, page, parsed.action, result, include_screenshot=self.trace_screenshots)
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "previous_action": action_payload,
                                    "result": _compact_result(result),
                                    "progress": shape_payload,
                                    "instruction": "Use the result to choose the next JSON action.",
                                },
                                ensure_ascii=True,
                            ),
                        }
                    )

                    if parsed.action.action == "emit_packet":
                        packet = parsed.action.args.get("packet")
                        answer = json.dumps(packet, ensure_ascii=True) if isinstance(packet, dict) else json.dumps(parsed.action.args)
                        break
                    if parsed.action.action == "stop":
                        answer = json.dumps(
                            {
                                "stopped": True,
                                "reason": parsed.action.args.get("reason", "stopped"),
                            },
                            ensure_ascii=True,
                        )
                        break

                if answer is None:
                    answer = ""
                run.trace.status = "completed" if answer else "error"
                run.trace.content = answer
                run.trace.extra.update(
                    {
                        "harness": "structured-cdp",
                        "model": self.model,
                        "steps": step,
                        "url": page.url,
                        "progress": _progress_payload(
                            observation_progress(
                                await _observe_page(page),
                                task_spec,
                                unsafe_attempts=run_stats.unsafe_attempts,
                                repeated_actions=run_stats.repeated_actions,
                                no_op_actions=run_stats.no_op_actions,
                                visited_urls=len(run_stats.visited_urls),
                            )
                        ),
                        "shaping_rows": shaping_rows,
                        "dense_reward_sum": round(sum(row["dense_reward"] for row in shaping_rows), 6),
                    }
                )
                run.record(AgentStep(content=answer, done=True, error=None if answer else "no answer emitted"))
            except Exception as exc:
                LOGGER.exception("structured CDP run failed")
                run.trace.status = "error"
                run.trace.content = json.dumps({"error": str(exc)}, ensure_ascii=True)
                run.record(Step(source="system", error=str(exc)))
            finally:
                await browser.close()


async def _chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
) -> str:
    endpoint = (base_url or os.getenv("OPENAI_LIKE_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    token = api_key or os.getenv("OPENAI_LIKE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise
                if attempt == 2:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _message_window(messages: list[dict[str, str]], max_tail: int = 6) -> list[dict[str, str]]:
    if len(messages) <= max_tail + 1:
        return messages
    return [messages[0], *messages[-max_tail:]]


def _task_spec_from_run(run: Any) -> ShoppingTaskSpec:
    prompt = str(getattr(run, "prompt_text", "") or "")
    return ShoppingTaskSpec(
        task_id=_match_text(prompt, r"Task:\s*(.*?)\n\nConstraints:", default="structured-cdp-task"),
        instruction=_match_text(prompt, r"Task:\s*(.*?)\n\nConstraints:", default=prompt[:240]),
        max_price=float(_match_text(prompt, r"Max price:\s*\$([0-9]+(?:\.[0-9]+)?)", default="0") or 0),
        must_have=_match_list(prompt, "Must have"),
        must_not_have=_match_list(prompt, "Must not have"),
        allowed_domains=_match_list(prompt, "Allowed domains") or ["target.com", "amazon.com"],
        require_cart=_match_text(prompt, r"Cart prep required:\s*(True|False)", default="False") == "True",
    )


def _match_text(text: str, pattern: str, *, default: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return default
    return match.group(1).strip()


def _match_list(text: str, label: str) -> list[str]:
    match = re.search(rf"{re.escape(label)}:\s*(\[.*?\])", text)
    if not match:
        return []
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _progress_payload(progress: Any) -> dict[str, Any]:
    return {
        "score": round(progress.score, 6),
        "allowed_domain": progress.allowed_domain,
        "price_found": progress.price_found,
        "must_have_hits": list(progress.must_have_hits),
        "evidence_like": progress.evidence_like,
        "unsafe_attempts": progress.unsafe_attempts,
        "repeated_actions": progress.repeated_actions,
        "no_op_actions": progress.no_op_actions,
        "visited_urls": progress.visited_urls,
    }


def _shaping_payload(shape: Any, *, repeated: bool) -> dict[str, Any]:
    return {
        "step": shape.step,
        "action": shape.action,
        "previous_score": round(shape.previous_score, 6),
        "next_score": round(shape.next_score, 6),
        "dense_reward": round(shape.dense_reward, 6),
        "reasons": list(shape.reasons),
        "repeated": repeated,
        "state": _progress_payload(shape.state),
    }


async def _observe_page(page: Any) -> dict[str, Any]:
    with contextlib.suppress(Exception):
        await page.wait_for_load_state("domcontentloaded", timeout=3000)
    refs = await page.evaluate(
        """
        (selector) => Array.from(document.querySelectorAll(selector))
          .map((el, selectorIndex) => ({
            selector_index: selectorIndex,
            tag: el.tagName.toLowerCase(),
            type: el.getAttribute('type') || null,
            text: (el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('name') || el.value || '').replace(/\\s+/g, ' ').trim().slice(0, 180),
            href: el.href || null,
            visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length)
          }))
          .filter(item => item.visible && (item.text || item.href))
          .slice(0, 40)
          .map((item, ref) => ({...item, ref}))
        """,
        INTERACTIVE_SELECTOR,
    )
    scroll = await page.evaluate(
        """
        () => ({
          x: window.scrollX,
          y: window.scrollY,
          viewport_width: window.innerWidth,
          viewport_height: window.innerHeight,
          page_width: document.documentElement.scrollWidth,
          page_height: document.documentElement.scrollHeight
        })
        """
    )
    text = await page.evaluate("() => document.body ? document.body.innerText : ''")
    return {
        "url": page.url,
        "title": await page.title(),
        "scroll": scroll,
        "text": compact_text(text, limit=2500),
        "refs": refs,
    }


async def _execute_action(page: Any, action: StructuredAction, observation: dict[str, Any]) -> dict[str, Any]:
    match action.action:
        case "open_url":
            url = str(action.args.get("url", ""))
            if not _safe_url(url):
                return {"ok": False, "error": f"blocked unsafe URL: {url}"}
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            return {"ok": True, "url": page.url}
        case "search_retailer":
            url = retailer_search_url(str(action.args.get("retailer", "")), str(action.args.get("query", "")))
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            return {"ok": True, "url": page.url}
        case "click_ref":
            ref = int(action.args.get("ref"))
            refs = observation.get("refs", [])
            if ref < 0 or ref >= len(refs):
                return {"ok": False, "error": f"ref out of range: {ref}"}
            item = refs[ref]
            href = item.get("href")
            text = str(item.get("text", ""))
            if _blocked_click(text, str(href or "")):
                return {"ok": False, "error": f"blocked unsafe click target: {text or href}"}
            if href and _safe_url(str(href)):
                await page.goto(href, wait_until="domcontentloaded", timeout=20000)
            else:
                await page.locator(INTERACTIVE_SELECTOR).nth(int(item["selector_index"])).click(timeout=8000)
            return {"ok": True, "clicked": item, "url": page.url}
        case "fill_ref":
            ref = int(action.args.get("ref"))
            refs = observation.get("refs", [])
            if ref < 0 or ref >= len(refs):
                return {"ok": False, "error": f"ref out of range: {ref}"}
            text = str(action.args.get("text", ""))
            item = refs[ref]
            if _blocked_click(str(item.get("text", "")), str(item.get("href") or "")):
                return {"ok": False, "error": f"blocked unsafe fill target: {item.get('text') or item.get('href')}"}
            locator = page.locator(INTERACTIVE_SELECTOR).nth(int(item["selector_index"]))
            await locator.fill(text, timeout=8000)
            return {"ok": True, "filled": item, "text_length": len(text), "url": page.url}
        case "press":
            key = str(action.args.get("key", "Enter"))
            await page.keyboard.press(key)
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            return {"ok": True, "key": key, "url": page.url}
        case "scroll":
            direction = str(action.args.get("direction", "down")).lower()
            amount = int(action.args.get("amount", 700))
            delta = -abs(amount) if direction == "up" else abs(amount)
            await page.mouse.wheel(0, delta)
            await page.wait_for_timeout(500)
            scroll = await page.evaluate(
                "() => ({x: window.scrollX, y: window.scrollY, page_height: document.documentElement.scrollHeight, viewport_height: window.innerHeight})"
            )
            return {"ok": True, "direction": direction, "amount": amount, "scroll": scroll, "url": page.url}
        case "go_back":
            response = await page.go_back(wait_until="domcontentloaded", timeout=10000)
            return {"ok": True, "url": page.url, "status": response.status if response else None}
        case "extract_page":
            return {"ok": True, "url": page.url, "title": await page.title(), "text": compact_text(await page.locator("body").inner_text(timeout=5000), 3000)}
        case "find_text":
            pattern = str(action.args.get("pattern", ""))
            haystack = observation.get("text", "")
            index = haystack.lower().find(pattern.lower())
            return {"ok": True, "pattern": pattern, "found": index >= 0, "index": index}
        case "screenshot":
            return {"ok": True, "url": page.url, "screenshot_attached": True}
        case "emit_packet":
            return {"ok": True, "emitted": True}
        case "stop":
            return {"ok": True, "stopped": True, "reason": action.args.get("reason")}


async def _record_tool_step(
    run: Any,
    page: Any,
    action: StructuredAction,
    result: dict[str, Any],
    *,
    include_screenshot: bool,
) -> None:
    call = MCPToolCall(name=f"structured_cdp.{action.action}", arguments=action.model_dump(mode="json"))
    content: list[mcp_types.ContentBlock] = [
        mcp_types.TextContent(type="text", text=json.dumps(_compact_result(result), indent=2, ensure_ascii=True))
    ]
    if include_screenshot:
        with contextlib.suppress(Exception):
            png = await page.screenshot(full_page=False, timeout=5000)
            content.append(
                mcp_types.ImageContent(
                    type="image",
                    mimeType="image/png",
                    data=base64.b64encode(png).decode("ascii"),
                )
            )
    run.record(
        ToolStep(
            call=call,
            result=MCPToolResult(call_id=call.id, content=content, isError=not bool(result.get("ok", False))),
        )
    )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(result)
    if isinstance(compacted.get("text"), str):
        compacted["text"] = compact_text(compacted["text"], 1500)
    return compacted


def _safe_url(url: str) -> bool:
    if not url.startswith("https://"):
        return False
    return not _blocked_click("", url)


def _blocked_click(text: str, href: str) -> bool:
    target = f"{text} {href}".lower()
    blocked_terms = (
        "signin",
        "sign in",
        "login",
        "account",
        "register",
        "cart",
        "checkout",
        "payment",
        "orders",
        "buy now",
    )
    return any(term in target for term in blocked_terms)


def _observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": observation.get("url"),
        "title": observation.get("title"),
        "scroll": observation.get("scroll"),
        "text": compact_text(str(observation.get("text", "")), 800),
        "refs": observation.get("refs", [])[:20],
    }


def _ws_to_http(url: str) -> str:
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CartScout through a fixed structured CDP action space.")
    parser.add_argument("--task-id", default="usb-c-charger-30w-under-40")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.1")
    parser.add_argument("--api-key-env", default="HUD_API_KEY")
    parser.add_argument("--base-url", default="https://inference.beta.hud.ai")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runtime-timeout", type=float, default=1800.0)
    parser.add_argument("--rollout-timeout", type=float, default=1800.0)
    parser.add_argument("--no-trace-screenshots", action="store_true")
    parser.add_argument("--runtime", choices=["hud", "tcp"], default="hud")
    parser.add_argument("--runtime-url", default="tcp://127.0.0.1:8765")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s")
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None

    taskset = Taskset.from_file("tasks.py").filter([args.task_id])
    if len(taskset) != 1:
        raise SystemExit(f"task not found: {args.task_id}")

    agent = StructuredCDPAgent(
        model=args.model,
        max_steps=args.max_steps,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        trace_screenshots=not args.no_trace_screenshots,
    )
    runtime = (
        HUDRuntime(run_timeout=args.runtime_timeout)
        if args.runtime == "hud"
        else TCPRuntime(args.runtime_url)
    )
    job = await taskset.run(
        agent,
        runtime=runtime,
        max_concurrent=1,
        rollout_timeout=args.rollout_timeout,
    )
    print(f"job: https://hud.ai/jobs/{job.id}")
    for run in job.runs:
        print(
            {
                "slug": run.slug,
                "reward": run.reward,
                "status": run.trace.status,
                "answer": (run.trace.content or "")[:1000],
                "extra": run.trace.extra,
            }
        )


if __name__ == "__main__":
    asyncio.run(main())
