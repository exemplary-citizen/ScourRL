from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import os
from dataclasses import dataclass
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

from cart_scout.structured_cdp import (
    StructuredAction,
    build_controller_prompt,
    compact_text,
    parse_structured_action,
    retailer_search_url,
)


LOGGER = logging.getLogger("cart_scout.structured_cdp")
CDP_PROTOCOL = "cdp/1.3"
INTERACTIVE_SELECTOR = "a, button, input, textarea, select, [role='button'], [contenteditable='true']"


@dataclass
class StructuredCDPAgent(Agent):
    model: str
    max_steps: int
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    trace_screenshots: bool = True

    async def __call__(self, run):
        from playwright.async_api import async_playwright

        cdp_url = _ws_to_http(run.client.binding(CDP_PROTOCOL).url)
        LOGGER.info("structured CDP harness attaching to %s", cdp_url)
        system_prompt = build_controller_prompt(run.prompt_text)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(cdp_url)
            try:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
                answer: str | None = None

                for step in range(1, self.max_steps + 1):
                    observation = await _observe_page(page)
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "step": step,
                                    "max_steps": self.max_steps,
                                    "observation": observation,
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
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        reasoning_effort=self.reasoning_effort,
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
                    result = await _execute_action(page, parsed.action, observation)
                    await _record_tool_step(run, page, parsed.action, result, include_screenshot=self.trace_screenshots)
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "previous_action": action_payload,
                                    "result": _compact_result(result),
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
    top_p: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
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
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
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
