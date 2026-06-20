from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from hud.agents.base import Agent
from hud.agents.types import AgentStep
from hud.eval.runtime import HUDRuntime
from hud.eval.taskset import Taskset
from hud.types import Step


LOGGER = logging.getLogger("cart_scout.browser_use")
CDP_PROTOCOL = "cdp/1.3"


@dataclass
class BrowserUseCDPAgent(Agent):
    provider: str
    model: str
    max_steps: int
    api_key: str | None = None
    base_url: str | None = None

    async def __call__(self, run):
        from browser_use import Agent as BrowserUseAgent
        from browser_use import Browser

        cdp_url = _ws_to_http(run.client.binding(CDP_PROTOCOL).url)
        LOGGER.info("browser-use attaching to %s", cdp_url)

        llm = _build_llm(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        browser: Any = Browser(cdp_url=cdp_url)
        agent = BrowserUseAgent(task=run.prompt_text, llm=llm, browser=browser)

        try:
            history: Any = await agent.run(max_steps=self.max_steps)
        except Exception as exc:
            LOGGER.exception("browser-use run failed")
            run.trace.status = "error"
            run.record(Step(source="system", error=str(exc)))
            return
        finally:
            with contextlib.suppress(Exception):
                await browser.stop()

        successful = history.is_successful()
        content = history.final_result() or ""
        run.trace.status = "error" if successful is False else "completed"
        run.trace.content = content
        run.trace.extra.update(
            {
                "harness": "browser-use-cdp",
                "provider": self.provider,
                "model": self.model,
                "is_successful": successful,
                "steps": history.number_of_steps(),
                "urls": history.urls(),
            }
        )
        run.record(
            AgentStep(
                content=content,
                done=history.is_done(),
                error=content if successful is False else None,
            )
        )


def _build_llm(provider: str, model: str, api_key: str | None, base_url: str | None):
    match provider:
        case "anthropic":
            from browser_use import ChatAnthropic

            return ChatAnthropic(model=model, api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), base_url=base_url)
        case "openai":
            from browser_use.llm.openai.chat import ChatOpenAI

            return ChatOpenAI(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        case "openai-like":
            from browser_use.llm.openai.like import ChatOpenAILike

            return ChatOpenAILike(
                model=model,
                api_key=api_key or os.getenv("OPENAI_LIKE_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=base_url or os.getenv("OPENAI_LIKE_BASE_URL"),
            )
        case "ollama":
            from browser_use.llm.ollama.chat import ChatOllama

            return ChatOllama(model=model, host=base_url or os.getenv("OLLAMA_HOST"))
        case "browser-use":
            from browser_use.llm.browser_use.chat import ChatBrowserUse

            return ChatBrowserUse(model=model, api_key=api_key or os.getenv("BROWSER_USE_API_KEY"), base_url=base_url)
        case _:
            raise ValueError(f"unsupported provider: {provider}")


def _ws_to_http(url: str) -> str:
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CartScout via Browser Use over HUD's CDP capability.")
    parser.add_argument("--task-id", default="usb-c-charger-30w-under-40")
    parser.add_argument("--provider", choices=["anthropic", "openai", "openai-like", "ollama", "browser-use"], default="anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--api-key-env", default=None, help="Environment variable to read for the selected provider API key.")
    parser.add_argument("--base-url", default=None, help="Provider base URL, or Ollama host for --provider ollama.")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--runtime-timeout", type=float, default=1800.0)
    parser.add_argument("--rollout-timeout", type=float, default=1800.0)
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s")
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None

    taskset = Taskset.from_file("tasks.py").filter([args.task_id])
    if len(taskset) != 1:
        raise SystemExit(f"task not found: {args.task_id}")

    agent = BrowserUseCDPAgent(
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        api_key=api_key,
        base_url=args.base_url,
    )
    job = await taskset.run(
        agent,
        runtime=HUDRuntime(run_timeout=args.runtime_timeout),
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
