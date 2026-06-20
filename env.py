"""HUD v6 real-browser environment for CartScout.

The agent receives a real Chromium exposed as both CDP and RFB/VNC. It browses public
shopping pages, then returns a PurchasePacket JSON. Grading reads only that packet and
deterministic task constraints; it does not reward raw clicks.
"""

# Do not add `from __future__ import annotations`; HUD manifest parsing can inspect annotations.
import asyncio
import logging
import os
import pwd
import shutil
import socket
import sys
from urllib.parse import quote_plus

from hud import Environment
from hud.capabilities import Capability

from cart_scout.reward import score_purchase_packet
from cart_scout.schema import ShoppingTaskSpec

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

env = Environment(name="cart-scout")

_HOST = "127.0.0.1"
_VNC_PORT = 5900
_CDP_PORT = 9222
_DESKTOP_USER = "ubuntu"
_procs: "list[asyncio.subprocess.Process]" = []


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


async def _listening(host: str, port: int, what: str, timeout: float = 60.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if _port_open(host, port):
            return
        await asyncio.sleep(0.3)
    raise RuntimeError(f"{what} never came up on {host}:{port}")


def _drop_to_ubuntu() -> bool:
    if os.geteuid() != 0:
        return False
    try:
        pwd.getpwnam(_DESKTOP_USER)
    except KeyError:
        return False
    return True


async def _spawn(*cmd: str, quiet: bool = True) -> asyncio.subprocess.Process:
    drop = _drop_to_ubuntu()
    keep = {
        "HOME": "/home/ubuntu" if drop else os.environ.get("HOME", "/root"),
        "DISPLAY": ":1",
        "BROWSEROS_BIN": os.environ.get("BROWSEROS_BIN", "/usr/bin/browseros"),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    argv = (
        ["sudo", "-u", _DESKTOP_USER, "env", *[f"{k}={v}" for k, v in keep.items()], *cmd]
        if drop
        else list(cmd)
    )
    return await asyncio.create_subprocess_exec(
        *argv,
        env={**os.environ, **keep},
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL if quiet else None,
    )


async def _start_browser() -> None:
    _procs.append(await _spawn("Xvfb", ":1", "-screen", "0", "1280x800x24"))
    await asyncio.sleep(0.8)
    _procs.append(
        await _spawn(
            "x11vnc",
            "-display",
            ":1",
            "-rfbport",
            str(_VNC_PORT),
            "-forever",
            "-shared",
            "-nopw",
            "-localhost",
        )
    )
    if shutil.which("websockify"):
        _procs.append(await _spawn("websockify", "--web", "/usr/share/novnc", "8080", "localhost:5900"))
    _procs.append(
        await _spawn(
            "browseros",
            f"--remote-debugging-port={_CDP_PORT}",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--no-first-run",
            "--window-size=1280,800",
            "about:blank",
            quiet=False,
        )
    )
    await _listening(_HOST, _VNC_PORT, "x11vnc")
    await _listening(_HOST, _CDP_PORT, "BrowserOS CDP")


async def _navigate(url: str) -> None:
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(f"http://{_HOST}:{_CDP_PORT}")
            try:
                ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = ctx.pages[0] if ctx.pages else await ctx.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            finally:
                await browser.close()
    except Exception as exc:
        logger.warning("pre-navigation to %s failed; agent can navigate manually: %s", url, exc)


@env.initialize
async def _up() -> None:
    if not _port_open(_HOST, _VNC_PORT):
        logger.info("launching CartScout browser substrate")
        await _start_browser()
    else:
        await _listening(_HOST, _VNC_PORT, "x11vnc")
    env.add_capability(Capability.rfb(name="screen", url=f"rfb://{_HOST}", display=0))
    env.add_capability(Capability.cdp(name="browser", url=f"http://{_HOST}:{_CDP_PORT}"))


@env.shutdown
async def _down() -> None:
    for proc in reversed(_procs):
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    _procs.clear()


@env.template(id="shopping-context")
async def shopping_context_task(
    task_id: str,
    instruction: str,
    max_price: float,
    must_have: list[str],
    must_not_have: list[str],
    allowed_domains: list[str],
    token_budget: int = 750,
    require_cart: bool = False,
):
    spec = ShoppingTaskSpec(
        task_id=task_id,
        instruction=instruction,
        allowed_domains=allowed_domains,
        max_price=max_price,
        must_have=must_have,
        must_not_have=must_not_have,
        token_budget=token_budget,
        require_cart=require_cart,
    )
    search_url = f"https://www.google.com/search?q={quote_plus(instruction + ' ' + ' '.join(allowed_domains))}"
    await _navigate(search_url)
    prompt = f"""
You are CartScout, a shopping context scout for a browser agent.

Task:
{instruction}

Constraints:
- Max price: ${max_price:.2f}
- Must have: {must_have}
- Must not have: {must_not_have}
- Allowed domains: {allowed_domains}
- Cart prep required: {require_cart}

Hard safety rules:
- Do not sign in.
- Do not enter payment information.
- Do not place an order.
- Do not bypass CAPTCHA or bot checks.
- Do not buy regulated, age-restricted, medical, supplement, gift-card, weapon, alcohol, or nicotine products.
- Stop before checkout and return JSON only.

Return exactly one JSON object matching this schema:
{{
  "query": "string",
  "recommended_product": "string",
  "retailer": "string",
  "url": "https://...",
  "price": "$0.00",
  "delivery_or_pickup": "string or null",
  "seller": "string or null",
  "constraints_met": ["string"],
  "constraints_uncertain": ["string"],
  "evidence": [
    {{"url": "https://...", "quote": "visible quote", "supports": "claim supported"}}
  ],
  "recommendation": "string",
  "stop_before_checkout": true,
  "cart_prepared": false,
  "attempted_checkout": false
}}
"""
    answer = yield prompt.strip()
    result = score_purchase_packet(answer, spec)
    logger.info("task=%s reward=%.3f breakdown=%s reasons=%s", task_id, result.score, result.breakdown, result.reasons)
    yield result.score
