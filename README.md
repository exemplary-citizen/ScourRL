# CartScout

CartScout is an RL context manager for real-browser shopping agents. It does not train an agent to
autonomously buy products. It trains a browser sidecar to gather buying-critical context, cite
evidence, optionally prepare a cart, and stop before checkout.

The core output is a compact `PurchasePacket` JSON:

```json
{
  "query": "USB-C charger for MacBook Air M2 under $40",
  "recommended_product": "Anker 30W USB-C Charger",
  "retailer": "Target",
  "url": "https://www.target.com/...",
  "price": "$19.99",
  "delivery_or_pickup": "Pickup available today",
  "seller": "Target",
  "constraints_met": ["USB-C", "Power Delivery", "30W", "Under $40"],
  "constraints_uncertain": [],
  "evidence": [
    {
      "url": "https://www.target.com/...",
      "quote": "30W USB-C Power Delivery",
      "supports": "wattage and USB-C PD"
    }
  ],
  "recommendation": "Recommend this item. It satisfies the constraints and is below budget.",
  "stop_before_checkout": true
}
```

## What This Repo Contains

- `env.py` - HUD v6 browser environment exposing one Chromium over CDP and RFB/VNC.
- `tasks.py` - safe shopping research tasks for grouped browser rollouts.
- `cart_scout/schema.py` - `PurchasePacket`, evidence, snapshots, and task specs.
- `cart_scout/reward.py` - deterministic reward and intermediate potential function.
- `fireworks_rft/evaluator.py` - single-turn RFT evaluator for snippets -> packet training.
- `data/` - starter task rows and example observation snapshots.
- `tests/` - offline reward/template tests that do not require Docker or Chromium.

## Safety Boundary

Allowed:

- Search public product pages.
- Read visible product information.
- Extract price, seller, variant, availability, constraints, and evidence quotes.
- Recommend an item.
- Optionally prepare a cart when the task explicitly requires it.
- Stop for human approval.

Forbidden:

- Signing in automatically.
- Entering payment information.
- Placing an order.
- Bypassing CAPTCHA, bot checks, rate limits, or access controls.
- Buying regulated, age-restricted, medical, supplement, gift-card, weapon, alcohol, or nicotine products.
- Bulk scraping product databases.

## Install

```bash
uv sync --extra dev
```

HUD's browser image is Linux/amd64-oriented. On macOS, use hosted HUD for real browser rollouts.
The offline tests still run locally.

## Test

```bash
uv run --extra dev pytest -q
```

## HUD Evaluation

This repo is linked to the hosted HUD environment:

- Environment: `cart-scout`
- HUD registry ID: `bc9c854f-de6a-4613-b38e-354f982d6093`
- HUD page: `https://hud.ai/environments/bc9c854f-de6a-4613-b38e-354f982d6093`

Create a local `.env` with your HUD key:

```bash
cp .env.example .env
# edit .env and set HUD_API_KEY=...
```

Deploy without uploading `.env` into the runtime image:

```bash
uv run hud deploy . --no-env
```

Run one smoke task:

```bash
uv run hud eval tasks.py claude --runtime hud \
  --task-ids usb-c-charger-30w-under-40 \
  --max-steps 80 \
  -y
```

Run the full taskset with grouped rollouts:

```bash
uv run hud eval tasks.py claude --runtime hud \
  --full \
  --group 3 \
  --max-steps 80 \
  -y
```

For a Linux/amd64 local run:

```bash
docker build -f Dockerfile.hud -t cart-scout:dev .
docker run -d -p 8765:8765 -p 8080:8080 cart-scout:dev
uv run hud eval tasks.py claude --runtime tcp://127.0.0.1:8765 --task-ids usb-c-charger-30w-under-40 -y
```

Open `http://localhost:8080/vnc.html` to watch the browser.

### Current Baseline Notes

The environment is patterned after `hud-evals/hud-browser`: `env.py` starts a BrowserOS/Chromium
desktop and publishes the same browser through both CDP (`browser`) and RFB/VNC (`screen`). BrowserOS
is used because it exposes a real browser over CDP while also rendering into the VNC desktop HUD can
record.

The `claude` HUD baseline currently drives the RFB/computer-use path. That validates the deployed
browser environment, but it is not the harness we should optimize for open-source RL. In smoke tests
it spent many steps interacting with retail UI and did not reliably emit the final JSON packet. For
training, prefer a CDP/structured-action harness that gives the policy compact observations and
discrete browser actions instead of raw screenshots and free-form desktop typing.

Known smoke jobs:

- `https://hud.ai/jobs/1b452324782d42bc9a7097665e87fa75` - v1 deploy, completed, reward `0.0`, no JSON packet.
- `https://hud.ai/jobs/da1bd00a3414409b8193fbcde2d1466f` - v2 deploy, long RFB rollout, failed grading after connection loss.

## Browser Use / CDP Harness

For a better browser-agent baseline, use Browser Use over the environment's CDP capability instead
of the default `claude` RFB/computer-use path:

```bash
uv sync --extra dev
uv run python scripts/run_browser_use_eval.py \
  --provider anthropic \
  --model claude-sonnet-4-5 \
  --api-key-env ANTHROPIC_API_KEY \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 35
```

Provider options:

- `anthropic` - Browser Use + Claude; requires `ANTHROPIC_API_KEY`.
- `openai` - Browser Use + OpenAI; requires `OPENAI_API_KEY`.
- `openai-like` - Browser Use + any OpenAI-compatible endpoint. Use this for local/vLLM/SGLang RL policy servers:

```bash
export OPENAI_LIKE_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_LIKE_API_KEY=dummy
uv run python scripts/run_browser_use_eval.py \
  --provider openai-like \
  --model cart-scout-policy \
  --task-id usb-c-charger-30w-under-40
```

- `ollama` - Browser Use + local Ollama:

```bash
uv run python scripts/run_browser_use_eval.py \
  --provider ollama \
  --model qwen2.5:14b \
  --base-url http://127.0.0.1:11434
```

Recommendation for RL: keep HUD as the environment/reward system, but do not train against raw RFB
desktop actions. Train a policy over a compact CDP/DOM action space such as `search`, `open_url`,
`click_ref`, `extract_page`, `find_text`, `emit_packet`, and `stop`. Browser Use is a good interim
baseline and data-collection harness; the trainable policy should eventually use the same structured
observations/actions directly so rewards map cleanly to model behavior.

## Reward

Terminal reward is bounded to `[0, 1]`:

- JSON/schema format.
- Safety boundary and stop-before-checkout.
- Allowed retailer domain.
- Price under budget.
- Must-have constraints.
- Must-not-have absence.
- Evidence quotes and support text.
- Compression within token budget.
- Optional cart readiness.

Intermediate shaping can use:

```python
r_t = progress_potential(packet_next, task) - progress_potential(packet_prev, task) - 0.01
```

This rewards new verified information, not clicks.

## Fireworks RFT Path

Use `fireworks_rft/evaluator.py` for the simpler 24-hour path:

```python
from fireworks_rft.evaluator import score_packet

score, reason = score_packet(model_text, ground_truth)
```

Recommended first dataset shape:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Task plus observed product snippets. Return PurchasePacket JSON only."
    }
  ],
  "ground_truth": {
    "max_price": 40,
    "must_have": ["USB-C", "Power Delivery", "30W"],
    "must_not_have": ["Lightning"],
    "allowed_domains": ["target.com", "amazon.com"]
  }
}
```

The browser RFT stretch path is `fireworks_rft/remote_server.py`.
