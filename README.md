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
uv run pytest -q
```

## HUD Evaluation

```bash
cp .env.example .env
hud deploy .
hud eval tasks.py claude --runtime hud --full --group 3 --max-steps 40
```

For a Linux/amd64 local run:

```bash
docker build -f Dockerfile.hud -t cart-scout:dev .
docker run -d -p 8765:8765 -p 8080:8080 cart-scout:dev
hud eval tasks.py claude --runtime tcp://127.0.0.1:8765 --task-ids usb-c-charger-30w-under-40 -y
```

Open `http://localhost:8080/vnc.html` to watch the browser.

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
