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
- `fireworks_rft/remote_server.py` - Fireworks remote rollout endpoint for browser-agent RL.
- `fireworks_rft/remote_evaluator.py` - Eval Protocol scorer wired to the remote rollout endpoint.
- `scripts/generate_task_samples.py` - deterministic train/eval shopping task generator.
- `data/` - starter task rows, generated train/eval task rows, and example observation snapshots.
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

## Task Data

Starter task specs live in `data/shopping_tasks.jsonl`. Generated task splits live in:

- `data/shopping_train_1000.jsonl` - 1,000 training tasks.
- `data/shopping_eval_200.jsonl` - 200 evaluation tasks.

Regenerate the splits deterministically:

```bash
uv run python scripts/generate_task_samples.py \
  --train-count 1000 \
  --eval-count 200 \
  --train-out data/shopping_train_1000.jsonl \
  --eval-out data/shopping_eval_200.jsonl
```

The generator validates every row as a `ShoppingTaskSpec` and fails if train/eval overlap by
`task_id` or exact instruction. `token_budget` is the budget for the final `PurchasePacket` JSON,
not the browser context length; the GRPO reward uses it as a compression signal.

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

## Structured CDP Harness

For the trainable live-browser shopping agent, use the structured CDP harness. It attaches to HUD's
real Chromium CDP capability, gives the policy compact page observations, and accepts a fixed JSON
action space: `open_url`, `search_retailer`, `click_ref`, `fill_ref`, `press`, `scroll`, `go_back`,
`extract_page`, `find_text`, `screenshot`, `emit_packet`, and `stop`.

Hosted HUD:

```bash
set -a; source .env; set +a
uv run python scripts/run_structured_cdp_eval.py \
  --runtime hud \
  --model deepseek-ai/DeepSeek-V3.1 \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 8 \
  --rfb-watch-interval 3
```

Local HUD TCP substrate:

```bash
NOVNC_PORT=8082 ./scripts/run_local_env.sh
set -a; source .env; set +a
uv run python scripts/run_structured_cdp_eval.py \
  --runtime tcp \
  --runtime-url tcp://127.0.0.1:8765 \
  --model deepseek-ai/DeepSeek-V3.1 \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 8 \
  --rfb-watch-interval 3
```

Known smoke:

- `https://hud.ai/jobs/c02c7ce530234394bb9260ffa4054e95` - local HUD TCP, structured CDP,
  completed, reward `0.975`, emitted a final `PurchasePacket` after 8 steps.

## Browser Use / CDP Harness

Browser Use over CDP remains useful for comparison and data collection, but it is not the preferred
policy loop for GRPO because its action schema is less fixed than the structured CDP controller.
Use it when you want Browser Use's built-in agent behavior over the same HUD browser capability:

```bash
uv sync --extra dev
uv run python scripts/run_browser_use_eval.py \
  --provider anthropic \
  --model claude-sonnet-4-5 \
  --api-key-env ANTHROPIC_API_KEY \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 35
```

This harness records Browser Use actions, URLs, and screenshots into HUD trace steps. It will not
look exactly like the native RFB/computer-use replay because Browser Use drives Chrome over CDP, not
desktop mouse/keyboard screenshots. Use the native `hud eval ... claude` path when the goal is to
watch a human-like desktop replay; use this CDP path when the goal is a trainable browser-native
action loop.

For debugging, add an RFB watch stream. The agent still acts through CDP, but HUD also receives
periodic desktop screenshots from the same browser:

```bash
set -a; source .env; set +a
uv run python scripts/run_browser_use_eval.py \
  --provider openai-like \
  --model Qwen/Qwen3-30B-A3B \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 18 \
  --no-thinking \
  --flash-mode \
  --rfb-watch-interval 3
```

HUD Gateway example with a trainable OpenAI-compatible model:

```bash
set -a; source .env; set +a
uv run python scripts/run_browser_use_eval.py \
  --provider openai-like \
  --model Qwen/Qwen3-30B-A3B \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 18 \
  --no-thinking \
  --flash-mode
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
  --task-id usb-c-charger-30w-under-40 \
  --no-thinking \
  --flash-mode
```

- `ollama` - Browser Use + local Ollama:

```bash
uv run python scripts/run_browser_use_eval.py \
  --provider ollama \
  --model qwen2.5:14b \
  --base-url http://127.0.0.1:11434
```

Recommendation for RL: keep HUD as the environment/reward system, but do not train against raw RFB
desktop actions. Train the policy over the structured CDP/DOM action space so rewards map cleanly
to model behavior.

## Local Iteration

Hosted HUD jobs are the right final check, but they are slow for harness debugging. For faster
iteration, run the environment locally in Docker and watch the desktop through noVNC:

```bash
./scripts/run_local_env.sh
open http://127.0.0.1:8080/vnc.html
```

Then point the structured CDP harness at the local control channel:

```bash
set -a; source .env; set +a
uv run python scripts/run_structured_cdp_eval.py \
  --runtime tcp \
  --runtime-url tcp://127.0.0.1:8765 \
  --model deepseek-ai/DeepSeek-V3.1 \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 8 \
  --no-trace-screenshots
```

Use this loop for prompt/tool/observability changes. Use hosted `--runtime hud` after a local run
looks good and you want the platform job artifact.

See [docs/observability.md](docs/observability.md) for the full debugging workflow and known smoke
jobs.

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

The HUD environment currently uses `score_purchase_packet(...)`, which keeps the original terminal
reward shape and optional snapshot-backed evidence verification.

For GRPO over final packet completions, use `score_grpo_packet(...)`. It keeps hard deterministic
safety gates, then applies this dense packet reward:

```text
format              0.05
domain              0.10
price               0.15
must_have           0.25
must_not            0.15
evidence_quality    0.20
recommendation      0.05
compression         0.05
```

Caps prevent common reward hacking: no allowed-domain URL, missing/over-budget price, missing
must-have constraints, present forbidden traits, and missing/weak evidence all limit the final score.

An optional Fireworks Qwen judge can refine the semantic components only:

```python
from cart_scout.reward import FireworksQwenJudge, score_grpo_packet

judge = FireworksQwenJudge()  # reads FIREWORKS_API_KEY
result = score_grpo_packet(answer, task, judge=judge)
```

The judge can adjust `must_have`, `evidence_quality`, and `recommendation`; it does not override
hard safety gates, domain checks, price parsing, compression, or caps. If the judge call fails, the
reward falls back to deterministic scoring and records the failure in `RewardResult.reasons`.

Intermediate shaping can use:

```python
r_t = progress_potential(packet_next, task) - progress_potential(packet_prev, task) - 0.01
```

This rewards new verified information, not clicks.

## Fireworks Remote GRPO Path

The active Fireworks path is remote-rollout-first. The training dataset contains task-only
`ShoppingTaskSpec` rows; Fireworks calls a remote service for each rollout, and that service is
responsible for launching the shopping browser agent, letting it browse allowed live retailer pages,
collecting the final `PurchasePacket`, and scoring it.

The previous synthetic snippet datasets were removed because they cannot train the main shopping
agent to browse. They were only useful for packet-format smoke testing.

Local protocol smoke test:

```bash
uv sync --extra dev
uv run --extra dev pytest -q tests/test_remote_server.py
```

Run the local remote server manually:

```bash
uv run --extra dev uvicorn fireworks_rft.remote_server:app \
  --host 127.0.0.1 \
  --port 9000
```

The default server mode is `CART_SCOUT_REMOTE_MODE=stub`. It returns a deterministic valid
`PurchasePacket` so the Fireworks `/init` and `/status` contract can be tested without launching a
browser. This is not the final browser-agent rollout.

For live browsing through the existing HUD CDP harness:

```bash
export CART_SCOUT_REMOTE_MODE=structured-cdp
export CART_SCOUT_HUD_RUNTIME=hud
export HUD_API_KEY=...
uv run --extra dev uvicorn fireworks_rft.remote_server:app \
  --host 0.0.0.0 \
  --port 9000
```

For a local Docker HUD substrate, use TCP instead:

```bash
NOVNC_PORT=8082 ./scripts/run_local_env.sh
export CART_SCOUT_REMOTE_MODE=structured-cdp
export CART_SCOUT_HUD_RUNTIME=tcp
export CART_SCOUT_HUD_RUNTIME_URL=tcp://127.0.0.1:8765
uv run --extra dev uvicorn fireworks_rft.remote_server:app \
  --host 127.0.0.1 \
  --port 9000
```

In `structured-cdp` mode the remote server builds a one-row HUD `shopping-context` task from the
incoming `ShoppingTaskSpec`, runs `StructuredCDPAgent` against HUD's CDP capability, then scores the
agent's final `PurchasePacket`. `browser-use` remains available as an alternate mode, but structured
CDP is the intended GRPO rollout path.

Remote browser modes record screenshots on each browser action by default. Structured CDP remote
runs also record periodic desktop snapshots every 3 seconds by default through HUD's RFB capability,
matching the high-observability smoke-test viewing experience. Set
`CART_SCOUT_STRUCTURED_TRACE_SCREENSHOTS=false` or `CART_SCOUT_BROWSER_TRACE_SCREENSHOTS=false` to
disable action screenshots, and set `CART_SCOUT_STRUCTURED_RFB_WATCH_INTERVAL=0` to disable periodic
structured-CDP desktop snapshots when you intentionally want lower-volume traces.

To create the task-only Eval Protocol rows without launching a Fireworks job:

```bash
uv run --extra dev python train_hud.py --poc --skip-train
```

To dry-run a Fireworks POC against a deployed remote server:

```bash
export EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL=https://your-public-rollout-server.example
uv run --extra dev python train_hud.py --poc --dry-run
```

To launch a small POC job:

```bash
uv run --extra dev python train_hud.py --poc
```

`--poc` uses 16 task rows, chunk size 4, 2 response candidates, and
`accounts/fireworks/models/qwen3-vl-8b-instruct` for the smoke base model. If your Fireworks account
has a private Qwen3-VL-4B model, pass it explicitly:

```bash
uv run --extra dev python train_hud.py --poc \
  --base-model accounts/fireworks/models/qwen3-vl-4b-instruct \
  --output-model cart-scout-qwen3-vl-4b-remote-poc-grpo
```

To train the target shopping agent:

```bash
uv run --extra dev python train_hud.py \
  --remote-base-url https://your-public-rollout-server.example
```

The default target base model is `accounts/fireworks/models/qwen3-vl-32b-instruct`.

This installed `eval-protocol` release does not expose the cookbook's `--remote-server-url` flag.
`train_hud.py` passes both `EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL` and
`CART_SCOUT_REMOTE_BASE_URL` to local evaluator discovery/upload. The Fireworks evaluator runtime
also needs one of those environment variables set to the public rollout server URL; otherwise the
uploaded evaluator falls back to `http://127.0.0.1:9000`, which is only useful for local smoke tests.

Remote rollout TODO:

- Run an end-to-end public-server smoke with `CART_SCOUT_REMOTE_MODE=structured-cdp`.
- Tune structured CDP settings for Qwen3-VL rollouts: max steps, sampling params, screenshot
  tracing, and timeout values.
- Add trace/snapshot-backed evidence checks on top of `score_grpo_packet(...)`.
- Replace in-memory rollout status with durable storage before running high-volume jobs.

The lower-level Fireworks Training API RL cookbook is still a separate future path. It uses
`async_rl_loop`, where the project would provide browser rollouts that return token ids, logprobs,
loss masks, and scalar rewards, then configure `policy_loss="grpo"` in `train.py`.
