# CartScout Observability

CartScout has two different browser-control modes, and they produce different debugging views.

## Root Cause

The original `hud eval ... claude` run used HUD's RFB/computer-use path. The model acted through
desktop screenshots, mouse, and keyboard, so HUD naturally showed a visual replay.

The Browser Use harness acts through CDP. That is a better fit for browser-native RL, but Browser
Use owns its internal action loop. If we only submit the final result, HUD sees setup/evaluate plus
the final answer, not each browser action.

The fix is to mirror Browser Use's internal loop into HUD-native trace events:

- `AgentStep` for the model's browser decision.
- `ToolStep` for each Browser Use action and result.
- Screenshot-bearing `ToolStep`s from an optional RFB watch stream.

## Debugging Modes

### 1. Native RFB Replay

Use this when the priority is a live desktop replay:

```bash
set -a; source .env; set +a
uv run hud eval tasks.py claude \
  --runtime hud \
  --task-ids usb-c-charger-30w-under-40 \
  --max-steps 40 \
  -y
```

This is easiest to watch, but it trains/debugs a desktop action space, not the CDP browser action
space we want for open-source RL.

### 2. Structured CDP Harness

Use this when the priority is the trainable browser-native action contract:

```bash
set -a; source .env; set +a
uv run python scripts/run_structured_cdp_eval.py \
  --model deepseek-ai/DeepSeek-V3.1 \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 8
```

Expected trace shape:

- `agent_message` rows contain the observation summary, raw model output, parsed action, and parse error if any.
- `tool_call` rows named `structured_cdp.*` contain action results and screenshots.
- The final answer is emitted only through `emit_packet`, so the task grader sees the required packet JSON.

Known structured CDP smoke job:

- `https://hud.ai/jobs/fa38c6b655294b10aa60cb0c0ccd889c` - DeepSeek structured CDP run, reward `0.975`.

### 3. Browser Use CDP With RFB Watch

Use this when the priority is the trainable browser-native harness plus enough screenshots to debug:

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

Expected trace shape:

- `agent_message` rows contain Browser Use memory, action JSON, and results.
- `tool_call` rows named `browser_use.*` contain action outputs and screenshots.
- `tool_call` rows named `browser_snapshot` contain periodic RFB desktop screenshots.

Verify from the CLI:

```bash
uv run hud jobs --json <job_id>
uv run hud trace --json <trace_id>
```

Healthy observability should show `mcp_tool_steps > 0` and `is_screenshot: true` tool calls.

Known smoke job after the trace fix:

- `https://hud.ai/jobs/aff49fe2fb614b21aa830322d8ba48dc` - short two-step debug run,
  `mcp_tool_steps: 7`, screenshot-bearing `browser_snapshot` and `browser_use.navigate` tool calls.

Known successful CDP job:

- `https://hud.ai/jobs/c071f3f9a1174c458c63891fc748ae38` - Qwen CDP run, reward `0.975`.

## Fast Local Loop

Do not use hosted jobs for every prompt/tool/trace change. Run the environment locally and watch
the browser directly:

```bash
./scripts/run_local_env.sh
open http://127.0.0.1:8080/vnc.html
```

Then run the CDP harness against the local control channel:

```bash
set -a; source .env; set +a
uv run python scripts/run_structured_cdp_eval.py \
  --runtime tcp \
  --runtime-url tcp://127.0.0.1:8765 \
  --model deepseek-ai/DeepSeek-V3.1 \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 6
```

Browser Use can still be run locally for comparison:

```bash
set -a; source .env; set +a
uv run python scripts/run_browser_use_eval.py \
  --runtime tcp \
  --runtime-url tcp://127.0.0.1:8765 \
  --provider openai-like \
  --model Qwen/Qwen3-30B-A3B \
  --api-key-env HUD_API_KEY \
  --base-url https://inference.beta.hud.ai \
  --task-id usb-c-charger-30w-under-40 \
  --max-steps 6 \
  --no-thinking \
  --flash-mode \
  --rfb-watch-interval 3
```

Use hosted `--runtime hud` only when the local run looks good and you need a shareable platform
artifact.

## Harness Direction

Browser Use is a useful baseline and data-collection bridge, but it still hides too much of the
policy/action contract inside its own loop. For RL, prefer the structured CDP/DOM action space:

- `search(query, retailer)`
- `open_url(url)`
- `click_ref(ref_id)`
- `extract_page(query)`
- `find_text(pattern)`
- `emit_packet(packet)`
- `stop`

That keeps the trainable action log compact, deterministic, and easy to replay.
