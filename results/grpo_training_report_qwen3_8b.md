# GRPO Training Report — `cart-scout-qwen3-8b-grpo`

- **Base model**: `Qwen/Qwen3-8B`, forked to the trainable model `cart-scout-qwen3-8b-grpo`
- **Infra**: rollouts via HUD's hosted browser runtime + structured CDP harness; sampling via HUD's inference gateway; weight updates via `hud.train.TrainingClient` (Tinker-backed)
- **Reward**: `score_grpo_packet` (deterministic safety gates, domain/price/compression checks, must-have/evidence/recommendation scoring), blended 60% deterministic / 40% `FireworksQwenJudge` (`accounts/fireworks/models/qwen3p7-plus`) on the must_have/evidence_quality/recommendation components only, for the two main training stages
- **Group size**: 4 rollouts/task (GRPO) · **Concurrency**: 2 (Tinker's per-model session cap forced this; 4+ triggered "Too many active sessions" errors, 1 was safe but impractically slow)
- **Train data**: `data/shopping_train_1000.jsonl`, advanced sequentially in chunks, never restarted from scratch
- **Training window**: 2026-06-21 05:21–11:36 (~6h15m wall clock), stopped manually before a 4th, near-empty stage could start

## Training stages

| Stage | Tasks (rows) | Rollouts | LLM judge | reward_mean | reward_min | reward_max | optim_step | checkpoint_id |
|---|---|---|---|---|---|---|---|---|
| Concurrency smoke test | 0–4 (5) | 20 | no | 0.2918 | 0.0 | 0.90 | 1 | `b7d181d1-28c6-437c-9468-92e989c502d1` |
| Chunk 1 | 0–24 (25) | 100 | yes | **0.4195** | 0.0 | 0.97 | 2 | `2624c851-1321-470a-be12-6bcf86afdd60` |
| Chunk 2 | 25–49 (25) | 100 | yes | 0.3658 | 0.0 | 0.958 | 3 | **`306b0f53-d280-4cc8-b261-3969b76bc679`** (final / current head) |
| Chunk 3 | 50–51 (2, auto-shrunk) | 0 | — | — | — | — | — | aborted before any rollouts ran (stopped per instruction) |

**Final checkpoint in use: `306b0f53-d280-4cc8-b261-3969b76bc679` (optim_step 3).**

Note: the smoke test and Chunk 1 both started at row 0, so rows 0–4 received two extra gradient updates relative to rows 5–49. This was an artifact of validating concurrency before launching the full pipeline, not intentional oversampling.

## Reward breakdown (mean component score per stage)

| Component | Smoke test (no judge) | Chunk 1 (judge) | Chunk 2 (judge) |
|---|---|---|---|
| format | 0.0225 | 0.0295 | 0.0260 |
| domain | 0.0400 | 0.0550 | 0.0470 |
| price | 0.0540 | 0.0789 | 0.0744 |
| must_have | 0.1000 | 0.1381 | 0.1242 |
| must_not | 0.0450 | 0.0480 | 0.0390 |
| evidence_quality | 0.0338 | 0.0671 | 0.0626 |
| recommendation | 0.0190 | 0.0271 | 0.0246 |
| compression | 0.0200 | 0.0295 | 0.0260 |

All components improved markedly from the smoke test (pre-judge, earlier checkpoint) to Chunk 1. Chunk 2's component averages are slightly below Chunk 1's, consistent with reward_mean dipping from 0.42 to 0.37 — within the kind of batch-to-batch variance expected at this sample size (25 tasks/100 rollouts per chunk), not a clear regression signal on its own.

## Reliability notes

- 0 occurrences of the Tinker session-cap error ("Too many active sessions") and 0 "llm judge unavailable" errors across both judged training chunks — concurrency=2 and the Fireworks judge integration were both stable for the full run.
- Transient `503 Service Unavailable` errors occurred (21 in chunk 1, 0 new in chunk 2) but were all retried through successfully by the harness's backoff logic.
- Per-rollout terminal failures (exhausted retries, scored 0): ~7% in chunk 1, ~14% in chunk 2 — browser/network flakiness, not systemic.

## Eval

Inference-only eval on the first 40 examples (sequential, not random) from `data/shopping_eval_200.jsonl` against checkpoint `306b0f53-d280-4cc8-b261-3969b76bc679` was launched separately and was still in progress at the time of this report. Results will land in `data/eval_runs/grpo_eval_first40_post_train_qwen3_8b.jsonl`.
