# Error Analysis — GRPO Training Set (cart-scout-qwen3-8b-grpo)

Covers all logged rollouts from the smoke test and training chunks 1-2 (50 tasks, 220 rollouts, rows 0-49 of `data/shopping_train_1000.jsonl`). Built from `task=/reward=/breakdown=` log lines; the scorer's free-text `reasons` and the raw packet text were not persisted to these logs, so failure modes are inferred from the reward breakdown, not the exact gate message.

## Reliability summary

| Stage | Rows | Rollouts logged | Infra run failures | Hard-zero rollouts | Hard-zero rate |
|---|---|---|---|---|---|
| smoke_test | rows 0-4 | 20 | 1 | 12 | 60.0% |
| chunk_1 | rows 0-24 | 100 | 11 | 41 | 41.0% |
| chunk_2 | rows 25-49 | 100 | 14 | 48 | 48.0% |

**Overall hard-zero rate: 45.9%** (101/220 rollouts scored exactly 0.0 — invalid packet JSON, a crossed checkout/safety boundary, or a regulated-product mention). `Infra run failures` are harness-level errors (retries exhausted on a single rollout, e.g. transient gateway/browser issues) and are a *subset* of hard-zero rollouts, not additional to them.

## Hardest tasks (all 4 rollouts scored exactly 0.0)

- `train_0038_aa-batteries-for-a-compact-setup`
- `train_0047_measuring-cups-for-a-small-apartment`
- `train_0048_coffee-filters-for-a-small-apartment`

## Easiest tasks (mean reward ≥ 0.8 across all 4 rollouts)

- `train_0016_coffee-filters-for-a-small-apartment` — mean reward 0.9287
- `train_0003_a-shower-curtain-liner-for-a-budget-restock` — mean reward 0.8641
- `train_0049_kitchen-trash-bags-for-a-basic-starter-kit` — mean reward 0.847

## Component weakness ranking (excluding hard-zero rollouts)

Among rollouts that passed the format/safety gates (so the component breakdown reflects real partial credit, not a blanket zero), this is how often each component still scored 0:

| Component | % of rollouts scoring 0 | Mean score |
|---|---|---|
| must_not | 42.0% | 0.0807 |
| domain | 7.6% | 0.0924 |
| format | 0.0% | 0.05 |
| price | 0.0% | 0.1379 |
| must_have | 0.0% | 0.2372 |
| evidence_quality | 0.0% | 0.1147 |
| recommendation | 0.0% | 0.0466 |
| compression | 0.0% | 0.05 |

## Highest-variance tasks (largest spread between best and worst of its 4 rollouts)

| Task | Mean reward | Spread (max-min) |
|---|---|---|
| `train_0015_black-ink-pens-for-a-small-apartment` | 0.705 | 0.97 |
| `train_0042_a-storage-bin-for-a-budget-restock` | 0.2395 | 0.958 |
| `train_0002_a-cutting-board-for-a-small-apartment` | 0.4878 | 0.94 |
| `train_0041_a-usb-c-charging-cable-for-a-basic-starter-kit` | 0.5913 | 0.94 |
| `train_0005_a-usb-c-charging-cable-for-everyday-household-us` | 0.2735 | 0.925 |
| `train_0006_an-insulated-lunch-bag-for-a-basic-starter-kit` | 0.446 | 0.925 |
| `train_0018_an-indoor-extension-cord-for-a-basic-starter-kit` | 0.2313 | 0.925 |
| `train_0020_sticky-notes-for-a-budget-restock` | 0.65 | 0.925 |
| `train_0027_an-indoor-extension-cord-for-a-dorm-room` | 0.2313 | 0.925 |
| `train_0032_coffee-filters-for-everyday-household-use` | 0.6938 | 0.925 |

## Figures

- `error_analysis_hard_zero_rate.png` — hard-failure rate per training stage
- `error_analysis_component_zero_rate.png` — which reward component most often still scores 0
- `error_analysis_per_task_reward_hist.png` — distribution of per-task mean reward
