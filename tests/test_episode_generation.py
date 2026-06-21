from cart_scout.episodes import build_episode_plans, write_hud_taskset
from cart_scout.task_bank import TASK_SEEDS, summarize_task_specs


def test_build_episode_plans_default_shape():
    plans = build_episode_plans()
    assert len(plans) == 300
    assert len({plan.group_id for plan in plans}) == 100
    assert {plan.rollout_index for plan in plans[:3]} == {0, 1, 2}
    assert plans[0].provider == "anthropic"
    assert plans[0].model == "claude-sonnet-4-5"
    assert plans[0].task.allowed_domains


def test_task_bank_is_diverse():
    plans = build_episode_plans(episode_count=90, rollouts_per_task=3)
    summary = summarize_task_specs([plan.task for plan in plans[::3]])
    assert len(TASK_SEEDS) >= 60
    assert len(summary) >= 8
    assert all(count > 0 for count in summary.values())


def test_write_hud_taskset(tmp_path):
    plans = build_episode_plans(episode_count=6, rollouts_per_task=3)
    path = tmp_path / "generated_tasks.py"
    write_hud_taskset(path, plans)
    text = path.read_text(encoding="utf-8")
    assert "shopping_context_task" in text
    assert "tasks = [" in text
    assert plans[0].episode_id in text
    assert text.count("    _episode_task(") == 6
