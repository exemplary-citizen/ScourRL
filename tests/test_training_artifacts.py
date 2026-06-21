from cart_scout.episodes import EpisodeRecord, build_episode_plans
from scripts.train_rl import _preference_rows, _sft_rows


def test_training_artifact_rows_from_grouped_records():
    plans = build_episode_plans(episode_count=3, rollouts_per_task=3)
    records = [
        EpisodeRecord(**plans[0].model_dump(mode="json"), status="completed", reward=0.95, answer='{"ok": true}'),
        EpisodeRecord(**plans[1].model_dump(mode="json"), status="completed", reward=0.40, answer='{"ok": false}'),
        EpisodeRecord(**plans[2].model_dump(mode="json"), status="completed", reward=0.85, answer='{"ok": true}'),
    ]

    sft_rows = _sft_rows(records, min_reward=0.75)
    preference_rows = _preference_rows(records, margin=0.10)

    assert len(sft_rows) == 2
    assert sft_rows[0]["messages"][0]["role"] == "user"
    assert len(preference_rows) == 1
    assert preference_rows[0]["chosen_reward"] == 0.95
    assert preference_rows[0]["rejected_reward"] == 0.40
