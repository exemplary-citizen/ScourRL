"""CartScout: reward, task, and episode primitives for shopping browser agents."""

from cart_scout.episodes import EpisodePlan, EpisodeRecord, build_episode_plans
from cart_scout.reward import (
    FireworksJudgeConfig,
    FireworksQwenJudge,
    LLMJudgeResult,
    RewardResult,
    progress_potential,
    score_grpo_packet,
    score_purchase_packet,
)
from cart_scout.schema import Evidence, PurchasePacket, ShoppingTaskSpec
from cart_scout.task_bank import TASK_SEEDS, TaskSeed, generate_task_specs

__all__ = [
    "Evidence",
    "EpisodePlan",
    "EpisodeRecord",
    "FireworksJudgeConfig",
    "FireworksQwenJudge",
    "LLMJudgeResult",
    "PurchasePacket",
    "RewardResult",
    "ShoppingTaskSpec",
    "TASK_SEEDS",
    "TaskSeed",
    "build_episode_plans",
    "generate_task_specs",
    "progress_potential",
    "score_grpo_packet",
    "score_purchase_packet",
]
