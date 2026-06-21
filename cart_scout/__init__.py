"""CartScout: reward and task primitives for shopping browser agents."""

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

__all__ = [
    "Evidence",
    "FireworksJudgeConfig",
    "FireworksQwenJudge",
    "LLMJudgeResult",
    "PurchasePacket",
    "RewardResult",
    "ShoppingTaskSpec",
    "progress_potential",
    "score_grpo_packet",
    "score_purchase_packet",
]
