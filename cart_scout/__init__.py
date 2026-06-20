"""CartScout: reward and task primitives for shopping browser agents."""

from cart_scout.reward import RewardResult, progress_potential, score_purchase_packet
from cart_scout.schema import Evidence, PurchasePacket, ShoppingTaskSpec

__all__ = [
    "Evidence",
    "PurchasePacket",
    "RewardResult",
    "ShoppingTaskSpec",
    "progress_potential",
    "score_purchase_packet",
]
