from __future__ import annotations

import json
from typing import Any

from cart_scout.reward import score_purchase_packet
from cart_scout.schema import ShoppingTaskSpec


def score_packet(text: str, truth: dict[str, Any]) -> tuple[float, str]:
    """Fireworks single-turn RFT evaluator entry point.

    `truth` should contain the same fields as ShoppingTaskSpec, or at least
    max_price, must_have, must_not_have, and allowed_domains.
    """
    spec = ShoppingTaskSpec(
        task_id=truth.get("task_id", "fireworks-row"),
        instruction=truth.get("instruction", "Return a CartScout PurchasePacket JSON."),
        allowed_domains=truth.get("allowed_domains", ["amazon.com", "target.com"]),
        max_price=float(truth["max_price"]),
        must_have=list(truth.get("must_have", [])),
        must_not_have=list(truth.get("must_not_have", [])),
        token_budget=int(truth.get("token_budget", 750)),
        require_cart=bool(truth.get("require_cart", False)),
    )
    result = score_purchase_packet(text, spec)
    return result.score, json.dumps({"breakdown": result.breakdown, "reasons": result.reasons})
