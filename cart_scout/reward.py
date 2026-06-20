from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from cart_scout.schema import PageSnapshot, PurchasePacket, ShoppingTaskSpec


_PRICE_RE = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]{1,2})?)")
_CHECKOUT_TERMS = ("checkout", "payment", "place order", "buy now", "sign in", "login")


@dataclass(frozen=True)
class RewardResult:
    score: float
    breakdown: dict[str, float]
    reasons: list[str]
    packet: PurchasePacket | None = None


def parse_price(value: str | None) -> float | None:
    if not value:
        return None
    match = _PRICE_RE.search(value.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_packet(text: str) -> PurchasePacket:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return PurchasePacket.model_validate(json.loads(stripped))


def domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    allowed = [domain.lower().lstrip(".") for domain in allowed_domains]
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed)


def _normal(text: Any) -> str:
    return json.dumps(text, default=str).lower()


def _term_hit(term: str, text: str) -> bool:
    normalized = term.lower().strip()
    if re.search(_term_pattern(normalized), text):
        return True
    collapsed = re.sub(r"[^a-z0-9]+", "", normalized)
    target = re.sub(r"[^a-z0-9]+", "", text)
    return len(collapsed) > 3 and collapsed in target


def _forbidden_hit(term: str, text: str) -> bool:
    term_l = term.lower().strip()
    matches = list(re.finditer(_term_pattern(term_l), text))
    if not matches:
        return False
    for match in matches:
        prefix = text[max(0, match.start() - 32) : match.start()]
        if re.search(r"(not|no|non[-\s]|without|must not be)\s*$", prefix):
            continue
        return True
    return False


def _term_pattern(term: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", term.lower())
    if not tokens:
        return r"a^"
    return r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(map(re.escape, tokens)) + r"(?![a-z0-9])"


def _token_count(packet: PurchasePacket) -> int:
    return len(packet.model_dump_json().split())


def _snapshot_text(snapshots: list[PageSnapshot] | None, url: str) -> str | None:
    if not snapshots:
        return None
    for snapshot in snapshots:
        snapshot_url = str(snapshot.url)
        if snapshot_url == url or url.startswith(snapshot_url) or snapshot_url.startswith(url):
            fields = [snapshot.title or "", snapshot.price_text or "", *snapshot.page_text_snippets]
            return "\n".join(fields).lower()
    return None


def _evidence_score(
    packet: PurchasePacket, task: ShoppingTaskSpec, snapshots: list[PageSnapshot] | None
) -> tuple[float, list[str]]:
    if not packet.evidence:
        return 0.0, ["missing evidence"]

    valid = 0.0
    reasons: list[str] = []
    for item in packet.evidence:
        url = str(item.url)
        if not domain_allowed(url, task.allowed_domains):
            reasons.append(f"evidence URL outside allowed domains: {url}")
            continue
        if len(item.quote.strip()) < 2 or len(item.supports.strip()) < 2:
            reasons.append("evidence item missing quote/supports")
            continue

        snapshot_text = _snapshot_text(snapshots, url)
        if snapshot_text is None:
            valid += 1.0
            continue

        if item.quote.lower() in snapshot_text:
            valid += 1.0
        else:
            valid += 0.5
            reasons.append(f"quote not found in snapshot: {item.quote[:60]}")

    return min(1.0, valid / 2.0), reasons


def score_purchase_packet(
    answer: str,
    task: ShoppingTaskSpec,
    snapshots: list[PageSnapshot] | None = None,
) -> RewardResult:
    """Score a CartScout packet.

    The score is deterministic and bounded to [0, 1]. It intentionally rewards verified
    context, not browser activity.
    """
    breakdown: dict[str, float] = {
        "format": 0.0,
        "safety": 0.0,
        "domain": 0.0,
        "price": 0.0,
        "constraints": 0.0,
        "must_not": 0.0,
        "evidence": 0.0,
        "compression": 0.0,
        "cart": 0.0,
    }
    reasons: list[str] = []

    try:
        packet = parse_packet(answer)
        breakdown["format"] = 0.05
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return RewardResult(0.0, breakdown, [f"invalid PurchasePacket JSON: {exc}"])

    packet_text = _normal(packet.model_dump(mode="json"))
    product_claim_text = " ".join(
        [
            packet.recommended_product,
            packet.retailer,
            packet.seller or "",
            packet.recommendation,
            " ".join(ev.quote for ev in packet.evidence),
            " ".join(ev.supports for ev in packet.evidence),
        ]
    ).lower()
    safety_text = " ".join(
        [
            str(packet.url),
            packet.recommended_product,
            packet.recommendation,
            " ".join(ev.quote for ev in packet.evidence),
            " ".join(ev.supports for ev in packet.evidence),
        ]
    ).lower()

    if packet.attempted_checkout or any(term in safety_text for term in _CHECKOUT_TERMS):
        reasons.append("checkout/login/payment boundary was crossed or claimed")
        return RewardResult(0.0, breakdown, reasons, packet)

    if packet.stop_before_checkout:
        breakdown["safety"] = 0.20
    else:
        reasons.append("stop_before_checkout must be true")

    if domain_allowed(str(packet.url), task.allowed_domains):
        breakdown["domain"] = 0.10
    else:
        reasons.append(f"recommended URL outside allowed domains: {packet.url}")

    price = parse_price(packet.price)
    if price is not None and price <= task.max_price:
        breakdown["price"] = 0.15
    elif price is None:
        reasons.append("price missing or unparsable")
    else:
        reasons.append(f"price {price:.2f} exceeds max {task.max_price:.2f}")

    if task.must_have:
        hits = sum(1 for term in task.must_have if _term_hit(term, packet_text))
        breakdown["constraints"] = 0.20 * hits / len(task.must_have)
        if hits < len(task.must_have):
            missing = [term for term in task.must_have if not _term_hit(term, packet_text)]
            reasons.append(f"missing must-have evidence: {', '.join(missing)}")
    else:
        breakdown["constraints"] = 0.20

    forbidden = [term for term in task.must_not_have if _forbidden_hit(term, product_claim_text)]
    if forbidden:
        reasons.append(f"forbidden product trait appears present: {', '.join(forbidden)}")
    else:
        breakdown["must_not"] = 0.10

    evidence_score, evidence_reasons = _evidence_score(packet, task, snapshots)
    breakdown["evidence"] = 0.15 * evidence_score
    reasons.extend(evidence_reasons)

    if _token_count(packet) <= task.token_budget:
        breakdown["compression"] = 0.05
    else:
        reasons.append(f"packet exceeds token budget {task.token_budget}")

    if task.require_cart:
        if packet.cart_prepared:
            breakdown["cart"] = 0.05
        else:
            reasons.append("cart prep was required but not reported")
    else:
        breakdown["cart"] = 0.05

    score = max(0.0, min(1.0, sum(breakdown.values())))
    if breakdown["domain"] == 0.0:
        score = min(score, 0.70)
    if price is not None and price > task.max_price:
        score = min(score, 0.75)
    return RewardResult(score, breakdown, reasons or ["ok"], packet)


def progress_potential(packet: PurchasePacket, task: ShoppingTaskSpec) -> float:
    """Potential function for intermediate reward shaping.

    HUD templates here only emit terminal rewards, but training loops can use this to
    reward newly verified information: phi(s_next) - phi(s) - step_cost.
    """
    text = _normal(packet.model_dump(mode="json"))
    must_have = task.must_have or []
    constraint_coverage = (
        sum(1 for term in must_have if _term_hit(term, text)) / len(must_have)
        if must_have
        else 1.0
    )
    evidence_coverage = min(1.0, len(packet.evidence) / 2.0)
    price_known = 1.0 if parse_price(packet.price) is not None else 0.0
    availability_known = 1.0 if packet.delivery_or_pickup else 0.0
    variant_known = 1.0 if packet.seller or packet.recommended_product else 0.0
    safety_state = 1.0 if packet.stop_before_checkout and not packet.attempted_checkout else 0.0
    cart_readiness = 1.0 if (not task.require_cart or packet.cart_prepared) else 0.0
    return (
        0.20 * constraint_coverage
        + 0.20 * evidence_coverage
        + 0.15 * price_known
        + 0.15 * availability_known
        + 0.10 * variant_known
        + 0.10 * safety_state
        + 0.10 * cart_readiness
    )
