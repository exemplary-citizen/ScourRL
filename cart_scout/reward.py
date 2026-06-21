from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

from pydantic import ValidationError

from cart_scout.schema import PageSnapshot, PurchasePacket, ShoppingTaskSpec


_PRICE_RE = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]{1,2})?)")
_CHECKOUT_TERMS = ("checkout", "payment", "place order", "buy now", "sign in", "login")
_REGULATED_TERMS = (
    "regulated",
    "age-restricted",
    "medical",
    "medicine",
    "supplement",
    "gift card",
    "gift-card",
    "weapon",
    "alcohol",
    "beer",
    "wine",
    "liquor",
    "nicotine",
    "tobacco",
)
_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_FIREWORKS_QWEN_JUDGE_MODEL = "accounts/fireworks/models/qwen3p7-max"
_GRPO_FORMAT_WEIGHT = 0.05
_GRPO_DOMAIN_WEIGHT = 0.10
_GRPO_PRICE_WEIGHT = 0.15
_GRPO_MUST_HAVE_WEIGHT = 0.25
_GRPO_MUST_NOT_WEIGHT = 0.15
_GRPO_EVIDENCE_WEIGHT = 0.20
_GRPO_RECOMMENDATION_WEIGHT = 0.05
_GRPO_COMPRESSION_WEIGHT = 0.05


@dataclass(frozen=True)
class RewardResult:
    score: float
    breakdown: dict[str, float]
    reasons: list[str]
    packet: PurchasePacket | None = None


@dataclass(frozen=True)
class LLMJudgeResult:
    must_have_score: float
    evidence_quality_score: float
    recommendation_score: float
    reasons: list[str]


@dataclass(frozen=True)
class FireworksJudgeConfig:
    api_key: str | None = None
    base_url: str = _FIREWORKS_BASE_URL
    model: str = _FIREWORKS_QWEN_JUDGE_MODEL
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout_s: float = 30.0


class PacketJudge(Protocol):
    def judge(self, packet: PurchasePacket, task: ShoppingTaskSpec) -> LLMJudgeResult:
        ...


class FireworksQwenJudge:
    """LLM judge using Qwen3.7 Max through Fireworks' OpenAI-compatible API."""

    def __init__(self, config: FireworksJudgeConfig | None = None) -> None:
        self.config = config or FireworksJudgeConfig()

    def judge(self, packet: PurchasePacket, task: ShoppingTaskSpec) -> LLMJudgeResult:
        api_key = self.config.api_key or os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required for FireworksQwenJudge")

        payload = {
            "model": self.config.model,
            "messages": _judge_messages(packet, task),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        req = urlrequest.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.config.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Fireworks judge request failed: {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
            return _parse_llm_judge_result(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Fireworks judge response was invalid: {exc}") from exc


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


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


def _parse_llm_judge_result(text: str) -> LLMJudgeResult:
    data = _parse_json_object(text)
    reasons = data.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = []
    return LLMJudgeResult(
        must_have_score=_clamp01(data.get("must_have_score", 0.0)),
        evidence_quality_score=_clamp01(data.get("evidence_quality_score", 0.0)),
        recommendation_score=_clamp01(data.get("recommendation_score", 0.0)),
        reasons=[str(reason) for reason in reasons[:5]],
    )


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _judge_messages(packet: PurchasePacket, task: ShoppingTaskSpec) -> list[dict[str, str]]:
    rubric = {
        "must_have_score": "0 to 1. Fraction of required constraints that are semantically satisfied by the packet.",
        "evidence_quality_score": "0 to 1. Evidence quotes are specific, relevant, and internally support the claimed product facts. Do not browse.",
        "recommendation_score": "0 to 1. Recommendation is concise, safe, and explains why the product fits the task without unsupported claims.",
        "reasons": "Up to five short strings explaining the scores.",
    }
    content = {
        "task": task.model_dump(mode="json"),
        "packet": packet.model_dump(mode="json"),
        "rubric": rubric,
        "instructions": [
            "/no_think",
            "Judge only semantic quality. Do not give credit for checkout, payment, login, or unsafe buying behavior.",
            "Do not verify live web truth. Use only the task and packet.",
            "Return exactly one JSON object with keys: must_have_score, evidence_quality_score, recommendation_score, reasons.",
            "All scores must be numbers between 0 and 1.",
        ],
    }
    return [
        {
            "role": "system",
            "content": "/no_think\nYou are a strict deterministic reward judge for shopping-agent RL. Return JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(content, ensure_ascii=True),
        },
    ]


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


def _negated_term_hit(term: str, text: str) -> bool:
    term_l = term.lower().strip()
    matches = list(re.finditer(_term_pattern(term_l), text))
    if not matches:
        return False
    return all(_is_negated_match(text, match.start()) for match in matches)


def _is_negated_match(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32) : start]
    return re.search(r"(not|no|non[-\s]|without|must not be)\s*$", prefix) is not None


def _term_pattern(term: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", term.lower())
    if not tokens:
        return r"a^"
    return r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(map(re.escape, tokens)) + r"(?![a-z0-9])"


def _token_count(packet: PurchasePacket) -> int:
    return len(packet.model_dump_json().split())


def _packet_value_text(packet: PurchasePacket) -> str:
    fields = [
        packet.query,
        packet.recommended_product,
        packet.retailer,
        str(packet.url),
        packet.price or "",
        packet.delivery_or_pickup or "",
        packet.seller or "",
        " ".join(packet.constraints_met),
        " ".join(packet.constraints_uncertain),
        " ".join(ev.quote for ev in packet.evidence),
        " ".join(ev.supports for ev in packet.evidence),
        packet.recommendation,
    ]
    return " ".join(fields).lower()


def _claim_text(packet: PurchasePacket) -> str:
    fields = [
        packet.recommended_product,
        packet.retailer,
        packet.seller or "",
        " ".join(ev.quote for ev in packet.evidence),
        " ".join(ev.supports for ev in packet.evidence),
        packet.recommendation,
    ]
    return " ".join(fields).lower()


def _must_have_text(packet: PurchasePacket) -> str:
    fields = [
        packet.recommended_product,
        " ".join(packet.constraints_met),
        " ".join(ev.quote for ev in packet.evidence),
        " ".join(ev.supports for ev in packet.evidence),
        packet.recommendation,
    ]
    return " ".join(fields).lower()


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


def score_grpo_packet(answer: str, task: ShoppingTaskSpec, judge: PacketJudge | None = None) -> RewardResult:
    """Score a final packet for GRPO training.

    This packet-only scorer is dense enough to rank samples within a GRPO group,
    while keeping safety failures and unusable outputs as hard zeros.
    """
    breakdown: dict[str, float] = {
        "format": 0.0,
        "domain": 0.0,
        "price": 0.0,
        "must_have": 0.0,
        "must_not": 0.0,
        "evidence_quality": 0.0,
        "recommendation": 0.0,
        "compression": 0.0,
    }
    reasons: list[str] = []
    caps: list[tuple[float, str]] = []

    try:
        packet = parse_packet(answer)
        breakdown["format"] = _GRPO_FORMAT_WEIGHT
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return RewardResult(0.0, breakdown, [f"invalid PurchasePacket JSON: {exc}"])

    value_text = _packet_value_text(packet)
    if packet.attempted_checkout:
        return RewardResult(0.0, breakdown, ["attempted_checkout must be false"], packet)
    if not packet.stop_before_checkout:
        return RewardResult(0.0, breakdown, ["stop_before_checkout must be true"], packet)
    if any(term in value_text for term in _CHECKOUT_TERMS):
        return RewardResult(0.0, breakdown, ["checkout/login/payment boundary was crossed or claimed"], packet)
    if any(term in value_text for term in _REGULATED_TERMS):
        return RewardResult(0.0, breakdown, ["regulated or forbidden product category appears present"], packet)

    if domain_allowed(str(packet.url), task.allowed_domains):
        breakdown["domain"] = _GRPO_DOMAIN_WEIGHT
    elif any(domain_allowed(str(ev.url), task.allowed_domains) for ev in packet.evidence):
        breakdown["domain"] = _GRPO_DOMAIN_WEIGHT / 2
        reasons.append(f"recommended URL outside allowed domains: {packet.url}")
    else:
        reasons.append("no allowed-domain recommended or evidence URL")
        caps.append((0.70, "domain"))

    price = parse_price(packet.price)
    if price is None:
        reasons.append("price missing or unparsable")
        caps.append((0.85, "missing_price"))
    elif price <= task.max_price:
        breakdown["price"] = _GRPO_PRICE_WEIGHT
    elif price <= task.max_price * 1.10:
        breakdown["price"] = 0.08
        reasons.append(f"price {price:.2f} slightly exceeds max {task.max_price:.2f}")
        caps.append((0.75, "over_price"))
    else:
        breakdown["price"] = 0.03
        reasons.append(f"price {price:.2f} exceeds max {task.max_price:.2f}")
        caps.append((0.75, "over_price"))

    must_have_text = _must_have_text(packet)
    must_have_ratio = 1.0
    if task.must_have:
        must_have_hits = sum(1 for term in task.must_have if _term_hit(term, must_have_text))
        must_have_ratio = must_have_hits / len(task.must_have)
        breakdown["must_have"] = _GRPO_MUST_HAVE_WEIGHT * must_have_ratio
        if must_have_hits == 0:
            reasons.append("missing all must-have evidence")
            caps.append((0.60, "missing_all_must_have"))
        elif must_have_hits < len(task.must_have):
            missing = [term for term in task.must_have if not _term_hit(term, must_have_text)]
            reasons.append(f"missing must-have evidence: {', '.join(missing)}")
            caps.append((0.85, "partial_must_have"))
    else:
        breakdown["must_have"] = _GRPO_MUST_HAVE_WEIGHT

    claim_text = _claim_text(packet)
    forbidden_present = [term for term in task.must_not_have if _forbidden_hit(term, claim_text)]
    forbidden_negated = [
        term for term in task.must_not_have if not _forbidden_hit(term, claim_text) and _negated_term_hit(term, claim_text)
    ]
    if forbidden_present:
        reasons.append(f"forbidden product trait appears present: {', '.join(forbidden_present)}")
        caps.append((0.50, "forbidden_trait"))
    elif forbidden_negated:
        breakdown["must_not"] = _GRPO_MUST_NOT_WEIGHT / 2
        reasons.append(f"forbidden traits only appear negated: {', '.join(forbidden_negated)}")
    else:
        breakdown["must_not"] = _GRPO_MUST_NOT_WEIGHT

    evidence_quality, evidence_reasons = _grpo_evidence_quality(packet, task, price)
    breakdown["evidence_quality"] = _GRPO_EVIDENCE_WEIGHT * evidence_quality
    reasons.extend(evidence_reasons)
    if not packet.evidence:
        caps.append((0.65, "missing_evidence"))
    elif evidence_quality < 0.50:
        caps.append((0.80, "weak_evidence"))

    breakdown["recommendation"] = _grpo_recommendation_reward(packet, task)
    if breakdown["recommendation"] == 0.0:
        reasons.append("recommendation missing or unsafe")
    elif breakdown["recommendation"] < _GRPO_RECOMMENDATION_WEIGHT:
        reasons.append("recommendation is generic or weakly supported")

    if judge is not None:
        try:
            judge_result = judge.judge(packet, task)
            breakdown["must_have"] = _GRPO_MUST_HAVE_WEIGHT * max(must_have_ratio, judge_result.must_have_score)
            breakdown["evidence_quality"] = _GRPO_EVIDENCE_WEIGHT * judge_result.evidence_quality_score
            breakdown["recommendation"] = _GRPO_RECOMMENDATION_WEIGHT * judge_result.recommendation_score
            reasons.extend(f"llm judge: {reason}" for reason in judge_result.reasons)
            if judge_result.must_have_score >= 1.0:
                caps = [
                    cap
                    for cap in caps
                    if cap[1] not in {"missing_all_must_have", "partial_must_have"}
                ]
            if packet.evidence and judge_result.evidence_quality_score >= 0.50:
                caps = [cap for cap in caps if cap[1] != "weak_evidence"]
        except Exception as exc:
            reasons.append(f"llm judge unavailable: {exc}")

    tokens = _token_count(packet)
    if tokens <= task.token_budget:
        breakdown["compression"] = _GRPO_COMPRESSION_WEIGHT
    elif tokens <= task.token_budget * 1.25:
        breakdown["compression"] = _GRPO_COMPRESSION_WEIGHT / 2
        reasons.append(f"packet exceeds token budget {task.token_budget}")
    else:
        reasons.append(f"packet far exceeds token budget {task.token_budget}")

    score = max(0.0, min(1.0, sum(breakdown.values())))
    for cap, reason in caps:
        if score > cap:
            score = cap
        reasons.append(f"score capped at {cap:.2f}: {reason}")
    return RewardResult(score, breakdown, reasons or ["ok"], packet)


def _grpo_evidence_quality(packet: PurchasePacket, task: ShoppingTaskSpec, price: float | None) -> tuple[float, list[str]]:
    if not packet.evidence:
        return 0.0, ["missing evidence"]

    item_scores: list[float] = []
    reasons: list[str] = []
    for item in packet.evidence:
        item_score = 0.0
        quote = item.quote.strip()
        supports = item.supports.strip()
        combined = f"{quote} {supports}".lower()

        if domain_allowed(str(item.url), task.allowed_domains):
            item_score += 0.25
        else:
            reasons.append(f"evidence URL outside allowed domains: {item.url}")
        if _specific_text(quote):
            item_score += 0.25
        else:
            reasons.append("evidence quote is too vague")
        if _supports_real_claim(supports, task):
            item_score += 0.25
        else:
            reasons.append("evidence supports field does not name a task claim")
        if _evidence_overlaps_claim(combined, packet, task, price):
            item_score += 0.25
        else:
            reasons.append("evidence does not overlap price, constraints, product, seller, or availability")

        item_scores.append(item_score)

    return min(1.0, sum(item_scores) / 2.0), reasons


def _specific_text(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9$]+", text.lower())
    return len(text.strip()) >= 4 and len(tokens) >= 1


def _supports_real_claim(supports: str, task: ShoppingTaskSpec) -> bool:
    text = supports.lower()
    claim_terms = (
        "price",
        "cost",
        "under",
        "budget",
        "constraint",
        "required",
        "seller",
        "availability",
        "available",
        "delivery",
        "pickup",
        "watt",
        "pack",
        "count",
        "size",
        "type",
        "format",
    )
    return any(term in text for term in claim_terms) or any(_term_hit(term, text) for term in task.must_have)


def _evidence_overlaps_claim(
    text: str, packet: PurchasePacket, task: ShoppingTaskSpec, price: float | None
) -> bool:
    if price is not None and _term_hit(f"{price:g}", text):
        return True
    if packet.price and packet.price.lower() in text:
        return True
    if any(_term_hit(term, text) for term in task.must_have):
        return True
    if any(_term_hit(term, text) for term in task.must_not_have):
        return True
    if packet.seller and _term_hit(packet.seller, text):
        return True
    if packet.delivery_or_pickup and any(
        token in text for token in ("pickup", "delivery", "available", "shipping")
    ):
        return True
    product_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", packet.recommended_product.lower())
        if len(token) >= 4
    ]
    return any(token in text for token in product_tokens)


def _grpo_recommendation_reward(packet: PurchasePacket, task: ShoppingTaskSpec) -> float:
    recommendation = packet.recommendation.strip().lower()
    if not recommendation:
        return 0.0
    if any(term in recommendation for term in _CHECKOUT_TERMS):
        return 0.0

    has_price_reason = any(term in recommendation for term in ("price", "under", "budget", "below", "$"))
    has_constraint_reason = (
        any(_term_hit(term, recommendation) for term in task.must_have)
        or any(term in recommendation for term in ("constraint", "satisfies", "meets", "required", "fits"))
    )
    if has_price_reason and has_constraint_reason:
        return _GRPO_RECOMMENDATION_WEIGHT
    if len(recommendation) >= 12:
        return _GRPO_RECOMMENDATION_WEIGHT * 0.6
    return _GRPO_RECOMMENDATION_WEIGHT * 0.3


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
