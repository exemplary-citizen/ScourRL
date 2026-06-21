from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote_plus, urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator

from cart_scout.reward import parse_price
from cart_scout.schema import ShoppingTaskSpec


ActionName = Literal[
    "open_url",
    "search_retailer",
    "click_ref",
    "fill_ref",
    "press",
    "scroll",
    "go_back",
    "extract_page",
    "find_text",
    "screenshot",
    "emit_packet",
    "stop",
]


class StructuredAction(BaseModel):
    action: ActionName
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("args")
    @classmethod
    def ensure_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("args must be an object")
        return value


@dataclass(frozen=True)
class ParsedAction:
    action: StructuredAction | None
    raw: str
    error: str | None = None


@dataclass(frozen=True)
class ProgressState:
    score: float
    allowed_domain: bool
    price_found: bool
    must_have_hits: tuple[str, ...]
    evidence_source: str
    evidence_like: bool
    unsafe_attempts: int
    repeated_actions: int
    no_op_actions: int
    visited_urls: int


@dataclass(frozen=True)
class ShapingStep:
    step: int
    action: str
    previous_score: float
    next_score: float
    dense_reward: float
    reasons: tuple[str, ...]
    state: ProgressState


def parse_structured_action(text: str) -> ParsedAction:
    """Extract a single structured action from model output.

    Open-source models frequently leak thinking tags, Markdown fences, or prose before
    the JSON object. The trainable contract is still one JSON action, but this parser
    is intentionally forgiving so eval failures reflect browser policy quality rather
    than punctuation around the action.
    """
    raw = text.strip()
    stripped = strip_reasoning(raw)
    candidate = extract_first_json_object(stripped)
    if candidate is None:
        return ParsedAction(None, raw=raw, error="no JSON object found")
    try:
        data = json.loads(candidate)
        return ParsedAction(StructuredAction.model_validate(data), raw=raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return ParsedAction(None, raw=raw, error=str(exc))


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def retailer_search_url(retailer: str, query: str) -> str:
    normalized = retailer.lower().strip()
    encoded = quote_plus(query)
    if normalized in {"target", "target.com"}:
        return f"https://www.target.com/s?searchTerm={encoded}"
    if normalized in {"amazon", "amazon.com"}:
        return f"https://www.amazon.com/s?k={encoded}"
    raise ValueError(f"unsupported retailer: {retailer}")


def observation_progress(
    observation: dict[str, Any],
    task: ShoppingTaskSpec,
    *,
    unsafe_attempts: int = 0,
    repeated_actions: int = 0,
    no_op_actions: int = 0,
    visited_urls: int = 0,
) -> ProgressState:
    """Measure task-relevant information visible in an observation.

    This is not the terminal reward. It is a debugging/training signal for whether
    browser actions are improving the agent's state.
    """
    url = str(observation.get("url") or "")
    page_text = _page_evidence_text(observation)
    refs_text = _refs_text(observation)
    evidence_text = " ".join([page_text, refs_text]).lower()
    source = _evidence_source(url)
    allowed_domain = _domain_allowed(url, task.allowed_domains)
    price_found = parse_price(evidence_text) is not None
    must_have_hits = tuple(term for term in task.must_have if _term_hit(term, evidence_text))
    evidence_like = bool(
        price_found
        and (must_have_hits or _product_like_text(evidence_text))
        and source != "query"
    )

    score = 0.0
    if allowed_domain:
        score += 0.15
    if price_found and source != "query":
        score += 0.20
    if task.must_have:
        source_weight = 1.0 if source == "product" else 0.55
        score += 0.35 * len(must_have_hits) / len(task.must_have) * source_weight
    else:
        score += 0.35
    if evidence_like:
        score += 0.20
    if visited_urls > 1:
        score += 0.10
    score -= min(0.30, 0.10 * unsafe_attempts)
    score -= min(0.20, 0.05 * repeated_actions)
    score -= min(0.20, 0.05 * no_op_actions)

    return ProgressState(
        score=max(0.0, min(1.0, score)),
        allowed_domain=allowed_domain,
        price_found=price_found,
        must_have_hits=must_have_hits,
        evidence_source=source,
        evidence_like=evidence_like,
        unsafe_attempts=unsafe_attempts,
        repeated_actions=repeated_actions,
        no_op_actions=no_op_actions,
        visited_urls=visited_urls,
    )


def shaping_reward(
    *,
    step: int,
    action_name: str,
    previous: ProgressState,
    current: ProgressState,
    step_cost: float = 0.01,
) -> ShapingStep:
    delta = current.score - previous.score
    penalties = 0.0
    reasons: list[str] = []
    if current.unsafe_attempts > previous.unsafe_attempts:
        penalties += 0.25
        reasons.append("unsafe_attempt")
    if current.repeated_actions > previous.repeated_actions:
        penalties += 0.05
        reasons.append("repeated_action")
    if current.no_op_actions > previous.no_op_actions:
        penalties += 0.05
        reasons.append("no_op_action")
    if delta > 0:
        reasons.append("progress")
    if not reasons:
        reasons.append("step_cost")

    return ShapingStep(
        step=step,
        action=action_name,
        previous_score=previous.score,
        next_score=current.score,
        dense_reward=delta - step_cost - penalties,
        reasons=tuple(reasons),
        state=current,
    )


def compact_text(text: str, limit: int = 6000) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rsplit(" ", 1)[0] + " ..."


def _page_evidence_text(observation: dict[str, Any]) -> str:
    return " ".join(
        [
            str(observation.get("title") or ""),
            str(observation.get("text") or ""),
        ]
    ).lower()


def _refs_text(observation: dict[str, Any]) -> str:
    return " ".join(str(ref.get("text", "")) for ref in observation.get("refs", []) if isinstance(ref, dict)).lower()


def _evidence_source(url: str) -> str:
    path = (urlparse(url).path or "").lower()
    query = (urlparse(url).query or "").lower()
    if "/p/" in path or "/dp/" in path or "/gp/product/" in path:
        return "product"
    if "search" in path or "searchterm=" in query or "k=" in query:
        return "search"
    return "page"


def _domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    allowed = [domain.lower().lstrip(".") for domain in allowed_domains]
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed)


def _term_hit(term: str, text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", term.lower())
    if not tokens:
        return False
    pattern = r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(map(re.escape, tokens)) + r"(?![a-z0-9])"
    if re.search(pattern, text):
        return True
    collapsed = re.sub(r"[^a-z0-9]+", "", term.lower())
    target = re.sub(r"[^a-z0-9]+", "", text)
    return len(collapsed) > 3 and collapsed in target


def _product_like_text(text: str) -> bool:
    product_terms = ("add to cart", "shipping", "pickup", "delivery", "ratings", "reviews", "sold by", "brand")
    return any(term in text for term in product_terms)


def build_controller_prompt(task_prompt: str) -> str:
    return f"""You are controlling a browser through a fixed JSON action API.

Return exactly one JSON object per turn. Do not use Markdown. Do not include prose.

Available actions:
- {{"action":"open_url","args":{{"url":"https://..."}}}}
- {{"action":"search_retailer","args":{{"retailer":"target|amazon","query":"search query"}}}}
- {{"action":"click_ref","args":{{"ref":0}}}}
- {{"action":"fill_ref","args":{{"ref":0,"text":"value to type"}}}}
- {{"action":"press","args":{{"key":"Enter"}}}}
- {{"action":"scroll","args":{{"direction":"down|up","amount":700}}}}
- {{"action":"go_back","args":{{}}}}
- {{"action":"extract_page","args":{{}}}}
- {{"action":"find_text","args":{{"pattern":"text to find"}}}}
- {{"action":"screenshot","args":{{}}}}
- {{"action":"emit_packet","args":{{"packet":{{...required task JSON...}}}}}}
- {{"action":"stop","args":{{"reason":"why stopping without packet"}}}}

Use search_retailer for Target/Amazon shopping searches. Use scroll when useful items may be below the fold. Never sign in, check out, or enter payment details.
When you have one plausible product with price and evidence, call emit_packet with the exact task JSON object.

Task prompt:
{task_prompt}
"""
