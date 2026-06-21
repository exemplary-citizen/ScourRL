from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, ValidationError, field_validator


ActionName = Literal[
    "open_url",
    "search_retailer",
    "click_ref",
    "extract_page",
    "find_text",
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


def compact_text(text: str, limit: int = 6000) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rsplit(" ", 1)[0] + " ..."


def build_controller_prompt(task_prompt: str) -> str:
    return f"""You are controlling a browser through a fixed JSON action API.

Return exactly one JSON object per turn. Do not use Markdown. Do not include prose.

Available actions:
- {{"action":"open_url","args":{{"url":"https://..."}}}}
- {{"action":"search_retailer","args":{{"retailer":"target|amazon","query":"search query"}}}}
- {{"action":"click_ref","args":{{"ref":0}}}}
- {{"action":"extract_page","args":{{}}}}
- {{"action":"find_text","args":{{"pattern":"text to find"}}}}
- {{"action":"emit_packet","args":{{"packet":{{...required task JSON...}}}}}}
- {{"action":"stop","args":{{"reason":"why stopping without packet"}}}}

Use search_retailer instead of general web search. Click product links only. Never sign in, check out, or enter payment details.
When you have one plausible product with price and evidence, call emit_packet with the exact task JSON object.

Task prompt:
{task_prompt}
"""
