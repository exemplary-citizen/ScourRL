import json

import pytest

from cart_scout.structured_cdp import (
    extract_first_json_object,
    observation_progress,
    parse_structured_action,
    retailer_search_url,
    shaping_reward,
    strip_reasoning,
)
from cart_scout.schema import ShoppingTaskSpec


def test_parse_structured_action_strips_think_and_markdown():
    raw = """<think>I should search Amazon.</think>
```json
{"action": "search_retailer", "args": {"retailer": "amazon", "query": "usb c charger 30w power delivery"}}
```"""

    parsed = parse_structured_action(raw)

    assert parsed.error is None
    assert parsed.action is not None
    assert parsed.action.action == "search_retailer"
    assert parsed.action.args["retailer"] == "amazon"


def test_parse_structured_action_extracts_json_from_prose():
    packet = {
        "query": "usb-c charger",
        "recommended_product": "Example 30W USB-C PD Charger",
    }
    raw = "Here is the action:\n" + json.dumps({"action": "emit_packet", "args": {"packet": packet}})

    parsed = parse_structured_action(raw)

    assert parsed.error is None
    assert parsed.action is not None
    assert parsed.action.action == "emit_packet"
    assert parsed.action.args["packet"]["recommended_product"] == "Example 30W USB-C PD Charger"


def test_parse_structured_action_reports_invalid_action():
    parsed = parse_structured_action('{"action": "type_into_search_box", "args": {}}')

    assert parsed.action is None
    assert parsed.error


def test_parse_structured_browser_primitives():
    for raw, expected in [
        ('{"action": "scroll", "args": {"direction": "down", "amount": 900}}', "scroll"),
        ('{"action": "fill_ref", "args": {"ref": 3, "text": "usb c charger"}}', "fill_ref"),
        ('{"action": "press", "args": {"key": "Enter"}}', "press"),
        ('{"action": "go_back", "args": {}}', "go_back"),
        ('{"action": "screenshot", "args": {}}', "screenshot"),
    ]:
        parsed = parse_structured_action(raw)
        assert parsed.error is None
        assert parsed.action is not None
        assert parsed.action.action == expected


def test_extract_first_json_object_handles_nested_strings():
    text = 'prefix {"action":"find_text","args":{"pattern":"USB-C {PD}"}} suffix'

    assert extract_first_json_object(text) == '{"action":"find_text","args":{"pattern":"USB-C {PD}"}}'


def test_strip_reasoning_keeps_json_body():
    assert strip_reasoning("<think>x</think>{\"action\":\"stop\",\"args\":{}}") == '{"action":"stop","args":{}}'


def test_retailer_search_url():
    assert retailer_search_url("amazon", "usb c charger") == "https://www.amazon.com/s?k=usb+c+charger"
    assert retailer_search_url("target.com", "printer paper") == "https://www.target.com/s?searchTerm=printer+paper"
    with pytest.raises(ValueError):
        retailer_search_url("google", "usb c charger")


def test_observation_progress_rewards_information_not_actions():
    task = ShoppingTaskSpec(
        task_id="usb",
        instruction="Find a USB-C charger under $40.",
        allowed_domains=["target.com", "amazon.com"],
        max_price=40,
        must_have=["USB-C", "Power Delivery", "30W"],
        must_not_have=["Lightning"],
    )
    empty = observation_progress(
        {
            "url": "https://www.target.com/s?searchTerm=usb+c+charger",
            "title": "Target search",
            "text": "Search results",
            "refs": [],
        },
        task,
        visited_urls=1,
    )
    richer = observation_progress(
        {
            "url": "https://www.target.com/p/example",
            "title": "Anker 30W USB-C Charger",
            "text": "$19.99 Rapid USB-C Power Delivery 30W wall charger. Shipping available.",
            "refs": [{"text": "Anker 30W USB-C Charger $19.99"}],
        },
        task,
        visited_urls=2,
    )

    assert richer.score > empty.score
    assert richer.price_found
    assert richer.must_have_hits == ("USB-C", "Power Delivery", "30W")
    assert richer.evidence_source == "product"
    assert richer.evidence_like


def test_observation_progress_does_not_count_query_terms_as_evidence():
    task = ShoppingTaskSpec(
        task_id="usb",
        instruction="Find a USB-C charger under $40.",
        allowed_domains=["target.com", "amazon.com"],
        max_price=40,
        must_have=["USB-C", "Power Delivery", "30W"],
        must_not_have=["Lightning"],
    )

    search_only = observation_progress(
        {
            "url": "https://www.target.com/s?searchTerm=USB-C+charger+30W+Power+Delivery",
            "title": "Target search",
            "text": "Search results",
            "refs": [],
        },
        task,
        visited_urls=1,
    )

    assert search_only.evidence_source == "search"
    assert search_only.must_have_hits == ()
    assert not search_only.price_found
    assert not search_only.evidence_like
    assert search_only.score < 0.3


def test_shaping_reward_penalizes_unsafe_and_repeated_actions():
    task = ShoppingTaskSpec(
        task_id="usb",
        instruction="Find a USB-C charger under $40.",
        allowed_domains=["target.com"],
        max_price=40,
        must_have=["USB-C"],
        must_not_have=[],
    )
    previous = observation_progress({"url": "https://www.target.com", "text": "", "refs": []}, task)
    current = observation_progress(
        {"url": "https://www.target.com", "text": "", "refs": []},
        task,
        unsafe_attempts=1,
        repeated_actions=1,
    )
    shape = shaping_reward(step=2, action_name="click_ref", previous=previous, current=current)

    assert shape.dense_reward < 0
    assert "unsafe_attempt" in shape.reasons
    assert "repeated_action" in shape.reasons
