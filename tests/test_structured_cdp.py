import json

import pytest

from cart_scout.structured_cdp import (
    extract_first_json_object,
    parse_structured_action,
    retailer_search_url,
    strip_reasoning,
)


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
