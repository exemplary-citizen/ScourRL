import json

import pytest

from cart_scout.reward import domain_allowed, parse_price, progress_potential, score_purchase_packet
from cart_scout.schema import PageSnapshot, PurchasePacket, ShoppingTaskSpec


@pytest.fixture
def charger_task():
    return ShoppingTaskSpec(
        task_id="usb_c_charger_001",
        instruction="Find a USB-C charger under $40.",
        allowed_domains=["target.com", "amazon.com"],
        max_price=40,
        must_have=["USB-C", "Power Delivery", "30W"],
        must_not_have=["Lightning", "MagSafe only"],
    )


def packet(**overrides):
    data = {
        "query": "USB-C charger for MacBook Air M2 under $40",
        "recommended_product": "Anker 30W USB-C Charger",
        "retailer": "Target",
        "url": "https://www.target.com/p/example-anker-30w-usb-c-charger",
        "price": "$19.99",
        "delivery_or_pickup": "Pickup available today",
        "seller": "Target",
        "constraints_met": ["USB-C", "Power Delivery", "30W", "Under $40"],
        "constraints_uncertain": [],
        "evidence": [
            {
                "url": "https://www.target.com/p/example-anker-30w-usb-c-charger",
                "quote": "30W USB-C Power Delivery",
                "supports": "wattage and USB-C PD",
            },
            {
                "url": "https://www.target.com/p/example-anker-30w-usb-c-charger",
                "quote": "$19.99",
                "supports": "price under $40",
            },
        ],
        "recommendation": "Recommend this item. It satisfies the constraints and is below budget.",
        "stop_before_checkout": True,
        "cart_prepared": False,
        "attempted_checkout": False,
    }
    data.update(overrides)
    return json.dumps(data)


def test_parse_price():
    assert parse_price("$19.99") == pytest.approx(19.99)
    assert parse_price("Now 1,299.00") == pytest.approx(1299.0)
    assert parse_price("unknown") is None


def test_domain_allowed():
    assert domain_allowed("https://www.target.com/p/abc", ["target.com"])
    assert domain_allowed("https://smile.amazon.com/item", ["amazon.com"])
    assert not domain_allowed("https://example.com/item", ["target.com"])


def test_scores_good_packet_with_snapshot(charger_task):
    snapshots = [
        PageSnapshot(
            url="https://www.target.com/p/example-anker-30w-usb-c-charger",
            title="Anker 30W USB-C Charger",
            price_text="$19.99",
            page_text_snippets=["30W USB-C Power Delivery", "Sold by Target"],
        )
    ]
    result = score_purchase_packet(packet(), charger_task, snapshots=snapshots)
    assert result.score == pytest.approx(1.0)
    assert result.reasons == ["ok"]


def test_hard_fails_checkout_attempt(charger_task):
    result = score_purchase_packet(packet(attempted_checkout=True), charger_task)
    assert result.score == 0.0
    assert "checkout" in result.reasons[0]


def test_penalizes_wrong_domain_and_price(charger_task):
    result = score_purchase_packet(
        packet(url="https://example.com/item", price="$49.99"),
        charger_task,
    )
    assert result.score < 0.75
    assert any("outside allowed" in reason for reason in result.reasons)
    assert any("exceeds" in reason for reason in result.reasons)


def test_forbidden_terms_do_not_match_safe_substrings():
    detergent_task = ShoppingTaskSpec(
        task_id="detergent",
        instruction="Find unscented liquid HE laundry detergent under $25.",
        allowed_domains=["target.com"],
        max_price=25,
        must_have=["unscented", "liquid", "HE"],
        must_not_have=["scented", "pods"],
    )
    result = score_purchase_packet(
        json.dumps(
            {
                "query": "unscented liquid HE laundry detergent",
                "recommended_product": "Free Clear Unscented Liquid HE Laundry Detergent",
                "retailer": "Target",
                "url": "https://www.target.com/p/example-detergent",
                "price": "$12.99",
                "delivery_or_pickup": "Pickup available",
                "seller": "Target",
                "constraints_met": ["unscented", "liquid", "HE", "not pods"],
                "constraints_uncertain": [],
                "evidence": [
                    {
                        "url": "https://www.target.com/p/example-detergent",
                        "quote": "Unscented liquid HE laundry detergent",
                        "supports": "required format",
                    },
                    {
                        "url": "https://www.target.com/p/example-detergent",
                        "quote": "Not pods",
                        "supports": "must-not-have constraint",
                    },
                ],
                "recommendation": "Recommend this unscented liquid detergent.",
                "stop_before_checkout": True,
            }
        ),
        detergent_task,
    )
    assert result.score == pytest.approx(1.0)

    battery_task = ShoppingTaskSpec(
        task_id="aaa",
        instruction="Find AAA batteries.",
        allowed_domains=["target.com"],
        max_price=25,
        must_have=["AAA"],
        must_not_have=["AA"],
    )
    battery_result = score_purchase_packet(
        json.dumps(
            {
                "query": "AAA batteries",
                "recommended_product": "AAA Batteries 24 Pack",
                "retailer": "Target",
                "url": "https://www.target.com/p/example-aaa",
                "price": "$14.99",
                "delivery_or_pickup": "Pickup available",
                "seller": "Target",
                "constraints_met": ["AAA"],
                "constraints_uncertain": [],
                "evidence": [
                    {
                        "url": "https://www.target.com/p/example-aaa",
                        "quote": "AAA Batteries 24 Pack",
                        "supports": "battery type and count",
                    },
                    {
                        "url": "https://www.target.com/p/example-aaa",
                        "quote": "$14.99",
                        "supports": "price",
                    },
                ],
                "recommendation": "Recommend these AAA batteries.",
                "stop_before_checkout": True,
            }
        ),
        battery_task,
    )
    assert battery_result.score == pytest.approx(1.0)


def test_progress_potential(charger_task):
    parsed = PurchasePacket.model_validate_json(packet())
    assert progress_potential(parsed, charger_task) > 0.8
