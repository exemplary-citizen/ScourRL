import json

from fireworks_rft.evaluator import score_packet


def test_fireworks_score_packet():
    text = json.dumps(
        {
            "query": "USB-C charger under $40",
            "recommended_product": "Anker 30W USB-C Charger",
            "retailer": "Target",
            "url": "https://www.target.com/p/example",
            "price": "$19.99",
            "delivery_or_pickup": "Pickup available",
            "seller": "Target",
            "constraints_met": ["USB-C", "Power Delivery", "30W"],
            "constraints_uncertain": [],
            "evidence": [
                {
                    "url": "https://www.target.com/p/example",
                    "quote": "30W USB-C Power Delivery",
                    "supports": "wattage",
                }
            ],
            "recommendation": "Recommend this below-budget charger.",
            "stop_before_checkout": True,
        }
    )
    score, reason = score_packet(
        text,
        {
            "max_price": 40,
            "must_have": ["USB-C", "Power Delivery", "30W"],
            "must_not_have": ["Lightning"],
            "allowed_domains": ["target.com", "amazon.com"],
        },
    )
    assert score > 0.85
    assert "breakdown" in reason
