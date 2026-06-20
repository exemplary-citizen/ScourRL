import json

import env as M


async def test_shopping_context_template_grades(monkeypatch):
    async def noop(url):
        return None

    monkeypatch.setattr(M, "_navigate", noop)

    gen = M.shopping_context_task.func(
        task_id="usb",
        instruction="Find a USB-C charger under $40.",
        max_price=40,
        must_have=["USB-C", "Power Delivery", "30W"],
        must_not_have=["Lightning"],
        allowed_domains=["target.com"],
    )
    prompt = await gen.asend(None)
    assert "Return exactly one JSON object" in prompt

    answer = json.dumps(
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
                },
                {
                    "url": "https://www.target.com/p/example",
                    "quote": "$19.99",
                    "supports": "price",
                },
            ],
            "recommendation": "Recommend this item.",
            "stop_before_checkout": True,
        }
    )
    reward = await gen.asend(answer)
    assert reward > 0.9
