import json

import pytest

import cart_scout.reward as reward_module
from cart_scout.reward import (
    FireworksJudgeConfig,
    FireworksQwenJudge,
    LLMJudgeResult,
    domain_allowed,
    parse_price,
    progress_potential,
    score_grpo_packet,
    score_purchase_packet,
)
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


def test_grpo_scores_perfect_packet(charger_task):
    result = score_grpo_packet(packet(), charger_task)
    assert result.score == pytest.approx(1.0)
    assert result.breakdown["evidence_quality"] == pytest.approx(0.20)
    assert result.reasons == ["ok"]


def test_grpo_invalid_json_returns_zero(charger_task):
    result = score_grpo_packet("not json", charger_task)
    assert result.score == 0.0
    assert "invalid PurchasePacket JSON" in result.reasons[0]


def test_grpo_checkout_login_payment_returns_zero(charger_task):
    assert score_grpo_packet(packet(attempted_checkout=True), charger_task).score == 0.0
    assert score_grpo_packet(packet(recommendation="Sign in to buy now."), charger_task).score == 0.0
    assert score_grpo_packet(packet(stop_before_checkout=False), charger_task).score == 0.0


def test_grpo_wrong_domain_cap(charger_task):
    result = score_grpo_packet(
        packet(
            url="https://example.com/item",
            evidence=[
                {
                    "url": "https://example.com/item",
                    "quote": "30W USB-C Power Delivery",
                    "supports": "wattage and USB-C PD",
                },
                {
                    "url": "https://example.com/item",
                    "quote": "$19.99",
                    "supports": "price under $40",
                },
            ],
        ),
        charger_task,
    )
    assert result.score <= 0.70
    assert any("domain" in reason for reason in result.reasons)


def test_grpo_wrong_recommendation_domain_gets_partial_domain_from_evidence(charger_task):
    result = score_grpo_packet(packet(url="https://example.com/item"), charger_task)
    assert result.breakdown["domain"] == pytest.approx(0.05)
    assert result.score > 0.70


def test_grpo_over_budget_price_cap(charger_task):
    result = score_grpo_packet(packet(price="$49.99"), charger_task)
    assert result.score <= 0.75
    assert result.breakdown["price"] == pytest.approx(0.03)


def test_grpo_missing_price_cap(charger_task):
    result = score_grpo_packet(packet(price=None), charger_task)
    assert result.score <= 0.85
    assert result.breakdown["price"] == 0.0


def test_grpo_partial_must_have_gets_proportional_reward(charger_task):
    result = score_grpo_packet(
        packet(
            recommended_product="Anker USB-C Charger",
            constraints_met=["USB-C"],
            evidence=[
                {
                    "url": "https://www.target.com/p/example",
                    "quote": "USB-C charger",
                    "supports": "required connector",
                },
                {
                    "url": "https://www.target.com/p/example",
                    "quote": "$19.99",
                    "supports": "price under $40",
                },
            ],
            recommendation="Recommend this USB-C charger because it is below budget.",
        ),
        charger_task,
    )
    assert result.breakdown["must_have"] == pytest.approx(0.25 / 3)
    assert result.score <= 0.85


def test_grpo_forbidden_trait_present_caps_score(charger_task):
    result = score_grpo_packet(
        packet(recommended_product="Anker Lightning USB-C Charger"),
        charger_task,
    )
    assert result.score <= 0.50
    assert result.breakdown["must_not"] == 0.0


def test_grpo_negated_forbidden_trait_gets_partial_must_not_credit():
    detergent_task = ShoppingTaskSpec(
        task_id="detergent",
        instruction="Find unscented liquid HE laundry detergent under $25.",
        allowed_domains=["target.com"],
        max_price=25,
        must_have=["unscented", "liquid", "HE"],
        must_not_have=["pods"],
    )
    result = score_grpo_packet(
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
                "recommendation": "Recommend this detergent because it meets constraints and is below budget.",
                "stop_before_checkout": True,
                "cart_prepared": False,
                "attempted_checkout": False,
            }
        ),
        detergent_task,
    )
    assert result.breakdown["must_not"] == pytest.approx(0.075)
    assert result.score > 0.90


def test_grpo_no_evidence_cap(charger_task):
    result = score_grpo_packet(packet(evidence=[]), charger_task)
    assert result.score <= 0.65
    assert result.breakdown["evidence_quality"] == 0.0


def test_grpo_weak_evidence_scores_below_specific_evidence(charger_task):
    weak = score_grpo_packet(
        packet(
            evidence=[
                {
                    "url": "https://example.com/item",
                    "quote": "ok",
                    "supports": "ok",
                }
            ]
        ),
        charger_task,
    )
    strong = score_grpo_packet(packet(), charger_task)
    assert weak.breakdown["evidence_quality"] < strong.breakdown["evidence_quality"]
    assert weak.score <= 0.80


def test_grpo_long_packet_loses_compression_reward(charger_task):
    small_budget_task = charger_task.model_copy(update={"token_budget": 10})
    result = score_grpo_packet(packet(), small_budget_task)
    assert result.breakdown["compression"] == 0.0
    assert result.score < 1.0


class FakeJudge:
    def __init__(self, result: LLMJudgeResult):
        self.result = result

    def judge(self, packet, task):
        return self.result


def test_grpo_llm_judge_adjusts_semantic_components(charger_task):
    judged = score_grpo_packet(
        packet(
            recommended_product="Anker USB-C Charger",
            constraints_met=["USB-C"],
            evidence=[
                {
                    "url": "https://www.target.com/p/example",
                    "quote": "USB-C charger",
                    "supports": "required connector",
                }
            ],
            recommendation="Recommend this charger.",
        ),
        charger_task,
        judge=FakeJudge(
            LLMJudgeResult(
                must_have_score=1.0,
                evidence_quality_score=0.8,
                recommendation_score=0.9,
                reasons=["PD wording semantically covers Power Delivery and 30W claim is implied by evidence"],
            )
        ),
    )
    unjudged = score_grpo_packet(
        packet(
            recommended_product="Anker USB-C Charger",
            constraints_met=["USB-C"],
            evidence=[
                {
                    "url": "https://www.target.com/p/example",
                    "quote": "USB-C charger",
                    "supports": "required connector",
                }
            ],
            recommendation="Recommend this charger.",
        ),
        charger_task,
    )
    assert judged.score > unjudged.score
    assert judged.breakdown["must_have"] == pytest.approx(0.25)
    assert judged.breakdown["evidence_quality"] == pytest.approx(0.16)
    assert judged.breakdown["recommendation"] == pytest.approx(0.045)
    assert any(reason.startswith("llm judge:") for reason in judged.reasons)


def test_grpo_llm_judge_failure_falls_back_to_deterministic(charger_task):
    class BrokenJudge:
        def judge(self, packet, task):
            raise RuntimeError("offline")

    result = score_grpo_packet(packet(), charger_task, judge=BrokenJudge())
    assert result.score == pytest.approx(1.0)
    assert any("llm judge unavailable" in reason for reason in result.reasons)


def test_fireworks_qwen_judge_requires_api_key(monkeypatch, charger_task):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    judge = FireworksQwenJudge()
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        judge.judge(PurchasePacket.model_validate_json(packet()), charger_task)


def test_fireworks_qwen_judge_calls_chat_completions(monkeypatch, charger_task):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "must_have_score": 0.75,
                                        "evidence_quality_score": 0.5,
                                        "recommendation_score": 1.0,
                                        "reasons": ["reasonable semantic fit"],
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr(reward_module.urlrequest, "urlopen", fake_urlopen)
    judge = FireworksQwenJudge(
        FireworksJudgeConfig(
            api_key="fw-test",
            base_url="https://api.fireworks.ai/inference/v1",
            timeout_s=12.0,
        )
    )

    result = judge.judge(PurchasePacket.model_validate_json(packet()), charger_task)

    assert result.must_have_score == pytest.approx(0.75)
    assert result.evidence_quality_score == pytest.approx(0.5)
    assert result.recommendation_score == pytest.approx(1.0)
    assert result.reasons == ["reasonable semantic fit"]
    assert len(calls) == 1
    req, timeout = calls[0]
    assert req.full_url == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert timeout == pytest.approx(12.0)
    assert req.headers["Authorization"] == "Bearer fw-test"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["model"] == "accounts/fireworks/models/qwen3p7-max"
    assert payload["temperature"] == 0.0
    assert payload["messages"][0]["role"] == "system"
