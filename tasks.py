"""CartScout HUD taskset.

`hud eval tasks.py ...` discovers the public `tasks` list. These are safe,
low-risk shopping research tasks; none requires checkout.
"""

from env import env, shopping_context_task  # noqa: F401


def _task(
    slug: str,
    instruction: str,
    max_price: float,
    must_have: list[str],
    must_not_have: list[str],
    allowed_domains: list[str] | None = None,
):
    task = shopping_context_task(
        task_id=slug,
        instruction=instruction,
        max_price=max_price,
        must_have=must_have,
        must_not_have=must_not_have,
        allowed_domains=allowed_domains or ["target.com", "amazon.com"],
        token_budget=750,
        require_cart=False,
    )
    task.slug = slug
    return task


tasks = [
    _task(
        "usb-c-charger-30w-under-40",
        "Find a USB-C charger for a MacBook Air M2 under $40. Must support USB-C Power Delivery and at least 30W. Prepare a recommendation only.",
        40.0,
        ["USB-C", "Power Delivery", "30W"],
        ["Lightning", "MagSafe only"],
    ),
    _task(
        "unscented-liquid-he-detergent-under-25",
        "Find unscented liquid HE laundry detergent under $25. It must not be pods. Prepare a recommendation only.",
        25.0,
        ["unscented", "liquid", "HE"],
        ["pods", "scented"],
    ),
    _task(
        "aaa-batteries-20-pack-under-25",
        "Find AAA batteries, pack of 20 or more, under $25. Prepare a recommendation only.",
        25.0,
        ["AAA", "20"],
        ["AA", "charger bundle"],
    ),
    _task(
        "letter-printer-paper-500-under-15",
        "Find letter-size printer paper, at least 500 sheets, under $15. Prepare a recommendation only.",
        15.0,
        ["letter", "500 sheets"],
        ["photo paper", "legal size"],
    ),
    _task(
        "basket-coffee-filters-under-10",
        "Find basket coffee filters under $10. They should be compatible with basket-style coffee makers. Prepare a recommendation only.",
        10.0,
        ["basket", "coffee filters"],
        ["cone", "permanent filter"],
    ),
    _task(
        "non-glass-food-storage-under-20",
        "Find non-glass food storage containers under $20. Prepare a recommendation only.",
        20.0,
        ["food storage", "plastic"],
        ["glass"],
    ),
    _task(
        "storage-bin-lid-under-20",
        "Find a storage bin with a lid under $20. Prepare a recommendation only.",
        20.0,
        ["storage bin", "lid"],
        ["no lid"],
    ),
    _task(
        "usb-c-cable-6ft-under-15",
        "Find a USB-C cable at least 6 feet long under $15. Prepare a recommendation only.",
        15.0,
        ["USB-C", "6 ft"],
        ["Lightning", "Micro USB"],
    ),
    _task(
        "dish-soap-free-clear-under-10",
        "Find free-and-clear dish soap under $10. Prepare a recommendation only.",
        10.0,
        ["dish soap", "free", "clear"],
        ["dishwasher detergent", "pods"],
    ),
    _task(
        "notebook-college-ruled-under-8",
        "Find a college-ruled notebook under $8. Prepare a recommendation only.",
        8.0,
        ["college ruled", "notebook"],
        ["wide ruled", "planner"],
    ),
]
