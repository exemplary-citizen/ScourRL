from __future__ import annotations

import random
import re
from dataclasses import dataclass

from cart_scout.schema import ShoppingTaskSpec


DEFAULT_ALLOWED_DOMAINS = ("target.com", "amazon.com")


@dataclass(frozen=True)
class TaskSeed:
    category: str
    slug: str
    instruction: str
    max_price: float
    must_have: tuple[str, ...]
    must_not_have: tuple[str, ...]
    allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS


_TASK_ROWS: tuple[tuple[str, str, str, float, tuple[str, ...], tuple[str, ...]], ...] = (
    ("electronics", "usb-c-charger-30w-under-40", "Find a USB-C charger for a MacBook Air M2 under $40. Must support USB-C Power Delivery and at least 30W. Prepare a recommendation only.", 40.0, ("USB-C", "Power Delivery", "30W"), ("Lightning", "MagSafe only")),
    ("electronics", "usb-c-cable-6ft-under-15", "Find a USB-C cable at least 6 feet long under $15. Prepare a recommendation only.", 15.0, ("USB-C", "6 ft"), ("Lightning", "Micro USB")),
    ("electronics", "hdmi-cable-4k-under-12", "Find an HDMI cable under $12 that supports 4K video. Prepare a recommendation only.", 12.0, ("HDMI", "4K"), ("DisplayPort", "VGA")),
    ("electronics", "wireless-mouse-usb-receiver-under-25", "Find a wireless mouse under $25 with a USB receiver included. Prepare a recommendation only.", 25.0, ("wireless", "USB receiver"), ("wired", "Bluetooth only")),
    ("electronics", "wired-keyboard-under-30", "Find a wired full-size keyboard under $30. Prepare a recommendation only.", 30.0, ("wired", "keyboard"), ("wireless", "gaming bundle")),
    ("electronics", "laptop-sleeve-13-inch-under-25", "Find a 13-inch laptop sleeve under $25. Prepare a recommendation only.", 25.0, ("13", "laptop sleeve"), ("15 inch", "backpack")),
    ("electronics", "aaa-batteries-20-pack-under-25", "Find AAA batteries, pack of 20 or more, under $25. Prepare a recommendation only.", 25.0, ("AAA", "20"), ("AA", "charger bundle")),
    ("electronics", "aa-batteries-24-pack-under-25", "Find AA batteries, pack of 24 or more, under $25. Prepare a recommendation only.", 25.0, ("AA", "24"), ("AAA", "charger bundle")),
    ("electronics", "power-strip-6-outlet-under-20", "Find a 6-outlet power strip under $20. Prepare a recommendation only.", 20.0, ("6 outlet", "power strip"), ("extension cord only", "surge protector over $20")),
    ("electronics", "phone-stand-adjustable-under-15", "Find an adjustable phone stand under $15. Prepare a recommendation only.", 15.0, ("adjustable", "phone stand"), ("car mount", "tablet only")),
    ("office", "letter-printer-paper-500-under-15", "Find letter-size printer paper, at least 500 sheets, under $15. Prepare a recommendation only.", 15.0, ("letter", "500 sheets"), ("photo paper", "legal size")),
    ("office", "college-ruled-notebook-under-8", "Find a college-ruled notebook under $8. Prepare a recommendation only.", 8.0, ("college ruled", "notebook"), ("wide ruled", "planner")),
    ("office", "black-gel-pens-under-12", "Find black gel pens under $12. Prepare a recommendation only.", 12.0, ("black", "gel pens"), ("blue", "ballpoint only")),
    ("office", "assorted-highlighters-under-10", "Find assorted color highlighters under $10. Prepare a recommendation only.", 10.0, ("highlighters", "assorted"), ("markers", "single color")),
    ("office", "sticky-notes-3x3-under-8", "Find 3x3 sticky notes under $8. Prepare a recommendation only.", 8.0, ("3x3", "sticky notes"), ("index cards", "page flags only")),
    ("office", "manila-folders-100-under-15", "Find manila file folders, pack of 100, under $15. Prepare a recommendation only.", 15.0, ("manila", "100"), ("hanging folders", "colored only")),
    ("office", "binder-1-inch-under-8", "Find a 1-inch three-ring binder under $8. Prepare a recommendation only.", 8.0, ("1 inch", "three-ring binder"), ("2 inch", "zip binder")),
    ("office", "dry-erase-markers-low-odor-under-12", "Find low-odor dry erase markers under $12. Prepare a recommendation only.", 12.0, ("dry erase", "low odor"), ("permanent", "chalk markers")),
    ("cleaning", "unscented-liquid-he-detergent-under-25", "Find unscented liquid HE laundry detergent under $25. It must not be pods. Prepare a recommendation only.", 25.0, ("unscented", "liquid", "HE"), ("pods", "scented")),
    ("cleaning", "free-clear-dish-soap-under-10", "Find free-and-clear dish soap under $10. Prepare a recommendation only.", 10.0, ("dish soap", "free", "clear"), ("dishwasher detergent", "pods")),
    ("cleaning", "paper-towels-6-roll-under-20", "Find paper towels, at least 6 rolls, under $20. Prepare a recommendation only.", 20.0, ("paper towels", "6 rolls"), ("toilet paper", "napkins")),
    ("cleaning", "tall-kitchen-trash-bags-13-gallon-under-18", "Find tall kitchen trash bags under $18. Must fit 13 gallon cans. Prepare a recommendation only.", 18.0, ("13 gallon", "trash bags"), ("lawn bags", "contractor bags")),
    ("cleaning", "microfiber-cloths-12-pack-under-15", "Find microfiber cleaning cloths, pack of 12 or more, under $15. Prepare a recommendation only.", 15.0, ("microfiber", "12"), ("paper towels", "single cloth")),
    ("cleaning", "non-scratch-sponges-under-8", "Find non-scratch kitchen sponges under $8. Prepare a recommendation only.", 8.0, ("non-scratch", "sponges"), ("steel wool", "scrub brush only")),
    ("cleaning", "bleach-free-all-purpose-cleaner-under-8", "Find bleach-free all-purpose cleaner under $8. Prepare a recommendation only.", 8.0, ("all-purpose cleaner", "bleach-free"), ("bleach", "disinfecting wipes")),
    ("cleaning", "toilet-paper-12-roll-under-18", "Find toilet paper, at least 12 rolls, under $18. Prepare a recommendation only.", 18.0, ("toilet paper", "12 rolls"), ("paper towels", "single roll")),
    ("kitchen", "basket-coffee-filters-under-10", "Find basket coffee filters under $10. They should be compatible with basket-style coffee makers. Prepare a recommendation only.", 10.0, ("basket", "coffee filters"), ("cone", "permanent filter")),
    ("kitchen", "non-glass-food-storage-under-20", "Find non-glass food storage containers under $20. Prepare a recommendation only.", 20.0, ("food storage", "plastic"), ("glass",)),
    ("kitchen", "manual-can-opener-under-15", "Find a manual can opener under $15. Prepare a recommendation only.", 15.0, ("manual", "can opener"), ("electric", "bottle opener only")),
    ("kitchen", "measuring-cups-set-under-12", "Find a measuring cups set under $12. Prepare a recommendation only.", 12.0, ("measuring cups", "set"), ("measuring spoons only", "scale")),
    ("kitchen", "plastic-cutting-board-under-15", "Find a plastic cutting board under $15. Prepare a recommendation only.", 15.0, ("plastic", "cutting board"), ("wood", "glass")),
    ("kitchen", "dish-drying-rack-under-25", "Find a dish drying rack under $25. Prepare a recommendation only.", 25.0, ("dish drying rack",), ("dishwasher part", "over-sink over $25")),
    ("kitchen", "reusable-water-bottle-24oz-under-20", "Find a reusable water bottle around 24 oz under $20. Prepare a recommendation only.", 20.0, ("water bottle", "24 oz"), ("glass", "kids only")),
    ("kitchen", "silicone-spatula-set-under-15", "Find a silicone spatula set under $15. Prepare a recommendation only.", 15.0, ("silicone", "spatula set"), ("wood", "single spatula")),
    ("organization", "storage-bin-lid-under-20", "Find a storage bin with a lid under $20. Prepare a recommendation only.", 20.0, ("storage bin", "lid"), ("no lid",)),
    ("organization", "drawer-organizer-under-15", "Find a drawer organizer under $15. Prepare a recommendation only.", 15.0, ("drawer organizer",), ("desk organizer", "hanging organizer")),
    ("organization", "velvet-hangers-20-pack-under-20", "Find velvet hangers, pack of 20 or more, under $20. Prepare a recommendation only.", 20.0, ("velvet hangers", "20"), ("plastic hangers", "kids hangers")),
    ("organization", "shoe-rack-2-tier-under-25", "Find a 2-tier shoe rack under $25. Prepare a recommendation only.", 25.0, ("2-tier", "shoe rack"), ("over-door", "bench")),
    ("organization", "reusable-cable-ties-under-10", "Find reusable cable ties under $10. Prepare a recommendation only.", 10.0, ("reusable", "cable ties"), ("single-use", "zip ties only")),
    ("organization", "underbed-storage-bag-under-20", "Find an under-bed storage bag under $20. Prepare a recommendation only.", 20.0, ("under bed", "storage bag"), ("hard bin", "vacuum bag")),
    ("organization", "clear-stackable-bin-under-18", "Find a clear stackable storage bin under $18. Prepare a recommendation only.", 18.0, ("clear", "stackable", "storage bin"), ("opaque", "no lid")),
    ("personal-home", "bath-towels-2-pack-under-25", "Find bath towels, pack of 2, under $25. Prepare a recommendation only.", 25.0, ("bath towels", "2"), ("hand towels", "washcloths only")),
    ("personal-home", "shower-curtain-liner-under-12", "Find a shower curtain liner under $12. Prepare a recommendation only.", 12.0, ("shower curtain liner",), ("curtain only", "hooks only")),
    ("personal-home", "fragrance-free-hand-soap-under-8", "Find fragrance-free hand soap under $8. Prepare a recommendation only.", 8.0, ("fragrance-free", "hand soap"), ("scented", "sanitizer")),
    ("personal-home", "cotton-swabs-500-under-8", "Find cotton swabs, at least 500 count, under $8. Prepare a recommendation only.", 8.0, ("cotton swabs", "500"), ("cotton balls", "makeup pads")),
    ("personal-home", "travel-toothbrush-case-under-8", "Find a travel toothbrush case under $8. Prepare a recommendation only.", 8.0, ("toothbrush case", "travel"), ("toothbrush", "toothpaste")),
    ("personal-home", "unscented-body-wash-under-12", "Find unscented body wash under $12. Prepare a recommendation only.", 12.0, ("unscented", "body wash"), ("scented", "bar soap")),
    ("personal-home", "basic-white-socks-6-pack-under-15", "Find basic white socks, pack of 6 or more, under $15. Prepare a recommendation only.", 15.0, ("white socks", "6"), ("dress socks", "single pair")),
    ("pet", "dog-waste-bags-120-under-12", "Find dog waste bags, at least 120 count, under $12. Prepare a recommendation only.", 12.0, ("dog waste bags", "120"), ("cat litter", "diaper bags")),
    ("pet", "cat-litter-scoop-under-10", "Find a cat litter scoop under $10. Prepare a recommendation only.", 10.0, ("cat litter scoop",), ("litter box", "dog scoop")),
    ("pet", "pet-food-storage-container-under-25", "Find a pet food storage container under $25. Prepare a recommendation only.", 25.0, ("pet food", "storage container"), ("food bowl", "treats")),
    ("pet", "dog-leash-6ft-under-15", "Find a 6-foot dog leash under $15. Prepare a recommendation only.", 15.0, ("dog leash", "6 ft"), ("retractable", "cat harness")),
    ("pet", "cardboard-cat-scratcher-under-20", "Find a cardboard cat scratcher under $20. Prepare a recommendation only.", 20.0, ("cat scratcher", "cardboard"), ("cat tree", "toy only")),
    ("baby-family", "fragrance-free-baby-wipes-under-15", "Find fragrance-free baby wipes under $15. Prepare a recommendation only.", 15.0, ("fragrance-free", "baby wipes"), ("scented", "diapers")),
    ("baby-family", "kids-water-bottle-12oz-under-15", "Find a kids water bottle around 12 oz under $15. Prepare a recommendation only.", 15.0, ("kids", "water bottle", "12 oz"), ("glass", "adult tumbler")),
    ("baby-family", "school-glue-sticks-12-pack-under-10", "Find school glue sticks, pack of 12, under $10. Prepare a recommendation only.", 10.0, ("glue sticks", "12"), ("liquid glue", "hot glue")),
    ("baby-family", "washable-markers-under-8", "Find washable markers under $8. Prepare a recommendation only.", 8.0, ("washable", "markers"), ("permanent", "highlighters")),
    ("baby-family", "lunch-box-under-15", "Find a lunch box under $15. Prepare a recommendation only.", 15.0, ("lunch box",), ("bento accessories only", "backpack")),
    ("outdoor", "led-flashlight-under-15", "Find an LED flashlight under $15. Prepare a recommendation only.", 15.0, ("LED", "flashlight"), ("lantern", "headlamp")),
    ("outdoor", "garden-gloves-under-12", "Find garden gloves under $12. Prepare a recommendation only.", 12.0, ("garden gloves",), ("work gloves", "disposable gloves")),
    ("outdoor", "plant-saucer-10-inch-under-10", "Find a 10-inch plant saucer under $10. Prepare a recommendation only.", 10.0, ("10 inch", "plant saucer"), ("pot", "tray set over $10")),
    ("outdoor", "empty-spray-bottle-under-8", "Find an empty spray bottle under $8. Prepare a recommendation only.", 8.0, ("empty", "spray bottle"), ("cleaner included", "aerosol")),
    ("outdoor", "indoor-outdoor-doormat-under-20", "Find an indoor/outdoor doormat under $20. Prepare a recommendation only.", 20.0, ("doormat", "indoor", "outdoor"), ("bath mat", "rug pad")),
)

TASK_SEEDS: tuple[TaskSeed, ...] = tuple(TaskSeed(*row) for row in _TASK_ROWS)


def generate_task_specs(count: int, seed: int = 13) -> list[ShoppingTaskSpec]:
    """Generate stable, diverse safe-shopping task specs for rollout datasets."""
    if count < 1:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    seeds = list(TASK_SEEDS)
    rng.shuffle(seeds)

    specs: list[ShoppingTaskSpec] = []
    for index in range(count):
        task_seed = seeds[index % len(seeds)]
        cycle = index // len(seeds)
        suffix = f"-v{cycle + 1}" if cycle else ""
        specs.append(
            ShoppingTaskSpec(
                task_id=f"{task_seed.slug}{suffix}",
                instruction=_instruction_variant(task_seed.instruction, index, cycle),
                allowed_domains=_allowed_domains(index, task_seed.allowed_domains),
                max_price=task_seed.max_price + _budget_offset(index, cycle),
                must_have=list(task_seed.must_have),
                must_not_have=list(task_seed.must_not_have),
                token_budget=750,
                require_cart=False,
            )
        )
    return specs


def task_category(task_id: str) -> str | None:
    base = re.sub(r"-v\d+$", "", task_id)
    for seed in TASK_SEEDS:
        if seed.slug == base:
            return seed.category
    return None


def summarize_task_specs(specs: list[ShoppingTaskSpec]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for spec in specs:
        category = task_category(spec.task_id) or "unknown"
        summary[category] = summary.get(category, 0) + 1
    return dict(sorted(summary.items()))


def _allowed_domains(index: int, default: tuple[str, ...]) -> list[str]:
    mode = index % 6
    if mode == 1:
        return ["target.com"]
    if mode == 2:
        return ["amazon.com"]
    return list(default)


def _budget_offset(index: int, cycle: int) -> float:
    if cycle == 0:
        return 0.0
    offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
    return offsets[index % len(offsets)]


def _instruction_variant(instruction: str, index: int, cycle: int) -> str:
    if cycle == 0:
        return instruction
    lead_ins = (
        "For a budget-conscious household purchase, ",
        "For a quick restock order, ",
        "For a practical everyday replacement, ",
    )
    return f"{lead_ins[index % len(lead_ins)]}{instruction[0].lower()}{instruction[1:]}"
