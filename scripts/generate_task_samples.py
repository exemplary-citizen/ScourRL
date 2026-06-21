from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from cart_scout.schema import ShoppingTaskSpec


DEFAULT_TRAIN_OUT = Path("data/shopping_train_1000.jsonl")
DEFAULT_EVAL_OUT = Path("data/shopping_eval_200.jsonl")
DEFAULT_SEED = 20260621

DOMAIN_SETS = [
    ["target.com", "amazon.com"],
    ["amazon.com", "target.com"],
    ["walmart.com", "target.com"],
    ["amazon.com", "staples.com"],
    ["target.com", "bestbuy.com"],
    ["amazon.com", "homedepot.com"],
]

CONTEXTS = [
    "for everyday household use",
    "for a budget restock",
    "for a small apartment",
    "for a dorm room",
    "for a shared living space",
    "for a basic starter kit",
    "for a compact setup",
    "for regular weekly use",
]

PRODUCTS: list[dict[str, Any]] = [
    {
        "slug": "usb_c_charger",
        "product": "a USB-C wall charger",
        "must_have": [["USB-C", "Power Delivery", "30W"], ["USB-C", "PD", "45W"], ["USB-C", "GaN", "30W"]],
        "must_not": ["Lightning", "MagSafe only"],
        "prices": [25, 30, 35, 40],
    },
    {
        "slug": "usb_c_cable",
        "product": "a USB-C charging cable",
        "must_have": [["USB-C", "6 ft"], ["USB-C", "10 ft"], ["USB-C", "braided"], ["USB-C", "60W"]],
        "must_not": ["Lightning", "Micro USB"],
        "prices": [10, 12, 15, 18],
    },
    {
        "slug": "aaa_batteries",
        "product": "AAA batteries",
        "must_have": [["AAA", "20 pack"], ["AAA", "24 pack"], ["AAA", "alkaline"], ["AAA", "rechargeable"]],
        "must_not": ["AA", "charger bundle"],
        "prices": [15, 20, 25, 30],
    },
    {
        "slug": "aa_batteries",
        "product": "AA batteries",
        "must_have": [["AA", "20 pack"], ["AA", "24 pack"], ["AA", "alkaline"], ["AA", "rechargeable"]],
        "must_not": ["AAA", "charger bundle"],
        "prices": [15, 20, 25, 30],
    },
    {
        "slug": "printer_paper",
        "product": "letter-size printer paper",
        "must_have": [["letter", "500 sheets"], ["8.5 x 11", "500 sheets"], ["copy paper", "letter"], ["printer paper", "500 sheets"]],
        "must_not": ["photo paper", "legal size"],
        "prices": [10, 12, 15, 18],
    },
    {
        "slug": "notebook",
        "product": "a notebook",
        "must_have": [["college ruled", "notebook"], ["spiral", "college ruled"], ["wide ruled", "notebook"], ["composition", "notebook"]],
        "must_not": ["planner", "sketchbook"],
        "prices": [5, 8, 10, 12],
    },
    {
        "slug": "pens",
        "product": "black ink pens",
        "must_have": [["black", "gel"], ["black", "ballpoint"], ["0.7 mm", "black"], ["retractable", "black"]],
        "must_not": ["pencil", "marker"],
        "prices": [6, 8, 10, 12],
    },
    {
        "slug": "sticky_notes",
        "product": "sticky notes",
        "must_have": [["sticky notes", "3 x 3"], ["Post-it", "3 x 3"], ["lined", "sticky notes"], ["recycled", "sticky notes"]],
        "must_not": ["index cards", "notebook"],
        "prices": [5, 8, 10, 12],
    },
    {
        "slug": "detergent",
        "product": "liquid HE laundry detergent",
        "must_have": [["unscented", "liquid", "HE"], ["free clear", "liquid", "HE"], ["sensitive skin", "liquid", "HE"], ["concentrated", "liquid", "HE"]],
        "must_not": ["pods", "scented"],
        "prices": [15, 20, 25, 30],
    },
    {
        "slug": "dish_soap",
        "product": "dish soap",
        "must_have": [["free", "clear", "dish soap"], ["unscented", "dish soap"], ["plant based", "dish soap"], ["dish soap", "refill"]],
        "must_not": ["dishwasher detergent", "pods"],
        "prices": [6, 8, 10, 12],
    },
    {
        "slug": "paper_towels",
        "product": "paper towels",
        "must_have": [["paper towels", "6 rolls"], ["paper towels", "select-a-size"], ["paper towels", "recycled"], ["paper towels", "mega rolls"]],
        "must_not": ["toilet paper", "napkins"],
        "prices": [12, 15, 20, 25],
    },
    {
        "slug": "trash_bags",
        "product": "kitchen trash bags",
        "must_have": [["13 gallon", "drawstring"], ["kitchen", "13 gallon"], ["tall kitchen", "drawstring"], ["13 gallon", "unscented"]],
        "must_not": ["lawn bags", "scented"],
        "prices": [10, 15, 20, 25],
    },
    {
        "slug": "coffee_filters",
        "product": "coffee filters",
        "must_have": [["basket", "coffee filters"], ["cone", "coffee filters"], ["number 4", "cone"], ["unbleached", "coffee filters"]],
        "must_not": ["permanent filter", "coffee pods"],
        "prices": [5, 8, 10, 12],
    },
    {
        "slug": "food_storage",
        "product": "food storage containers",
        "must_have": [["plastic", "food storage"], ["BPA-free", "food storage"], ["lids", "food storage"], ["stackable", "food storage"]],
        "must_not": ["glass", "single-use bags"],
        "prices": [12, 15, 20, 25],
    },
    {
        "slug": "storage_bin",
        "product": "a storage bin",
        "must_have": [["storage bin", "lid"], ["clear", "storage bin"], ["under bed", "storage bin"], ["stackable", "storage bin"]],
        "must_not": ["no lid", "fabric basket"],
        "prices": [12, 15, 20, 25],
    },
    {
        "slug": "shower_liner",
        "product": "a shower curtain liner",
        "must_have": [["shower liner", "PEVA"], ["clear", "shower liner"], ["mildew resistant", "liner"], ["weighted", "shower liner"]],
        "must_not": ["fabric curtain", "hooks only"],
        "prices": [8, 10, 12, 15],
    },
    {
        "slug": "light_bulbs",
        "product": "LED light bulbs",
        "must_have": [["LED", "A19", "soft white"], ["LED", "daylight", "A19"], ["LED", "60W equivalent"], ["LED", "dimmable"]],
        "must_not": ["incandescent", "smart bulb"],
        "prices": [10, 12, 15, 20],
    },
    {
        "slug": "extension_cord",
        "product": "an indoor extension cord",
        "must_have": [["indoor", "6 ft"], ["indoor", "10 ft"], ["3 outlet", "extension cord"], ["grounded", "extension cord"]],
        "must_not": ["outdoor only", "power strip"],
        "prices": [8, 10, 15, 20],
    },
    {
        "slug": "mouse_pad",
        "product": "a mouse pad",
        "must_have": [["non-slip", "mouse pad"], ["large", "mouse pad"], ["wrist rest", "mouse pad"], ["black", "mouse pad"]],
        "must_not": ["desk mat", "gaming RGB"],
        "prices": [8, 10, 12, 15],
    },
    {
        "slug": "water_bottle",
        "product": "a reusable water bottle",
        "must_have": [["BPA-free", "water bottle"], ["stainless steel", "water bottle"], ["24 oz", "water bottle"], ["leakproof", "water bottle"]],
        "must_not": ["single-use", "filter bottle"],
        "prices": [12, 15, 20, 25],
    },
    {
        "slug": "lunch_bag",
        "product": "an insulated lunch bag",
        "must_have": [["insulated", "lunch bag"], ["reusable", "lunch bag"], ["leak resistant", "lunch bag"], ["adult", "lunch bag"]],
        "must_not": ["kids character", "hard cooler"],
        "prices": [12, 15, 20, 25],
    },
    {
        "slug": "measuring_cups",
        "product": "measuring cups",
        "must_have": [["measuring cups", "set"], ["stainless steel", "measuring cups"], ["plastic", "measuring cups"], ["dishwasher safe", "measuring cups"]],
        "must_not": ["measuring spoons only", "glass"],
        "prices": [8, 10, 12, 15],
    },
    {
        "slug": "cutting_board",
        "product": "a cutting board",
        "must_have": [["plastic", "cutting board"], ["dishwasher safe", "cutting board"], ["non-slip", "cutting board"], ["BPA-free", "cutting board"]],
        "must_not": ["wood", "glass"],
        "prices": [10, 12, 15, 20],
    },
    {
        "slug": "zip_bags",
        "product": "zip-top storage bags",
        "must_have": [["quart", "zip"], ["gallon", "zip"], ["freezer", "zip"], ["slider", "storage bags"]],
        "must_not": ["trash bags", "paper bags"],
        "prices": [5, 8, 10, 12],
    },
    {
        "slug": "microfiber_cloths",
        "product": "microfiber cleaning cloths",
        "must_have": [["microfiber", "cleaning cloths"], ["washable", "microfiber"], ["12 pack", "microfiber"], ["lint free", "microfiber"]],
        "must_not": ["paper towels", "wet wipes"],
        "prices": [8, 10, 12, 15],
    },
]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def format_series(values: list[str], conjunction: str = "and") -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} {conjunction} {values[1]}"
    return ", ".join(values[:-1]) + f", {conjunction} {values[-1]}"


def build_instruction(product: dict[str, Any], must_have: list[str], must_not: list[str], price: int, context: str) -> str:
    must_have_text = format_series(must_have, "and")
    must_not_text = format_series(must_not, "or")
    return (
        f"Find {product['product']} {context} under ${price}. "
        f"Must have {must_have_text}. "
        f"Must not be or include {must_not_text}. "
        "Prepare recommendation only."
    )


def candidate_rows(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in PRODUCTS:
        for must_have in product["must_have"]:
            for price in product["prices"]:
                for context in CONTEXTS:
                    for allowed_domains in DOMAIN_SETS:
                        instruction = build_instruction(product, must_have, product["must_not"], price, context)
                        rows.append(
                            {
                                "task_id": "",
                                "instruction": instruction,
                                "allowed_domains": allowed_domains,
                                "max_price": price,
                                "must_have": must_have,
                                "must_not_have": product["must_not"],
                                "forbidden_actions": ["checkout", "payment", "login"],
                                "token_budget": 750,
                            }
                        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def generate_rows(train_count: int, eval_count: int, seed: int = DEFAULT_SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total = train_count + eval_count
    seen_instructions: set[str] = set()
    unique: list[dict[str, Any]] = []

    for row in candidate_rows(seed):
        if row["instruction"] in seen_instructions:
            continue
        seen_instructions.add(row["instruction"])
        unique.append(row)
        if len(unique) >= total:
            break

    if len(unique) < total:
        raise ValueError(f"only generated {len(unique)} unique rows, need {total}")

    train_rows = _assign_ids(unique[:train_count], "train")
    eval_rows = _assign_ids(unique[train_count:], "eval")
    _validate_distinct(train_rows, eval_rows)
    return train_rows, eval_rows


def _assign_ids(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        product_slug = slugify(row["instruction"].split(" under $", 1)[0].removeprefix("Find "))
        row = dict(row)
        row["task_id"] = f"{split}_{index:04d}_{product_slug[:48]}"
        ShoppingTaskSpec.model_validate(row)
        assigned.append(row)
    return assigned


def _validate_distinct(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    train_ids = {row["task_id"] for row in train_rows}
    eval_ids = {row["task_id"] for row in eval_rows}
    train_instructions = {row["instruction"] for row in train_rows}
    eval_instructions = {row["instruction"] for row in eval_rows}
    if train_ids & eval_ids:
        raise ValueError("train/eval task_id overlap detected")
    if train_instructions & eval_instructions:
        raise ValueError("train/eval instruction overlap detected")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic CartScout shopping task samples.")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--eval-count", type=int, default=200)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows, eval_rows = generate_rows(args.train_count, args.eval_count, seed=args.seed)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    print(
        json.dumps(
            {
                "train_out": str(args.train_out),
                "train_count": len(train_rows),
                "eval_out": str(args.eval_out),
                "eval_count": len(eval_rows),
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
