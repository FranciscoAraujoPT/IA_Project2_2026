#!/usr/bin/env python3
"""
00_generate_large_datasets.py

Generates larger synthetic versions of the three CaseOver datasets.

The three datasets intentionally represent different causal stories:
- A: response time is likely causal for conversion.
- B: inconclusive relationship between response time and conversion.
- C: response time is correlated with conversion, but seasonality/payday drive
  most of the signal.

Example:
python3 00_generate_large_datasets.py --rows 5000 --output-dir datasets_A_B_C_large
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DATASETS = {
    "A": {
        "slug": "dataset_A_causality_likely",
        "label": "causality likely",
    },
    "B": {
        "slug": "dataset_B_inconclusive",
        "label": "inconclusive",
    },
    "C": {
        "slug": "dataset_C_correlation_not_causality",
        "label": "correlation not causality",
    },
}

CUSTOMER_TYPES = ["lead", "one_time", "two_times", "vip"]
CUSTOMER_WEIGHTS = [0.36, 0.24, 0.18, 0.22]
CONTACT_SOURCE_TYPES = ["product_page", "general_contact"]
CONTACT_SOURCE_WEIGHTS = [0.64, 0.36]

START_DATE = datetime(2025, 1, 1, 8, 0)
END_DATE = datetime(2025, 12, 28, 22, 0)


@dataclass(frozen=True)
class Product:
    product_id: str
    product_category: str
    product_subcategory: str
    product_collection: str


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_choice(rng: random.Random, values: list, weights: list[float]):
    return rng.choices(values, weights=weights, k=1)[0]


def expected_payday(dt: datetime) -> int:
    return int(dt.day >= 25 or dt.day <= 2)


def expected_peak_season(dt: datetime) -> int:
    return int(dt.month in {9, 11, 12})


def load_product_catalog(reference_dir: Path) -> tuple[list[Product], list[float]]:
    """Builds a product catalog and weights from the existing small datasets."""

    frames = []
    for contacts_path in sorted(reference_dir.glob("dataset_*_contacts_train.csv")):
        frames.append(pd.read_csv(contacts_path))

    if not frames:
        raise FileNotFoundError(f"No reference contacts CSV files found in {reference_dir}")

    contacts = pd.concat(frames, ignore_index=True)
    product_cols = [
        "product_id",
        "product_category",
        "product_subcategory",
        "product_collection",
    ]

    product_rows = (
        contacts[contacts["product_id"] != "unknown"][product_cols]
        .drop_duplicates()
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    catalog = [Product(**row.to_dict()) for _, row in product_rows.iterrows()]

    product_counts = contacts[contacts["product_id"] != "unknown"]["product_id"].value_counts()
    weights = [float(product_counts.get(product.product_id, 1)) for product in catalog]
    return catalog, weights


def random_contact_datetime(rng: random.Random, scenario: str) -> datetime:
    """Samples dates with scenario-specific seasonality."""

    peak_month_weight = 1.0
    if scenario == "C":
        peak_month_weight = 1.8

    months = list(range(1, 13))
    month_weights = [peak_month_weight if month in {9, 11, 12} else 1.0 for month in months]
    month = weighted_choice(rng, months, month_weights)

    # Keep the day safely inside the month and before year-end purchase windows.
    days_in_month = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 28,
    }
    day = rng.randint(1, days_in_month[month])
    hour = rng.choices(
        list(range(8, 23)),
        weights=[0.6, 0.7, 0.9, 1.0, 1.1, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.1, 1.0, 0.8, 0.7],
        k=1,
    )[0]
    minute = rng.randint(0, 59)
    dt = datetime(2025, month, day, hour, minute)
    return max(START_DATE, min(END_DATE, dt))


def sample_response_time(rng: random.Random, scenario: str, contact_dt: datetime) -> int:
    """Samples response time in minutes according to each dataset story."""

    peak = expected_peak_season(contact_dt)
    payday = expected_payday(contact_dt)

    if scenario == "A":
        bucket = rng.choices(
            ["very_fast", "fast", "medium", "slow", "very_slow"],
            weights=[0.22, 0.25, 0.22, 0.14, 0.17],
            k=1,
        )[0]
    elif scenario == "B":
        bucket = rng.choices(
            ["very_fast", "fast", "medium", "slow", "very_slow"],
            weights=[0.13, 0.17, 0.24, 0.16, 0.30],
            k=1,
        )[0]
    else:
        # In C, seasonality creates operational correlation: peak/payday contacts
        # are handled faster, but conversion itself is mostly season-driven.
        if peak or payday:
            weights = [0.20, 0.24, 0.24, 0.14, 0.18]
        else:
            weights = [0.08, 0.12, 0.14, 0.12, 0.54]
        bucket = rng.choices(
            ["very_fast", "fast", "medium", "slow", "very_slow"],
            weights=weights,
            k=1,
        )[0]

    ranges = {
        "very_fast": (5, 59),
        "fast": (60, 239),
        "medium": (240, 719),
        "slow": (720, 1440),
        "very_slow": (1441, 5760),
    }
    low, high = ranges[bucket]
    return rng.randint(low, high)


def conversion_probability(
    scenario: str,
    response_time_min: int,
    contact_dt: datetime,
    customer_type: str,
    contact_source_type: str,
    product: Product,
) -> float:
    fast_hours = response_time_min / 60
    peak = expected_peak_season(contact_dt)
    payday = expected_payday(contact_dt)

    customer_effect = {
        "lead": -0.16,
        "one_time": -0.10,
        "two_times": 0.22,
        "vip": 0.28,
    }[customer_type]
    source_effect = 0.12 if contact_source_type == "product_page" else -0.08
    category_effect = {
        "cameras_and_accessories": 0.20,
        "kits_and_bundles": 0.16,
        "retro_consoles": 0.05,
        "car_accessories": -0.03,
        "earphones": -0.05,
        "other_accessories": -0.04,
        "unknown": -0.08,
    }.get(product.product_category, 0.0)

    if scenario == "A":
        # Strong direct response-time effect.
        response_effect = 1.35 if response_time_min < 60 else 0.85 if response_time_min < 240 else 0.15
        response_effect += -0.55 if response_time_min > 1440 else 0.0
        response_effect += -0.20 * max(0, fast_hours - 24) / 24
        logit = -0.55 + response_effect + customer_effect + source_effect + category_effect
        logit += 0.10 * payday + 0.12 * peak
        return clamp(sigmoid(logit), 0.04, 0.93)

    if scenario == "B":
        # Mostly noise. Response time has only a tiny effect.
        logit = -0.74 + 0.16 * source_effect + 0.20 * customer_effect + 0.12 * category_effect
        logit += 0.05 * peak + 0.02 * payday
        logit += -0.03 if response_time_min > 1440 else 0.02
        return clamp(sigmoid(logit), 0.18, 0.48)

    # C: conversion is mostly season/payday/product. Response time is not a
    # direct driver here, even though it will correlate with conversion.
    logit = -1.05 + 1.35 * peak + 0.45 * payday
    logit += 0.18 * source_effect + 0.10 * customer_effect + 0.45 * category_effect
    return clamp(sigmoid(logit), 0.08, 0.88)


def maybe_unknown_product(
    rng: random.Random,
    contact_source_type: str,
    product: Product,
) -> Product:
    if contact_source_type == "general_contact" and rng.random() < 0.48:
        return Product("unknown", "unknown", "unknown", "unknown")
    return product


def purchase_value(rng: random.Random, category: str) -> float:
    ranges = {
        "retro_consoles": (95, 180),
        "cameras_and_accessories": (45, 115),
        "kits_and_bundles": (70, 155),
        "car_accessories": (18, 55),
        "earphones": (18, 65),
        "other_accessories": (8, 35),
        "unknown": (15, 95),
    }
    low, high = ranges.get(category, (15, 100))
    return round(rng.uniform(low, high), 2)


def generate_dataset(
    scenario: str,
    rows: int,
    rng: random.Random,
    catalog: list[Product],
    product_weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contacts = []
    purchases = []
    customer_pool_size = max(50, int(rows * 0.55))
    customers = [f"{scenario}_U{i:04d}" for i in range(1, customer_pool_size + 1)]

    for index in range(1, rows + 1):
        contact_id = f"{scenario}_C{index:05d}"
        customer_id = rng.choice(customers)
        contact_dt = random_contact_datetime(rng, scenario)
        response_time_min = sample_response_time(rng, scenario, contact_dt)
        first_response_dt = contact_dt + timedelta(minutes=response_time_min)

        contact_source_type = weighted_choice(rng, CONTACT_SOURCE_TYPES, CONTACT_SOURCE_WEIGHTS)
        customer_type = weighted_choice(rng, CUSTOMER_TYPES, CUSTOMER_WEIGHTS)
        product = weighted_choice(rng, catalog, product_weights)
        product = maybe_unknown_product(rng, contact_source_type, product)

        prob = conversion_probability(
            scenario,
            response_time_min,
            contact_dt,
            customer_type,
            contact_source_type,
            product,
        )
        converted = int(rng.random() < prob)

        contacts.append(
            {
                "contact_id": contact_id,
                "customer_id": customer_id,
                "contact_datetime": contact_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "first_response_datetime": first_response_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "response_time_min": response_time_min,
                "converted": converted,
                "contact_source_type": contact_source_type,
                "customer_type": customer_type,
                "product_id": product.product_id,
                "product_category": product.product_category,
                "product_subcategory": product.product_subcategory,
                "product_collection": product.product_collection,
                "is_payday_period": expected_payday(contact_dt),
                "is_peak_season": expected_peak_season(contact_dt),
            }
        )

        if converted:
            purchase_id = f"{scenario}_P{len(purchases) + 1:05d}"
            purchase_dt = contact_dt + timedelta(minutes=rng.randint(60, 72 * 60))
            purchases.append(
                {
                    "purchase_id": purchase_id,
                    "customer_id": customer_id,
                    "purchase_datetime": purchase_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "product_id": product.product_id,
                    "product_category": product.product_category,
                    "product_subcategory": product.product_subcategory,
                    "product_collection": product.product_collection,
                    "purchase_value": purchase_value(rng, product.product_category),
                    "related_contact_id": contact_id,
                }
            )

    return pd.DataFrame(contacts), pd.DataFrame(purchases)


def print_summary(scenario: str, contacts: pd.DataFrame, purchases: pd.DataFrame) -> None:
    print(f"\nDataset {scenario} - {DATASETS[scenario]['label']}")
    print(f"- contacts: {len(contacts)}")
    print(f"- purchases: {len(purchases)}")
    print(f"- conversion rate: {contacts['converted'].mean() * 100:.1f}%")
    response_stats = contacts.groupby("converted")["response_time_min"].agg(["count", "mean", "median"])
    print("- response time by converted:")
    print(response_stats.round(1).to_string())
    print("- conversion by peak season:")
    print((contacts.groupby("is_peak_season")["converted"].mean() * 100).round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000, help="Number of contact rows per dataset")
    parser.add_argument(
        "--output-dir",
        default="datasets_A_B_C_large",
        help="Directory where generated CSV files will be saved",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Directory containing the existing small dataset CSV files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.rows < 100:
        raise ValueError("--rows should be at least 100 to keep class distributions stable")

    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir) if args.reference_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog, product_weights = load_product_catalog(reference_dir)
    rng = random.Random(args.seed)

    print("\n=== 00. GENERATE LARGE SYNTHETIC DATASETS ===")
    print(f"\nRows per dataset: {args.rows}")
    print(f"Reference directory: {reference_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Product catalog size: {len(catalog)}")

    for scenario, config in DATASETS.items():
        contacts, purchases = generate_dataset(
            scenario=scenario,
            rows=args.rows,
            rng=rng,
            catalog=catalog,
            product_weights=product_weights,
        )

        contacts_path = output_dir / f"{config['slug']}_contacts_train.csv"
        purchases_path = output_dir / f"{config['slug']}_purchases.csv"

        contacts.to_csv(contacts_path, index=False)
        purchases.to_csv(purchases_path, index=False)

        print_summary(scenario, contacts, purchases)
        print(f"- saved contacts: {contacts_path}")
        print(f"- saved purchases: {purchases_path}")

    print("\nLarge dataset generation completed.")


if __name__ == "__main__":
    main()
