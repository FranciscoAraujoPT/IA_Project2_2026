#!/usr/bin/env python3
"""
01_validate_data.py

Valida a estrutura básica do dataset de contactos.

Exemplo:
python 01_validate_data.py --contacts dataset_A_causality_likely_contacts_train_v3.csv
"""

import argparse
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "contact_id", "customer_id", "contact_datetime", "first_response_datetime",
    "response_time_min", "converted", "contact_source_type", "customer_type",
    "product_id", "product_category", "product_subcategory", "product_collection",
    "is_payday_period", "is_peak_season",
]

ALLOWED_CUSTOMER_TYPES = {"lead", "one_time", "two_times", "vip"}
ALLOWED_CONTACT_SOURCE_TYPES = {"product_page", "general_contact"}
ALLOWED_CONVERTED = {0, 1}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if condition:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts", required=True, help="Path to contacts_train CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory for validation outputs")
    args = parser.parse_args()

    contacts_path = Path(args.contacts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(contacts_path)
    errors = []

    print("\n=== 01. DATA VALIDATION ===\n")
    print(f"File: {contacts_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}\n")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    check(len(missing_cols) == 0, "All expected columns are present", errors)
    if missing_cols:
        print("Missing columns:", missing_cols)

    if extra_cols:
        print("⚠️ Extra columns found:", extra_cols)
    else:
        print("✅ No extra columns found")

    check(df["contact_id"].is_unique, "contact_id is unique", errors)

    unexpected_nulls = df[EXPECTED_COLUMNS].isna().sum()
    unexpected_nulls = unexpected_nulls[unexpected_nulls > 0]
    check(len(unexpected_nulls) == 0, "No unexpected null values", errors)
    if len(unexpected_nulls) > 0:
        print(unexpected_nulls)

    check((df["response_time_min"] >= 0).all(), "response_time_min >= 0", errors)

    df["contact_datetime"] = pd.to_datetime(df["contact_datetime"], errors="coerce")
    df["first_response_datetime"] = pd.to_datetime(df["first_response_datetime"], errors="coerce")

    check(df["contact_datetime"].notna().all(), "contact_datetime parses as datetime", errors)
    check(df["first_response_datetime"].notna().all(), "first_response_datetime parses as datetime", errors)
    check((df["first_response_datetime"] >= df["contact_datetime"]).all(), "first_response_datetime >= contact_datetime", errors)

    converted_values = set(df["converted"].dropna().astype(int).unique())
    check(converted_values.issubset(ALLOWED_CONVERTED), "converted only contains 0/1", errors)
    print("converted values:", sorted(converted_values))

    customer_values = set(df["customer_type"].dropna().unique())
    check(customer_values.issubset(ALLOWED_CUSTOMER_TYPES), "customer_type has only allowed values", errors)
    print("customer_type values:", sorted(customer_values))

    source_values = set(df["contact_source_type"].dropna().unique())
    check(source_values.issubset(ALLOWED_CONTACT_SOURCE_TYPES), "contact_source_type has only allowed values", errors)
    print("contact_source_type values:", sorted(source_values))

    validation_report = output_dir / "01_validation_report.txt"
    with validation_report.open("w", encoding="utf-8") as f:
        f.write("01. DATA VALIDATION REPORT\n\n")
        f.write(f"File: {contacts_path}\nRows: {len(df)}\nColumns: {len(df.columns)}\n\n")
        f.write(f"Missing columns: {missing_cols}\nExtra columns: {extra_cols}\n")
        f.write(f"Unexpected nulls:\n{unexpected_nulls.to_string() if len(unexpected_nulls) else 'None'}\n\n")
        f.write(f"Errors: {errors if errors else 'None'}\n")

    print(f"\nReport saved to: {validation_report}")
    if errors:
        raise SystemExit("\nValidation failed. Fix errors before continuing.")
    print("\n✅ Validation passed.")


if __name__ == "__main__":
    main()
