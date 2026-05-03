#!/usr/bin/env python3
"""
02_confirm_derived_variables.py

Confirma se as variáveis derivadas estão coerentes:
- response_time_min
- is_payday_period
- is_peak_season
- converted a partir da tabela Purchase

Nota importante:
Nos datasets sintéticos, a tabela purchases tem a coluna related_contact_id.
Por isso, para validar converted, usamos primeiro related_contact_id.
Isto evita falsos mismatches quando o mesmo customer_id aparece em vários contactos.

Exemplo:
python3 02_confirm_derived_variables.py \
  --contacts dataset_A_causality_likely_contacts_train.csv \
  --purchases dataset_A_causality_likely_purchases.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def expected_payday(dt: pd.Timestamp) -> int:
    return int(dt.day >= 25 or dt.day <= 2)


def expected_peak_season(dt: pd.Timestamp) -> int:
    return int(dt.month in {9, 11, 12})


def valid_purchase_for_contact(contact: pd.Series, purchases: pd.DataFrame) -> bool:
    """
    Checks whether a contact has at least one valid purchase.

    For these synthetic datasets:
    - If purchases has related_contact_id, that is the safest validation key.
    - Otherwise, falls back to customer_id + 3-day window + product/category rule.
    """

    # Preferred method for the synthetic datasets.
    if "related_contact_id" in purchases.columns:
        related = purchases[purchases["related_contact_id"] == contact["contact_id"]]
        return not related.empty

    # Fallback method for a real database without related_contact_id.
    customer_purchases = purchases[purchases["customer_id"] == contact["customer_id"]].copy()
    if customer_purchases.empty:
        return False

    start = contact["contact_datetime"]
    end = start + pd.Timedelta(days=3)

    customer_purchases = customer_purchases[
        (customer_purchases["purchase_datetime"] >= start)
        & (customer_purchases["purchase_datetime"] <= end)
    ]

    if customer_purchases.empty:
        return False

    if contact["contact_source_type"] == "general_contact":
        return True

    if contact["contact_source_type"] == "product_page":
        same_product = customer_purchases["product_id"].eq(contact["product_id"])
        same_category = customer_purchases["product_category"].eq(contact["product_category"])
        return bool((same_product | same_category).any())

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts", required=True, help="Path to contacts_train CSV")
    parser.add_argument("--purchases", required=True, help="Path to purchases CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory for validation outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contacts = pd.read_csv(args.contacts)
    purchases = pd.read_csv(args.purchases)

    contacts["contact_datetime"] = pd.to_datetime(contacts["contact_datetime"])
    contacts["first_response_datetime"] = pd.to_datetime(contacts["first_response_datetime"])

    if "purchase_datetime" in purchases.columns:
        purchases["purchase_datetime"] = pd.to_datetime(purchases["purchase_datetime"])

    print("\n=== 02. DERIVED VARIABLES CHECK ===\n")

    checks = {}

    expected_response_time = (
        (contacts["first_response_datetime"] - contacts["contact_datetime"])
        .dt.total_seconds()
        .div(60)
        .round()
        .astype(int)
    )
    contacts["expected_response_time_min"] = expected_response_time
    contacts["response_time_match"] = contacts["response_time_min"].astype(int).eq(expected_response_time)
    checks["response_time_min"] = contacts["response_time_match"].all()

    contacts["expected_is_payday_period"] = contacts["contact_datetime"].apply(expected_payday)
    contacts["is_payday_match"] = contacts["is_payday_period"].astype(int).eq(contacts["expected_is_payday_period"])
    checks["is_payday_period"] = contacts["is_payday_match"].all()

    contacts["expected_is_peak_season"] = contacts["contact_datetime"].apply(expected_peak_season)
    contacts["is_peak_match"] = contacts["is_peak_season"].astype(int).eq(contacts["expected_is_peak_season"])
    checks["is_peak_season"] = contacts["is_peak_match"].all()

    contacts["expected_converted"] = contacts.apply(
        lambda row: int(valid_purchase_for_contact(row, purchases)),
        axis=1,
    )
    contacts["converted_match"] = contacts["converted"].astype(int).eq(contacts["expected_converted"])
    checks["converted"] = contacts["converted_match"].all()

    for name, ok in checks.items():
        print(("✅" if ok else "❌") + f" {name}")

    mismatches = contacts[
        ~contacts["response_time_match"]
        | ~contacts["is_payday_match"]
        | ~contacts["is_peak_match"]
        | ~contacts["converted_match"]
    ].copy()

    mismatch_path = output_dir / "02_derived_variable_mismatches.csv"
    mismatches.to_csv(mismatch_path, index=False)

    summary_path = output_dir / "02_derived_variables_report.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("02. DERIVED VARIABLES REPORT\n\n")
        for name, ok in checks.items():
            f.write(f"{name}: {'OK' if ok else 'FAILED'}\n")
        f.write(f"\nNumber of mismatched rows: {len(mismatches)}\n")
        f.write(f"Mismatches file: {mismatch_path}\n")

    print(f"\nMismatches saved to: {mismatch_path}")
    print(f"Report saved to: {summary_path}")

    if len(mismatches) > 0:
        raise SystemExit("\nDerived variable check failed. Inspect mismatches before continuing.")

    print("\n✅ All derived variables are coherent.")


if __name__ == "__main__":
    main()
