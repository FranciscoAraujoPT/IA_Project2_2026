#!/usr/bin/env python3
"""
04_create_features.py

Cria variáveis úteis a partir de contact_datetime:
- month
- day_of_month
- day_of_week
- hour_of_day
- fast_response_24h

Exemplo:
python 04_create_features.py --contacts dataset_A_causality_likely_contacts_train_v3.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts", required=True, help="Path to contacts_train CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory for feature outputs")
    parser.add_argument("--output-file", default=None, help="Optional output CSV filename")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.contacts)
    df["contact_datetime"] = pd.to_datetime(df["contact_datetime"])

    df["month"] = df["contact_datetime"].dt.month
    df["day_of_month"] = df["contact_datetime"].dt.day
    df["day_of_week"] = df["contact_datetime"].dt.dayofweek  # Monday=0, Sunday=6
    df["hour_of_day"] = df["contact_datetime"].dt.hour
    df["fast_response_24h"] = (df["response_time_min"] <= 1440).astype(int)

    output_path = output_dir / (args.output_file if args.output_file else f"{Path(args.contacts).stem}_with_features.csv")
    df.to_csv(output_path, index=False)

    print("\n=== 04. FEATURE CREATION ===\n")
    print("Created variables:")
    print("- month")
    print("- day_of_month")
    print("- day_of_week")
    print("- hour_of_day")
    print("- fast_response_24h")
    print(f"\nOutput saved to: {output_path}")
    print("\nFeature creation completed.")


if __name__ == "__main__":
    main()
