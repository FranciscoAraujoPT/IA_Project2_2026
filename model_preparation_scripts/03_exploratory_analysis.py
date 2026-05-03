#!/usr/bin/env python3
"""
03_exploratory_analysis.py

Faz análise exploratória:
- taxa de conversão por variáveis principais
- tabelas CSV
- gráficos simples

Exemplo:
python 03_exploratory_analysis.py --contacts dataset_A_causality_likely_contacts_train_v3.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def response_bucket(minutes: int) -> str:
    if minutes < 60:
        return "<1h"
    if minutes < 240:
        return "1-4h"
    if minutes < 720:
        return "4-12h"
    if minutes <= 1440:
        return "12-24h"
    return ">24h"


def conversion_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    table = (
        df.groupby(column, dropna=False)
        .agg(contacts=("converted", "size"), conversions=("converted", "sum"), conversion_rate=("converted", "mean"))
        .reset_index()
    )
    table["conversion_rate"] = (table["conversion_rate"] * 100).round(1)
    return table.sort_values("conversion_rate", ascending=False)


def save_bar_chart(table: pd.DataFrame, x_col: str, title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(table[x_col].astype(str), table["conversion_rate"])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel("Conversion rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts", required=True, help="Path to contacts_train CSV")
    parser.add_argument("--output-dir", default="outputs/eda", help="Directory for EDA outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.contacts)
    df["contact_datetime"] = pd.to_datetime(df["contact_datetime"])
    df["response_time_bucket"] = df["response_time_min"].apply(response_bucket)

    print("\n=== 03. EXPLORATORY ANALYSIS ===\n")
    print(f"Rows: {len(df)}")
    print(f"Overall conversion rate: {df['converted'].mean() * 100:.1f}%")

    analysis_columns = [
        "response_time_bucket", "contact_source_type", "customer_type",
        "product_category", "product_subcategory", "product_collection",
        "is_payday_period", "is_peak_season",
    ]

    for col in analysis_columns:
        table = conversion_table(df, col)
        table_path = output_dir / f"conversion_by_{col}.csv"
        chart_path = output_dir / f"conversion_by_{col}.png"
        table.to_csv(table_path, index=False)
        save_bar_chart(table, col, f"Conversion rate by {col}", chart_path)
        print(f"\nConversion by {col}:")
        print(table.to_string(index=False))
        print(f"Saved: {table_path}")
        print(f"Saved: {chart_path}")

    response_stats = df.groupby("converted")["response_time_min"].describe().round(1).reset_index()
    response_stats_path = output_dir / "response_time_stats_by_converted.csv"
    response_stats.to_csv(response_stats_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.hist(df["response_time_min"], bins=30)
    plt.title("Distribution of response_time_min")
    plt.xlabel("response_time_min")
    plt.ylabel("Number of contacts")
    plt.tight_layout()
    hist_path = output_dir / "response_time_distribution.png"
    plt.savefig(hist_path)
    plt.close()

    print(f"\nResponse-time stats saved: {response_stats_path}")
    print(f"Response-time histogram saved: {hist_path}")
    print("\n✅ EDA completed.")


if __name__ == "__main__":
    main()
