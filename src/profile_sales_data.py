"""
Profile sales transaction history and data coverage.

Purpose:
    Establish a baseline understanding of the raw sales transaction dataset
    before data cleaning and weekly demand aggregation.

Outputs:
    reports/sales_data_profiling/
        sales_data_profile.json
        sku_transaction_coverage.csv
        transaction_quantity_distribution.csv

Usage:
    python src/profile_sales_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "sales_transactions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "sales_data_profiling"

DATE_COL = "transaction_date"
SKU_COL = "sku"
QUANTITY_COL = "quantity"

TOP_N_SKUS = 10


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_sales_data(file_path: Path) -> pd.DataFrame:
    """Load sales transactions and validate required columns."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sales dataset not found: {file_path}\n"
            "Check the input path and dataset location."
        )

    df = pd.read_csv(file_path)

    required_columns = {DATE_COL, SKU_COL, QUANTITY_COL}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print("=" * 65)
    print("SALES DATA PROFILING")
    print("=" * 65)
    print(f"Input file       : {file_path}")
    print(f"Rows             : {len(df):,}")
    print(f"Columns          : {len(df.columns):,}")
    print(f"Column names     : {df.columns.tolist()}")

    return df


# ============================================================
# 3. PROFILE DATA COVERAGE
# ============================================================

def profile_data_coverage(df: pd.DataFrame) -> dict:
    """Summarize dataset size, date coverage, and SKU coverage."""

    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    valid_dates = dates.dropna()
    valid_skus = df[SKU_COL].dropna()

    date_min = valid_dates.min() if not valid_dates.empty else None
    date_max = valid_dates.max() if not valid_dates.empty else None

    calendar_days = (
        (date_max - date_min).days + 1
        if date_min is not None and date_max is not None
        else None
    )

    summary = {
        "dataset": INPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "date_column": DATE_COL,
        "sku_column": SKU_COL,
        "quantity_column": QUANTITY_COL,
        "date_coverage": {
            "earliest_date": (
                date_min.strftime("%Y-%m-%d")
                if date_min is not None
                else None
            ),
            "latest_date": (
                date_max.strftime("%Y-%m-%d")
                if date_max is not None
                else None
            ),
            "calendar_days_inclusive": (
                int(calendar_days)
                if calendar_days is not None
                else None
            ),
            "invalid_or_missing_dates": int(dates.isna().sum()),
        },
        "sku_coverage": {
            "unique_skus": int(valid_skus.nunique()),
            "missing_sku_rows": int(df[SKU_COL].isna().sum()),
        },
        "transaction_coverage": {
            "transactions_per_calendar_day": (
                round(len(df) / calendar_days, 2)
                if calendar_days
                else None
            ),
            "transactions_per_sku": (
                round(len(df) / valid_skus.nunique(), 2)
                if valid_skus.nunique()
                else None
            ),
        },
    }

    return summary


# ============================================================
# 4. PROFILE QUANTITY DISTRIBUTION
# ============================================================

def profile_quantity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for transaction quantities."""

    quantity = pd.to_numeric(df[QUANTITY_COL], errors="coerce")
    valid_quantity = quantity.dropna()

    metrics = {
        "row_count": int(len(df)),
        "valid_quantity_count": int(valid_quantity.count()),
        "missing_or_invalid_quantity_count": int(quantity.isna().sum()),
        "unique_quantity_values": int(valid_quantity.nunique()),
        "mean": valid_quantity.mean(),
        "median": valid_quantity.median(),
        "standard_deviation": valid_quantity.std(),
        "minimum": valid_quantity.min(),
        "p25": valid_quantity.quantile(0.25),
        "p75": valid_quantity.quantile(0.75),
        "p90": valid_quantity.quantile(0.90),
        "p95": valid_quantity.quantile(0.95),
        "p99": valid_quantity.quantile(0.99),
        "maximum": valid_quantity.max(),
        "zero_quantity_rows": int((valid_quantity == 0).sum()),
        "negative_quantity_rows": int((valid_quantity < 0).sum()),
        "positive_quantity_rows": int((valid_quantity > 0).sum()),
    }

    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )


# ============================================================
# 5. PROFILE SKU TRANSACTION COVERAGE
# ============================================================

def profile_sku_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize transaction counts and date coverage for each SKU."""

    working = df[[DATE_COL, SKU_COL, QUANTITY_COL]].copy()

    working["_parsed_date"] = pd.to_datetime(
        working[DATE_COL],
        errors="coerce",
    )

    working["_numeric_quantity"] = pd.to_numeric(
        working[QUANTITY_COL],
        errors="coerce",
    )

    working = working.dropna(subset=[SKU_COL])

    sku_profile = (
        working.groupby(SKU_COL)
        .agg(
            transaction_count=(SKU_COL, "size"),
            first_transaction_date=("_parsed_date", "min"),
            last_transaction_date=("_parsed_date", "max"),
            valid_quantity_count=("_numeric_quantity", "count"),
            total_recorded_quantity=("_numeric_quantity", "sum"),
            average_transaction_quantity=("_numeric_quantity", "mean"),
            median_transaction_quantity=("_numeric_quantity", "median"),
        )
        .reset_index()
    )

    sku_profile["historical_span_days"] = (
        sku_profile["last_transaction_date"]
        - sku_profile["first_transaction_date"]
    ).dt.days + 1

    for column in ["first_transaction_date", "last_transaction_date"]:
        sku_profile[column] = sku_profile[column].dt.strftime("%Y-%m-%d")

    return sku_profile.sort_values(
        "transaction_count",
        ascending=False,
    )


# ============================================================
# 6. PROFILE TRANSACTION FREQUENCY
# ============================================================

def profile_transaction_frequency(df: pd.DataFrame) -> dict:
    """Summarize transaction count distribution across SKUs."""

    sku_counts = df[SKU_COL].dropna().value_counts()

    if sku_counts.empty:
        return {
            "sku_count_with_transactions": 0,
            "minimum_transactions_per_sku": None,
            "median_transactions_per_sku": None,
            "mean_transactions_per_sku": None,
            "maximum_transactions_per_sku": None,
            "top_skus_by_transaction_count": [],
        }

    return {
        "sku_count_with_transactions": int(sku_counts.size),
        "minimum_transactions_per_sku": int(sku_counts.min()),
        "median_transactions_per_sku": float(sku_counts.median()),
        "mean_transactions_per_sku": round(float(sku_counts.mean()), 2),
        "maximum_transactions_per_sku": int(sku_counts.max()),
        "top_skus_by_transaction_count": [
            {
                "sku": str(sku),
                "transaction_count": int(count),
            }
            for sku, count in sku_counts.head(TOP_N_SKUS).items()
        ],
    }


# ============================================================
# 7. SAVE REPORTS
# ============================================================

def save_reports(
    summary: dict,
    sku_profile: pd.DataFrame,
    quantity_profile: pd.DataFrame,
) -> None:
    """Save structured profiling outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "sales_data_profile.json"
    sku_path = OUTPUT_DIR / "sku_transaction_coverage.csv"
    quantity_path = OUTPUT_DIR / "transaction_quantity_distribution.csv"

    summary_path.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    sku_profile.to_csv(sku_path, index=False)
    quantity_profile.to_csv(quantity_path, index=False)

    print("\n" + "=" * 65)
    print("PROFILE SUMMARY")
    print("=" * 65)
    print(json.dumps(summary, indent=4, ensure_ascii=False, default=str))

    print("\nQUANTITY DISTRIBUTION")
    print(quantity_profile.to_string(index=False))

    print("\nTOP 10 SKUs BY TRANSACTION COUNT")
    print(sku_profile.head(TOP_N_SKUS).to_string(index=False))

    print("\nOUTPUT FILES")
    for path in [summary_path, sku_path, quantity_path]:
        print(f"- {path.relative_to(PROJECT_ROOT).as_posix()}")


# ============================================================
# 8. MAIN
# ============================================================

def main() -> None:
    df = load_sales_data(INPUT_FILE)

    summary = profile_data_coverage(df)
    summary["transaction_frequency"] = profile_transaction_frequency(df)

    quantity_profile = profile_quantity_distribution(df)
    sku_profile = profile_sku_coverage(df)

    save_reports(
        summary=summary,
        sku_profile=sku_profile,
        quantity_profile=quantity_profile,
    )

    print("\nProfiling completed.")
    print("The source dataset was not modified.")


if __name__ == "__main__":
    main()