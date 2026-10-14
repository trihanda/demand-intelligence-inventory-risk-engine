"""
Purpose:
    Aggregate standardized sales transactions into weekly demand per SKU.

Input:
    data/processed/standardized_sales_transactions.csv

Outputs:
    data/processed/weekly_demand.csv
    reports/weekly_demand_aggregation/weekly_demand_aggregation_report.json

Usage:
    python src/aggregate_weekly_demand.py

Notes:
    - Weeks run from Monday through Sunday and are labeled by week_start.
    - The source dataset is not modified.
    - The output contains one row per SKU per week, including zero-demand
      weeks within the observed dataset coverage.
    - Zero-filling is appropriate here only under the synthetic-data assumption
      that transaction capture is complete. For real data, a missing transaction
      row must not automatically be interpreted as zero demand.
    - Demand is aggregated across warehouses. Warehouse-level analysis is
      reserved for later inventory-risk work.
"""

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "standardized_sales_transactions.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "weekly_demand_aggregation"

OUTPUT_FILE = PROCESSED_DIR / "weekly_demand.csv"
REPORT_FILE = REPORT_DIR / "weekly_demand_aggregation_report.json"

REQUIRED_COLUMNS = {"transaction_date", "sku", "quantity"}
WEEK_FREQUENCY = "W-SUN"


# ---------------------------------------------------------------------------
# Data loading and input validation
# ---------------------------------------------------------------------------

def load_standardized_sales(input_file: Path) -> pd.DataFrame:
    """Load standardized sales and validate required columns and values."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Run src/standardize_sales_data.py first."
        )

    sales = pd.read_csv(input_file)

    missing_columns = REQUIRED_COLUMNS - set(sales.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input is missing required columns: {missing}")

    if sales.empty:
        raise ValueError("Input sales dataset is empty.")

    sales["transaction_date"] = pd.to_datetime(
        sales["transaction_date"],
        errors="coerce",
    )
    sales["quantity"] = pd.to_numeric(
        sales["quantity"],
        errors="coerce",
    )

    invalid_mask = (
        sales["transaction_date"].isna()
        | sales["sku"].isna()
        | sales["quantity"].isna()
        | (sales["quantity"] < 0)
    )

    if invalid_mask.any():
        raise ValueError(
            f"Input contains {int(invalid_mask.sum())} rows with invalid dates, "
            "missing SKU/quantity, or negative quantity. Inspect the input "
            "before aggregation."
        )

    return sales


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def aggregate_weekly_demand(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quantity by SKU and week, then complete the SKU-week grid."""
    working = sales[["transaction_date", "sku", "quantity"]].copy()

    # W-SUN defines Monday-Sunday weeks, labeled by the Sunday week_end.
    working["week_end"] = (
        working["transaction_date"]
        .dt.to_period(WEEK_FREQUENCY)
        .dt.end_time
        .dt.normalize()
    )

    weekly_demand = (
        working.groupby(["sku", "week_end"], as_index=False)
        .agg(weekly_quantity=("quantity", "sum"))
    )

    # The synthetic generator assumes complete transaction capture.
    # Therefore, missing SKU-week combinations can be filled with zero.
    week_ends = pd.date_range(
        start=working["week_end"].min(),
        end=working["week_end"].max(),
        freq=WEEK_FREQUENCY,
    )
    skus = sorted(working["sku"].unique())

    complete_index = pd.MultiIndex.from_product(
        [skus, week_ends],
        names=["sku", "week_end"],
    )

    weekly_demand = (
        weekly_demand.set_index(["sku", "week_end"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    weekly_demand["week_start"] = (
        weekly_demand["week_end"] - pd.Timedelta(days=6)
    )

    weekly_demand = weekly_demand[
        ["week_start", "week_end", "sku", "weekly_quantity"]
    ].sort_values(
        ["sku", "week_start"],
        ignore_index=True,
    )
    weekly_demand["weekly_quantity"] = (
        weekly_demand["weekly_quantity"].astype("int64")
    )

    return weekly_demand


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------

def validate_weekly_demand(
    source_sales: pd.DataFrame,
    weekly_demand: pd.DataFrame,
) -> dict:
    """Validate the aggregated dataset and return check results."""
    return {
        "unique_sku_week_pairs": bool(
            not weekly_demand.duplicated(["sku", "week_end"]).any()
        ),
        "no_missing_values": bool(weekly_demand.notna().all().all()),
        "no_negative_weekly_quantity": bool(
            (weekly_demand["weekly_quantity"] >= 0).all()
        ),
        "quantity_reconciles_to_source": bool(
            weekly_demand["weekly_quantity"].sum()
            == source_sales["quantity"].sum()
        ),
    }


def build_report(
    source_sales: pd.DataFrame,
    weekly_demand: pd.DataFrame,
    input_file: Path,
    output_file: Path,
) -> dict:
    """Build a machine-readable summary of weekly demand aggregation."""
    return {
        "task": "weekly_demand_aggregation",
        "input_file": str(input_file.relative_to(PROJECT_ROOT)),
        "output_file": str(output_file.relative_to(PROJECT_ROOT)),
        "week_definition": "Monday-Sunday; week_end is Sunday",
        "aggregation_grain": "one row per SKU per week",
        "aggregation_measure": "sum of quantity",
        "zero_demand_handling": (
            "Missing SKU-week combinations are filled with zero under the "
            "synthetic-data completeness assumption."
        ),
        "source_rows": int(len(source_sales)),
        "source_skus": int(source_sales["sku"].nunique()),
        "source_date_min": (
            source_sales["transaction_date"].min().date().isoformat()
        ),
        "source_date_max": (
            source_sales["transaction_date"].max().date().isoformat()
        ),
        "output_rows": int(len(weekly_demand)),
        "output_skus": int(weekly_demand["sku"].nunique()),
        "output_weeks": int(weekly_demand["week_end"].nunique()),
        "zero_demand_sku_weeks": int(
            (weekly_demand["weekly_quantity"] == 0).sum()
        ),
        "total_weekly_quantity": int(
            weekly_demand["weekly_quantity"].sum()
        ),
        "validation": validate_weekly_demand(source_sales, weekly_demand),
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """Run weekly demand aggregation and write validated outputs."""
    sales = load_standardized_sales(INPUT_FILE)
    weekly_demand = aggregate_weekly_demand(sales)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = build_report(
        source_sales=sales,
        weekly_demand=weekly_demand,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
    )

    weekly_demand.to_csv(OUTPUT_FILE, index=False)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("WEEKLY DEMAND AGGREGATION")
    print(f"Source rows        : {report['source_rows']:,}")
    print(f"SKUs               : {report['source_skus']:,}")
    print(
        f"Date range         : {report['source_date_min']} "
        f"to {report['source_date_max']}"
    )
    print(f"Output rows        : {report['output_rows']:,}")
    print(f"Weeks              : {report['output_weeks']:,}")
    print(f"Zero-demand rows   : {report['zero_demand_sku_weeks']:,}")
    print("Validation:")
    for check, passed in report["validation"].items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
    print(f"Dataset            : {OUTPUT_FILE}")
    print(f"Report             : {REPORT_FILE}")

    if not all(report["validation"].values()):
        raise ValueError(
            "One or more aggregation validation checks failed. "
            f"Review {REPORT_FILE}."
        )

    print("Weekly demand aggregation completed successfully.")


if __name__ == "__main__":
    main()
