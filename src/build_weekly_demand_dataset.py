"""Build a complete weekly demand panel for physical inventory-managed hardware.

Run from repository root (after standardizing sales):
    python src/standardize_sales_data.py
    python src/build_weekly_demand_dataset.py

Inputs:
    data/processed/standardized_sales_transactions.csv
    data/synthetic/products.csv
    data/synthetic/calendar.csv

Outputs:
    data/processed/weekly_demand_hardware.csv
    data/processed/weekly_demand_hardware_partial_week.csv
    reports/weekly_demand/weekly_demand_build_report.json

Design:
    - Forecast target is weekly quantity sold per physical hardware SKU.
    - Weeks run Monday through Sunday; week_start is Monday, week_end is Sunday.
    - Only weeks fully covered by calendar.csv are included in the modeling panel.
    - The final partial week is retained separately for traceability, not training.
    - Missing SKU-week combinations are zero-filled under the synthetic complete-capture assumption.
    - Stable product attributes and known calendar features are included.
    - demand_type_seed, current lifecycle status, and current active status are intentionally
      excluded from the weekly model table to avoid label leakage or time-related leakage.
    - Source files are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SALES_COLUMNS = {"transaction_date", "sku", "quantity"}
REQUIRED_PRODUCT_COLUMNS = {
    "sku", "product_type", "physical_inventory_flag", "brand",
    "product_category", "product_family", "base_uom",
}
REQUIRED_CALENDAR_COLUMNS = {
    "calendar_date", "day_of_week", "is_weekday", "is_weekend",
    "is_public_holiday", "public_holiday_name",
    "is_collective_leave", "collective_leave_name",
    "is_company_closure", "company_closure_name",
    "is_business_day", "closure_reason", "business_day_type",
}
WEEK_FREQ = "W-SUN"  # Monday-Sunday weeks labelled by Sunday.
STABLE_PRODUCT_ATTRIBUTES = [
    "sku", "brand", "product_category", "product_family", "base_uom",
]


def parse_bool(series: pd.Series, column_name: str) -> pd.Series:
    """Parse booleans robustly from CSV values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }
    parsed = series.astype("string").str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        bad = series.loc[parsed.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(f"Could not parse boolean column '{column_name}'. Examples: {bad}")
    return parsed.astype(bool)


def read_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_path = root / "data" / "processed" / "standardized_sales_transactions.csv"
    products_path = root / "data" / "synthetic" / "products.csv"
    calendar_path = root / "data" / "synthetic" / "calendar.csv"

    missing = [p for p in (sales_path, products_path, calendar_path) if not p.exists()]
    if missing:
        details = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Required input file(s) not found:\n" + details +
            "\nRun `python src/standardize_sales_data.py` first and confirm that "
            "products.csv and calendar.csv exist in data/synthetic/."
        )

    sales = pd.read_csv(sales_path, low_memory=False)
    products = pd.read_csv(products_path, low_memory=False)
    calendar = pd.read_csv(calendar_path, low_memory=False)

    for label, frame, required in [
        ("standardized sales", sales, REQUIRED_SALES_COLUMNS),
        ("products", products, REQUIRED_PRODUCT_COLUMNS),
        ("calendar", calendar, REQUIRED_CALENDAR_COLUMNS),
    ]:
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{label} is missing columns: {missing_columns}")

    sales["transaction_date"] = pd.to_datetime(sales["transaction_date"], errors="coerce").dt.normalize()
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce")
    if sales["transaction_date"].isna().any():
        raise ValueError("Standardized sales contains invalid transaction_date values.")
    if sales["sku"].isna().any() or sales["sku"].astype(str).str.strip().eq("").any():
        raise ValueError("Standardized sales contains missing SKU values.")
    if sales["quantity"].isna().any() or (sales["quantity"] < 0).any():
        raise ValueError("Standardized sales must have numeric, non-negative quantities.")
    if not np.allclose(sales["quantity"], np.round(sales["quantity"])):
        raise ValueError("Hardware demand quantity contains fractional units; review the unit-of-measure rules.")

    products["physical_inventory_flag"] = parse_bool(products["physical_inventory_flag"], "physical_inventory_flag")
    products["product_type"] = products["product_type"].astype("string").str.strip().str.lower()
    if products["sku"].isna().any() or products["sku"].duplicated().any():
        raise ValueError("products.csv must contain one unique, non-missing row per SKU.")

    calendar["calendar_date"] = pd.to_datetime(calendar["calendar_date"], errors="coerce").dt.normalize()
    if calendar["calendar_date"].isna().any() or calendar["calendar_date"].duplicated().any():
        raise ValueError("calendar.csv must contain valid, unique calendar_date values.")
    for flag in ["is_weekday", "is_weekend", "is_public_holiday", "is_collective_leave", "is_company_closure", "is_business_day"]:
        calendar[flag] = parse_bool(calendar[flag], flag)
    calendar["closure_reason"] = calendar["closure_reason"].fillna("").astype(str).str.strip()
    for name_col in ["public_holiday_name", "collective_leave_name", "company_closure_name", "business_day_type"]:
        calendar[name_col] = calendar[name_col].fillna("").astype(str).str.strip()
    if not (calendar["is_weekend"] == ~calendar["is_weekday"]).all():
        raise ValueError("calendar.csv has inconsistent weekend/weekday flags.")
    if (calendar["is_business_day"] & ~calendar["is_weekday"]).any():
        raise ValueError("calendar.csv marks a weekend as a business day.")
    if (calendar["is_business_day"] & calendar["is_public_holiday"]).any():
        raise ValueError("calendar.csv marks a public holiday as a business day.")
    if calendar["collective_leave_policy"].nunique() > 1:
        raise ValueError("calendar.csv must use one consistent collective leave policy.")

    return sales, products, calendar


def weekly_calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Summarize holiday, collective-leave and business-day information by week."""
    work = calendar.copy()
    work["week_start"] = work["calendar_date"] - pd.to_timedelta(work["calendar_date"].dt.dayofweek, unit="D")
    work["is_nonbusiness_weekday_closure"] = work["is_weekday"] & ~work["is_business_day"]

    def combine_reasons(values: pd.Series) -> str:
        unique = sorted({str(v) for v in values if str(v).strip()})
        return ";".join(unique)

    def combine_names(values: pd.Series) -> str:
        unique = sorted({str(v) for v in values if str(v).strip()})
        return ";".join(unique)

    weekly = (
        work.groupby("week_start", as_index=False)
        .agg(
            calendar_days_covered=("calendar_date", "nunique"),
            business_days_in_week=("is_business_day", "sum"),
            weekend_days_in_week=("is_weekend", "sum"),
            closure_days_in_week=("is_nonbusiness_weekday_closure", "sum"),
            public_holiday_days_in_week=("is_public_holiday", "sum"),
            public_holiday_weekdays_in_week=("is_public_holiday", lambda s: int((s & work.loc[s.index, "is_weekday"]).sum())),
            collective_leave_days_in_week=("is_collective_leave", "sum"),
            collective_leave_weekdays_in_week=("is_collective_leave", lambda s: int((s & work.loc[s.index, "is_weekday"]).sum())),
            company_closure_days_in_week=("is_company_closure", "sum"),
            company_closure_weekdays_in_week=("is_company_closure", lambda s: int((s & work.loc[s.index, "is_weekday"]).sum())),
            public_holiday_name=("public_holiday_name", combine_names),
            collective_leave_name=("collective_leave_name", combine_names),
            company_closure_name=("company_closure_name", combine_names),
            closure_reason=("closure_reason", combine_reasons),
        )
    )
    weekly["week_end"] = weekly["week_start"] + pd.Timedelta(days=6)
    weekly["is_complete_week"] = weekly["calendar_days_covered"].eq(7)
    weekly["is_closure_affected_week"] = weekly["closure_days_in_week"].gt(0)
    weekly["is_public_holiday_week"] = weekly["public_holiday_weekdays_in_week"].gt(0)
    weekly["is_collective_leave_week"] = weekly["collective_leave_weekdays_in_week"].gt(0)
    weekly["is_company_closure_week"] = weekly["company_closure_weekdays_in_week"].gt(0)
    weekly["is_full_closure_week"] = (
        weekly["is_complete_week"] & weekly["business_days_in_week"].eq(0)
    )
    weekly["week_status"] = np.where(weekly["is_complete_week"], "complete_week", "partial_week")
    return weekly


def build_weekly_panel(
    sales: pd.DataFrame, products: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Filter hardware sales, aggregate, zero-fill, and separate partial-week data."""
    product_cols = STABLE_PRODUCT_ATTRIBUTES + ["product_type", "physical_inventory_flag"]
    physical_hardware = products.loc[
        products["physical_inventory_flag"] & products["product_type"].eq("hardware"),
        product_cols,
    ].copy()
    if physical_hardware.empty:
        raise ValueError("No physical inventory-managed hardware SKUs found in products.csv.")

    hardware_skus = sorted(physical_hardware["sku"].astype(str).unique().tolist())
    sales = sales.copy()
    sales["sku"] = sales["sku"].astype(str).str.strip()
    sales["week_start"] = sales["transaction_date"] - pd.to_timedelta(
        sales["transaction_date"].dt.dayofweek, unit="D"
    )
    sales["week_end"] = sales["week_start"] + pd.Timedelta(days=6)

    unknown_skus = sorted(set(sales["sku"].unique()) - set(products["sku"].astype(str).unique()))
    if unknown_skus:
        raise ValueError(f"Sales contain SKUs missing from products.csv. Examples: {unknown_skus[:10]}")

    hardware_sales = sales.loc[sales["sku"].isin(hardware_skus)].copy()
    week_cal = weekly_calendar_features(calendar)
    week_calendar_map = week_cal.set_index("week_start")
    mapped = hardware_sales["week_start"].isin(week_calendar_map.index)
    if not mapped.all():
        examples = hardware_sales.loc[~mapped, "transaction_date"].head().astype(str).tolist()
        raise ValueError(f"Sales dates are outside calendar coverage. Examples: {examples}")

    hardware_sales = hardware_sales.merge(
        week_cal[["week_start", "is_complete_week"]],
        on="week_start", how="left", validate="many_to_one"
    )
    partial_sales = hardware_sales.loc[~hardware_sales["is_complete_week"]].copy()
    complete_sales = hardware_sales.loc[hardware_sales["is_complete_week"]].copy()

    # Use only full weeks completely covered by calendar.csv and within the transaction data horizon.
    if complete_sales.empty:
        raise ValueError("No hardware transactions fall within complete calendar weeks.")
    eligible_weeks = week_cal.loc[week_cal["is_complete_week"]].copy()
    eligible_weeks = eligible_weeks.loc[
        (eligible_weeks["week_start"] >= complete_sales["week_start"].min())
        & (eligible_weeks["week_start"] <= complete_sales["week_start"].max())
    ].sort_values("week_start")
    if eligible_weeks.empty:
        raise ValueError("No eligible complete weeks found for weekly demand panel.")
    week_starts = pd.DatetimeIndex(eligible_weeks["week_start"].drop_duplicates())

    weekly_observed = (
        complete_sales.groupby(["sku", "week_start"], as_index=False)
        .agg(weekly_quantity=("quantity", "sum"))
    )
    full_index = pd.MultiIndex.from_product(
        [hardware_skus, week_starts], names=["sku", "week_start"]
    )
    panel = weekly_observed.set_index(["sku", "week_start"]).reindex(full_index, fill_value=0).reset_index()
    panel["weekly_quantity"] = panel["weekly_quantity"].astype("int64")
    panel["week_end"] = panel["week_start"] + pd.Timedelta(days=6)

    panel = panel.merge(
        eligible_weeks[
            ["week_start", "business_days_in_week", "weekend_days_in_week",
             "closure_days_in_week", "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
             "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
             "company_closure_days_in_week", "company_closure_weekdays_in_week",
             "public_holiday_name", "collective_leave_name", "company_closure_name",
             "closure_reason", "is_closure_affected_week", "is_public_holiday_week",
             "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week"]
        ], on="week_start", how="left", validate="many_to_one"
    )
    panel = panel.merge(physical_hardware, on="sku", how="left", validate="many_to_one")
    # Use an explicit sentinel instead of blank cells so CSV readers do not reinterpret
    # non-closure weeks as missing data.
    panel["closure_reason"] = panel["closure_reason"].fillna("none").replace("", "none").astype(str)
    count_columns = [
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
        "company_closure_days_in_week", "company_closure_weekdays_in_week",
    ]
    for column in count_columns:
        panel[column] = panel[column].astype("int8")
    for column in ["is_closure_affected_week", "is_public_holiday_week", "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week"]:
        panel[column] = panel[column].astype(bool)
    for column in ["public_holiday_name", "collective_leave_name", "company_closure_name"]:
        panel[column] = panel[column].fillna("none").replace("", "none").astype(str)
    panel["week_status"] = "complete_week"

    # Keep a separate, traceable file for transactions assigned to any incomplete calendar week.
    # This is not included in the model panel.
    partial_output = partial_sales.copy()
    if not partial_output.empty:
        partial_output = partial_output[
            [c for c in ["invoice_line_id", "invoice_id", "transaction_date", "sku", "warehouse_id", "quantity", "transaction_status"] if c in partial_output.columns]
        ].sort_values(["transaction_date", "sku"], ignore_index=True)

    expected_rows = len(hardware_skus) * len(week_starts)
    summary = {
        "hardware_sku_count": int(len(hardware_skus)),
        "complete_week_count": int(len(week_starts)),
        "first_week_start": week_starts.min().date().isoformat(),
        "last_week_start": week_starts.max().date().isoformat(),
        "last_week_end": (week_starts.max() + pd.Timedelta(days=6)).date().isoformat(),
        "expected_panel_rows": int(expected_rows),
        "partial_week_hardware_transaction_rows_excluded": int(len(partial_sales)),
        "partial_week_hardware_quantity_excluded": int(partial_sales["quantity"].sum()) if not partial_sales.empty else 0,
        "partial_week_min_date": partial_sales["transaction_date"].min().date().isoformat() if not partial_sales.empty else None,
        "partial_week_max_date": partial_sales["transaction_date"].max().date().isoformat() if not partial_sales.empty else None,
        "full_closure_week_count": int(eligible_weeks["is_full_closure_week"].sum()),
        "closure_affected_week_count": int(eligible_weeks["is_closure_affected_week"].sum()),
        "public_holiday_affected_week_count": int(eligible_weeks["is_public_holiday_week"].sum()),
        "collective_leave_affected_week_count": int(eligible_weeks["is_collective_leave_week"].sum()),
        "company_closure_affected_week_count": int(eligible_weeks["is_company_closure_week"].sum()),
        "total_public_holiday_weekday_count": int(eligible_weeks["public_holiday_weekdays_in_week"].sum()),
        "total_collective_leave_weekday_count": int(eligible_weeks["collective_leave_weekdays_in_week"].sum()),
        "positive_sku_week_count": int(panel["weekly_quantity"].gt(0).sum()),
        "zero_sku_week_count": int(panel["weekly_quantity"].eq(0).sum()),
        "zero_sku_week_share": float(panel["weekly_quantity"].eq(0).mean()),
        "total_weekly_quantity": int(panel["weekly_quantity"].sum()),
        "complete_week_hardware_quantity_source": int(complete_sales["quantity"].sum()),
        "distinct_model_feature_fields_excluded": ["demand_type_seed", "lifecycle_status", "active_flag"],
    }
    return panel, partial_output, summary


def validate_panel(panel: pd.DataFrame, summary: dict[str, Any], weekly_calendar: pd.DataFrame) -> dict[str, bool]:
    """Run structural and arithmetic validations for the weekly target panel."""
    closure_weeks = set(
        weekly_calendar.loc[weekly_calendar["is_full_closure_week"], "week_start"]
    )
    full_closure_rows = panel["week_start"].isin(closure_weeks)
    return {
        "one_row_per_sku_week": not panel.duplicated(["sku", "week_start"]).any(),
        "complete_sku_week_grid": len(panel) == summary["expected_panel_rows"],
        "all_hardware_skus_present": panel["sku"].nunique() == summary["hardware_sku_count"],
        "no_missing_values_in_output": not panel.isna().any().any(),
        "weekly_quantity_reconciles_to_complete_week_sales": int(panel["weekly_quantity"].sum()) == int(summary["complete_week_hardware_quantity_source"]),
        "no_negative_weekly_quantity": bool((panel["weekly_quantity"] >= 0).all()),
        "quantity_is_integer": bool(np.allclose(panel["weekly_quantity"], np.round(panel["weekly_quantity"]))),
        "last_week_is_complete": bool((panel["week_end"].max() - panel["week_start"].max()).days == 6),
        "full_closure_weeks_have_zero_demand": bool(panel.loc[full_closure_rows, "weekly_quantity"].eq(0).all()),
        "no_demand_type_seed_in_output": "demand_type_seed" not in panel.columns,
        "no_current_lifecycle_or_active_flags_in_output": not ({"lifecycle_status", "active_flag"} & set(panel.columns)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root; defaults to parent of src/.")
    args = parser.parse_args()
    root = args.project_root.resolve()

    sales, products, calendar = read_inputs(root)
    panel, partial_output, summary = build_weekly_panel(sales, products, calendar)
    week_cal = weekly_calendar_features(calendar)
    validation = validate_panel(panel, summary, week_cal)

    output_dir = root / "data" / "processed"
    report_dir = root / "reports" / "weekly_demand"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "weekly_demand_hardware.csv"
    partial_path = output_dir / "weekly_demand_hardware_partial_week.csv"
    report_path = report_dir / "weekly_demand_build_report.json"

    columns = [
        "week_start", "week_end", "sku", "weekly_quantity",
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
        "company_closure_days_in_week", "company_closure_weekdays_in_week",
        "public_holiday_name", "collective_leave_name", "company_closure_name",
        "closure_reason", "is_closure_affected_week", "is_public_holiday_week",
        "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week",
        "week_status", "brand", "product_category", "product_family", "base_uom",
        "product_type", "physical_inventory_flag",
    ]
    panel = panel[columns].sort_values(["sku", "week_start"], ignore_index=True)
    panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    partial_output.to_csv(partial_path, index=False, date_format="%Y-%m-%d")

    report: dict[str, Any] = {
        "project": "Demand Intelligence & Inventory Risk Engine",
        "task": "weekly_hardware_demand_panel",
        "synthetic_data": True,
        "grain": "one row per physical hardware SKU per complete Monday-Sunday week",
        "target": "weekly_quantity = sum of valid posted sales quantity per SKU-week",
        "source_file": "data/processed/standardized_sales_transactions.csv",
        "product_file": "data/synthetic/products.csv",
        "calendar_file": "data/synthetic/calendar.csv",
        "output_file": "data/processed/weekly_demand_hardware.csv",
        "partial_week_trace_file": "data/processed/weekly_demand_hardware_partial_week.csv",
        "week_definition": "Monday through Sunday; week_start labels Monday and week_end labels Sunday",
        "zero_fill_assumption": "Synthetic transactions are assumed complete; missing hardware SKU-week combinations are filled with zero.",
        "quality_rules": [
            "Uses standardized posted transactions only.",
            "Excludes non-hardware products and products without physical_inventory_flag=True.",
            "Excludes incomplete calendar weeks from modeling panel and writes their hardware transactions to a separate trace file.",
            "Keeps missing customer IDs when the transaction otherwise passes standardization.",
            "Does not include demand_type_seed, lifecycle_status, or active_flag in the weekly table.",
            "Full-closure weeks remain zero-demand rows, with explicit calendar features.",
        ],
        "source_standardized_rows": int(len(sales)),
        "source_standardized_unique_skus": int(sales["sku"].nunique()),
        "source_standardized_date_min": sales["transaction_date"].min().date().isoformat(),
        "source_standardized_date_max": sales["transaction_date"].max().date().isoformat(),
        **summary,
        "output_rows": int(len(panel)),
        "output_unique_skus": int(panel["sku"].nunique()),
        "output_unique_weeks": int(panel["week_start"].nunique()),
        "output_total_weekly_quantity": int(panel["weekly_quantity"].sum()),
        "validation": validation,
        "all_validations_pass": bool(all(validation.values())),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== WEEKLY HARDWARE DEMAND BUILD REPORT ===")
    print(f"Project root                 : {root}")
    print(f"Standardized sales rows      : {len(sales):,}")
    print(f"Hardware SKU count           : {summary['hardware_sku_count']:,}")
    print(f"Complete weeks               : {summary['complete_week_count']:,}")
    print(f"Week range                   : {summary['first_week_start']} to {summary['last_week_end']}")
    print(f"Output panel rows            : {len(panel):,}")
    print(f"Zero-demand SKU-weeks        : {summary['zero_sku_week_count']:,} ({summary['zero_sku_week_share']:.1%})")
    print(f"Full closure weeks           : {summary['full_closure_week_count']:,}")
    print(f"Partial-week rows excluded   : {summary['partial_week_hardware_transaction_rows_excluded']:,}")
    print(f"Partial-week quantity        : {summary['partial_week_hardware_quantity_excluded']:,}")
    print("\nValidation:")
    for name, passed in validation.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("\nOutputs:")
    print(f"  {output_path}")
    print(f"  {partial_path}")
    print(f"  {report_path}")

    if not report["all_validations_pass"]:
        raise SystemExit("Weekly demand output generated, but one or more validation checks failed. Review the report.")


if __name__ == "__main__":
    main()
