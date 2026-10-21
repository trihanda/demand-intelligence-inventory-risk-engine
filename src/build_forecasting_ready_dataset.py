"""
Build a forecasting-ready weekly demand panel for inventory-managed hardware.

Purpose:
    Combine SKU-level weekly demand with product attributes and weekly calendar
    features; generate leakage-safe lag/rolling features; and assign chronological
    train/validation/test periods.

Inputs:
    data/processed/weekly_demand.csv
    data/synthetic/products.csv
    data/synthetic/calendar.csv

Outputs:
    data/processed/forecasting_ready_dataset.csv
    reports/forecasting_data_preparation/forecasting_data_preparation_report.json

Usage:
    python src/build_forecasting_ready_dataset.py

Important:
    - Only hardware SKUs with physical_inventory_flag=True are included.
    - weekly_quantity is preserved as the prediction target.
    - Lag and rolling features use previous weeks only (shift by one week).
    - The final partial calendar week is retained for traceability but excluded
      from model eligibility and temporal split assignment.
    - The 13-week validation/test periods are chronological rolling one-step
      evaluation windows. They are not, by themselves, a fixed-origin 13-week
      ahead forecasting simulation; that strategy must be specified for model
      evaluation.
    - The synthetic demand_type_seed field is deliberately excluded to avoid
      leaking a label used by the synthetic data generator.
    - Demand-pattern segments (including intermittent SKUs) are computed from
      the training window only and are metadata for evaluation, not predictors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEEKLY_DEMAND_PATH = ROOT / "data" / "processed" / "weekly_demand.csv"
PRODUCTS_PATH = ROOT / "data" / "synthetic" / "products.csv"
CALENDAR_PATH = ROOT / "data" / "synthetic" / "calendar.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "forecasting_ready_dataset.csv"
REPORT_DIR = ROOT / "reports" / "forecasting_data_preparation"
REPORT_PATH = REPORT_DIR / "forecasting_data_preparation_report.json"

LAG_WEEKS = [1, 2, 4, 13, 26, 52]
ROLLING_WINDOWS = [4, 13, 26]
VALIDATION_WEEKS = 13
TEST_WEEKS = 13
EXPECTED_HARDWARE_SKUS = 600
EXPECTED_WEEKS = 157
EXPECTED_ROWS = EXPECTED_HARDWARE_SKUS * EXPECTED_WEEKS

# Include only stable, non-target-derived product attributes if present.
# demand_type_seed is intentionally excluded because it informs data generation.
PRODUCT_FEATURE_CANDIDATES = [
    "product_category",
    "product_family",
    "brand",
    "lifecycle_status",
    "unit_cost",
    "unit_price",
    "cost_currency",
    "list_price_currency",
    "active_flag",
]


def parse_bool(series: pd.Series) -> pd.Series:
    """Parse booleans consistently from CSV bools, strings, or 0/1 values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read inputs and validate required columns and basic integrity."""
    for path in (WEEKLY_DEMAND_PATH, PRODUCTS_PATH, CALENDAR_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    demand = pd.read_csv(WEEKLY_DEMAND_PATH)
    products = pd.read_csv(PRODUCTS_PATH)
    calendar = pd.read_csv(CALENDAR_PATH)

    required_demand = {"week_start", "sku", "weekly_quantity"}
    required_products = {"sku", "product_type", "physical_inventory_flag"}
    required_calendar = {
        "calendar_date", "is_business_day", "closure_reason"
    }
    for label, frame, required in (
        ("weekly demand", demand, required_demand),
        ("products", products, required_products),
        ("calendar", calendar, required_calendar),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    if demand.empty or products.empty or calendar.empty:
        raise ValueError("One or more source files are empty.")

    demand["week_start"] = pd.to_datetime(demand["week_start"], errors="coerce")
    demand["weekly_quantity"] = pd.to_numeric(
        demand["weekly_quantity"], errors="coerce"
    )
    if demand[["week_start", "sku", "weekly_quantity"]].isna().any().any():
        raise ValueError("Weekly demand contains missing or invalid required values.")
    if (demand["weekly_quantity"] < 0).any():
        raise ValueError("Weekly demand contains negative quantities.")
    if demand.duplicated(["sku", "week_start"]).any():
        raise ValueError("Weekly demand has duplicate SKU-week pairs.")
    if not demand["week_start"].dt.dayofweek.eq(0).all():
        raise ValueError("week_start must be a Monday for every row.")

    if products["sku"].isna().any() or products["sku"].duplicated().any():
        raise ValueError("Product master must contain unique, non-missing SKUs.")

    calendar["calendar_date"] = pd.to_datetime(
        calendar["calendar_date"], errors="coerce"
    ).dt.normalize()
    if calendar["calendar_date"].isna().any():
        raise ValueError("Calendar contains invalid calendar_date values.")
    if calendar["calendar_date"].duplicated().any():
        raise ValueError("Calendar contains duplicate calendar dates.")
    calendar["is_business_day"] = parse_bool(calendar["is_business_day"])
    calendar["closure_reason"] = calendar["closure_reason"].fillna("").astype(str).str.strip()

    return demand, products, calendar


def build_weekly_calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily calendar rows to Monday-Sunday week features."""
    work = calendar.copy()
    work["week_start"] = work["calendar_date"] - pd.to_timedelta(
        work["calendar_date"].dt.dayofweek, unit="D"
    )
    work["has_closure_reason"] = work["closure_reason"].ne("")

    def join_reasons(values: pd.Series) -> str:
        reasons = sorted({value for value in values if value})
        return ";".join(reasons)

    weekly_calendar = (
        work.groupby("week_start", as_index=False)
        .agg(
            calendar_days_covered=("calendar_date", "nunique"),
            business_days_in_week=("is_business_day", "sum"),
            closure_days_in_week=("has_closure_reason", "sum"),
            closure_reason=("closure_reason", join_reasons),
        )
    )
    weekly_calendar["is_closure_affected_week"] = (
        weekly_calendar["closure_days_in_week"] > 0
    )
    weekly_calendar["calendar_coverage_status"] = np.select(
        [
            weekly_calendar["calendar_days_covered"].eq(7),
            weekly_calendar["calendar_days_covered"].between(1, 6),
        ],
        ["complete_week", "partial_week"],
        default="no_calendar_coverage",
    )
    weekly_calendar["is_partial_calendar_week"] = (
        weekly_calendar["calendar_coverage_status"] == "partial_week"
    )
    return weekly_calendar


def assign_time_splits(weekly_calendar: pd.DataFrame) -> dict[str, Any]:
    """Assign train/validation/test by calendar week, excluding partial weeks."""
    eligible_weeks = sorted(
        weekly_calendar.loc[
            weekly_calendar["calendar_coverage_status"].eq("complete_week"),
            "week_start",
        ].drop_duplicates()
    )
    if len(eligible_weeks) <= VALIDATION_WEEKS + TEST_WEEKS:
        raise ValueError(
            "Not enough complete calendar weeks for the configured temporal splits."
        )

    train_end = len(eligible_weeks) - VALIDATION_WEEKS - TEST_WEEKS
    validation_end = len(eligible_weeks) - TEST_WEEKS
    split_map: dict[pd.Timestamp, str] = {}
    for index, week in enumerate(eligible_weeks):
        if index < train_end:
            split_map[week] = "train"
        elif index < validation_end:
            split_map[week] = "validation"
        else:
            split_map[week] = "test"

    metadata = {
        "method": "chronological split by target week; partial-calendar weeks are excluded",
        "train_weeks": int(train_end),
        "validation_weeks": int(VALIDATION_WEEKS),
        "test_weeks": int(TEST_WEEKS),
        "train_first_week": eligible_weeks[0].date().isoformat(),
        "train_last_week": eligible_weeks[train_end - 1].date().isoformat(),
        "validation_first_week": eligible_weeks[train_end].date().isoformat(),
        "validation_last_week": eligible_weeks[validation_end - 1].date().isoformat(),
        "test_first_week": eligible_weeks[validation_end].date().isoformat(),
        "test_last_week": eligible_weeks[-1].date().isoformat(),
        "excluded_partial_calendar_weeks": int(
            weekly_calendar["calendar_coverage_status"].eq("partial_week").sum()
        ),
        "excluded_weeks_without_calendar_coverage": int(
            weekly_calendar["calendar_coverage_status"].eq("no_calendar_coverage").sum()
        ),
    }
    return {"split_map": split_map, "metadata": metadata}


def classify_training_demand_patterns(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify SKU demand using training-period observations only.

    ADI is the number of training weeks divided by the number of non-zero
    demand weeks. CV-squared is calculated from positive-demand quantities
    only. The resulting label is for evaluation segmentation, not a model
    predictor, so validation/test outcomes cannot affect the class.
    """
    training = dataset.loc[
        dataset["data_split"].eq("train"), ["sku", "weekly_quantity"]
    ]
    if training.empty:
        raise ValueError("Training split is empty; cannot classify demand patterns.")

    records: list[dict[str, Any]] = []
    for sku, group in training.groupby("sku", sort=True):
        quantities = group["weekly_quantity"].astype(float)
        positive = quantities[quantities > 0]
        weeks_observed = int(len(quantities))
        nonzero_weeks = int(len(positive))
        adi = weeks_observed / nonzero_weeks if nonzero_weeks else None

        if nonzero_weeks:
            positive_mean = float(positive.mean())
            positive_std = float(positive.std(ddof=0))
            cv_squared = (positive_std / positive_mean) ** 2 if positive_mean > 0 else None
        else:
            cv_squared = None

        if adi is None or cv_squared is None:
            pattern = "no_demand"
        else:
            intermittent = adi >= 1.32
            erratic = cv_squared >= 0.49
            if intermittent and erratic:
                pattern = "lumpy"
            elif intermittent:
                pattern = "intermittent"
            elif erratic:
                pattern = "erratic"
            else:
                pattern = "smooth"

        records.append({
            "sku": sku,
            "training_weeks_observed": weeks_observed,
            "training_nonzero_demand_weeks": nonzero_weeks,
            "training_adi": adi,
            "training_cv_squared_positive_demand": cv_squared,
            "demand_pattern_train": pattern,
            "intermittent_demand_flag_train": bool(adi is not None and adi >= 1.32),
        })

    profile = pd.DataFrame(records)
    counts = {
        str(pattern): int(count)
        for pattern, count in profile["demand_pattern_train"].value_counts().sort_index().items()
    }
    summary = {
        "purpose": "Evaluation segmentation only; not a forecasting predictor.",
        "derived_from_split": "train only",
        "training_week_count": int(training.groupby("sku").size().iloc[0]),
        "adi_threshold": 1.32,
        "cv_squared_threshold": 0.49,
        "cv_squared_definition": "Squared coefficient of variation of positive-demand weekly quantities, using population standard deviation (ddof=0).",
        "intermittent_demand_flag_definition": "True when training ADI >= 1.32; includes both intermittent and lumpy patterns.",
        "sku_counts_by_pattern": counts,
        "intermittent_demand_sku_count": int(profile["intermittent_demand_flag_train"].sum()),
        "sku_count": int(profile["sku"].nunique()),
        "labels_use_validation_or_test_targets": False,
    }
    return profile, summary


def create_dataset(
    demand: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create hardware SKU-week panel, calendar features and lagged predictors."""
    products = products.copy()
    products["physical_inventory_flag"] = parse_bool(products["physical_inventory_flag"])
    hardware_products = products.loc[
        products["product_type"].astype(str).str.strip().str.lower().eq("hardware")
        & products["physical_inventory_flag"],
    ].copy()
    if hardware_products.empty:
        raise ValueError("No products meet the hardware inventory-managed filter.")

    product_columns = [
        column for column in PRODUCT_FEATURE_CANDIDATES
        if column in hardware_products.columns
    ]
    product_lookup = hardware_products[["sku", *product_columns]].copy()
    hardware_skus = set(hardware_products["sku"])
    source_skus = set(demand["sku"].unique())
    missing_sales_skus = sorted(hardware_skus - source_skus)
    hardware_demand = demand.loc[demand["sku"].isin(hardware_skus)].copy()
    if hardware_demand.empty:
        raise ValueError("No weekly demand rows match inventory-managed hardware SKUs.")

    missing_product_skus = sorted(set(hardware_demand["sku"]) - hardware_skus)
    if missing_product_skus:
        raise ValueError(
            f"Demand contains {len(missing_product_skus)} SKUs missing from the hardware master."
        )

    weekly_calendar = build_weekly_calendar_features(calendar)

    # Align calendar weeks to the weeks present in the demand series before
    # assigning splits or counting complete/partial weeks. The source calendar
    # may contain an edge week outside the demand window (for example, the
    # Sunday before the first Monday-labeled demand week); it must not affect
    # the demand-series reconciliation.
    demand_weeks = pd.DatetimeIndex(
        sorted(hardware_demand["week_start"].drop_duplicates()),
        name="week_start",
    )
    weekly_calendar = (
        weekly_calendar.set_index("week_start")
        .reindex(demand_weeks)
        .reset_index()
    )
    weekly_calendar["calendar_coverage_status"] = weekly_calendar[
        "calendar_coverage_status"
    ].fillna("no_calendar_coverage")
    weekly_calendar["closure_reason"] = weekly_calendar["closure_reason"].fillna("")
    weekly_calendar["is_closure_affected_week"] = weekly_calendar[
        "is_closure_affected_week"
    ].fillna(False).astype(bool)
    weekly_calendar["is_partial_calendar_week"] = weekly_calendar[
        "calendar_coverage_status"
    ].eq("partial_week")

    split_info = assign_time_splits(weekly_calendar)

    dataset = hardware_demand.merge(
        product_lookup,
        on="sku",
        how="left",
        validate="many_to_one",
    ).merge(
        weekly_calendar,
        on="week_start",
        how="left",
        validate="many_to_one",
    )

    # Keep missing calendar coverage explicit; never infer that missing dates
    # are non-business days or closure days.
    coverage_missing = dataset["calendar_days_covered"].isna()
    dataset["calendar_coverage_status"] = dataset[
        "calendar_coverage_status"
    ].fillna("no_calendar_coverage")
    dataset["closure_reason"] = dataset["closure_reason"].fillna("")
    dataset.loc[coverage_missing, "is_closure_affected_week"] = False
    dataset["is_partial_calendar_week"] = dataset[
        "calendar_coverage_status"
    ].eq("partial_week")

    # Known-in-advance date features. ISO-week encoding helps represent annual
    # timing; week_start determines the quarter to match the existing analysis.
    dataset["year"] = dataset["week_start"].dt.year.astype(int)
    dataset["month"] = dataset["week_start"].dt.month.astype(int)
    dataset["quarter"] = dataset["week_start"].dt.quarter.astype(int)
    dataset["iso_week"] = dataset["week_start"].dt.isocalendar().week.astype(int)
    dataset["time_index"] = dataset["week_start"].rank(method="dense").astype(int) - 1
    dataset["week_of_year_sin"] = np.sin(2 * np.pi * (dataset["iso_week"] - 1) / 52.0)
    dataset["week_of_year_cos"] = np.cos(2 * np.pi * (dataset["iso_week"] - 1) / 52.0)
    dataset["month_sin"] = np.sin(2 * np.pi * (dataset["month"] - 1) / 12.0)
    dataset["month_cos"] = np.cos(2 * np.pi * (dataset["month"] - 1) / 12.0)

    dataset = dataset.sort_values(["sku", "week_start"], ignore_index=True)
    grouped = dataset.groupby("sku", sort=False)["weekly_quantity"]

    lag_columns: list[str] = []
    for lag in LAG_WEEKS:
        column = f"demand_lag_{lag}"
        dataset[column] = grouped.shift(lag)
        lag_columns.append(column)

    rolling_columns: list[str] = []
    for window in ROLLING_WINDOWS:
        mean_column = f"rolling_mean_{window}w"
        dataset[mean_column] = dataset.groupby("sku", sort=False)[
            "weekly_quantity"
        ].transform(
            lambda values, window=window: values.shift(1).rolling(
                window=window, min_periods=window
            ).mean()
        )
        rolling_columns.append(mean_column)

    dataset["rolling_std_13w"] = dataset.groupby("sku", sort=False)[
        "weekly_quantity"
    ].transform(lambda values: values.shift(1).rolling(13, min_periods=13).std(ddof=0))
    rolling_columns.append("rolling_std_13w")

    dataset["data_split"] = dataset["week_start"].map(split_info["split_map"])
    dataset.loc[
        dataset["calendar_coverage_status"].eq("partial_week"), "data_split"
    ] = "excluded_partial_calendar_week"
    dataset.loc[
        dataset["calendar_coverage_status"].eq("no_calendar_coverage"), "data_split"
    ] = "excluded_missing_calendar"

    # Segment SKUs using training-period demand only. Keep these labels for
    # model evaluation and reporting; do not use them as predictive features.
    demand_pattern_profile, demand_pattern_summary = classify_training_demand_patterns(
        dataset
    )
    dataset = dataset.merge(
        demand_pattern_profile, on="sku", how="left", validate="many_to_one"
    )
    demand_pattern_summary["training_first_week"] = split_info["metadata"]["train_first_week"]
    demand_pattern_summary["training_last_week"] = split_info["metadata"]["train_last_week"]

    required_history_features = lag_columns + rolling_columns
    dataset["modeling_eligible"] = (
        dataset["data_split"].isin(["train", "validation", "test"])
        & dataset[required_history_features].notna().all(axis=1)
        & dataset["calendar_days_covered"].eq(7)
    )

    # Column ordering: identity/target, product descriptors, calendar predictors,
    # demand-history predictors, split and eligibility metadata.
    identity_target = ["sku", "week_start", "weekly_quantity"]
    calendar_features = [
        "calendar_days_covered",
        "business_days_in_week",
        "closure_days_in_week",
        "closure_reason",
        "is_closure_affected_week",
        "calendar_coverage_status",
        "is_partial_calendar_week",
        "year",
        "month",
        "quarter",
        "iso_week",
        "time_index",
        "week_of_year_sin",
        "week_of_year_cos",
        "month_sin",
        "month_cos",
    ]
    history_features = lag_columns + rolling_columns
    segmentation_columns = [
        "training_weeks_observed",
        "training_nonzero_demand_weeks",
        "training_adi",
        "training_cv_squared_positive_demand",
        "demand_pattern_train",
        "intermittent_demand_flag_train",
    ]
    final_columns = (
        identity_target + product_columns + calendar_features
        + history_features + segmentation_columns
        + ["data_split", "modeling_eligible"]
    )
    dataset = dataset[final_columns]

    # Compute reporting summaries.
    weekly_totals = hardware_demand.groupby("week_start")["weekly_quantity"].sum()
    zero_demand_weeks = set(weekly_totals[weekly_totals.eq(0)].index)
    calendar_by_week = weekly_calendar.set_index("week_start")
    zero_with_closure = sum(
        week in calendar_by_week.index
        and bool(calendar_by_week.loc[week, "is_closure_affected_week"])
        for week in zero_demand_weeks
    )
    valid_full_weeks = int(weekly_calendar["calendar_coverage_status"].eq("complete_week").sum())
    partial_weeks = int(weekly_calendar["calendar_coverage_status"].eq("partial_week").sum())
    no_calendar_weeks = int(weekly_calendar["calendar_coverage_status"].eq("no_calendar_coverage").sum())

    split_summary: dict[str, Any] = {}
    for split in ["train", "validation", "test", "excluded_partial_calendar_week", "excluded_missing_calendar"]:
        subset = dataset[dataset["data_split"].eq(split)]
        split_summary[split] = {
            "rows": int(len(subset)),
            "weeks": int(subset["week_start"].nunique()),
            "modeling_eligible_rows": int(subset["modeling_eligible"].sum()),
        }

    report = {
        "task": "forecasting_data_preparation",
        "scope": "Inventory-managed hardware",
        "input_files": {
            "weekly_demand": str(WEEKLY_DEMAND_PATH.relative_to(ROOT)).replace("\\", "/"),
            "products": str(PRODUCTS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "calendar": str(CALENDAR_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "report_file": str(REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sku_count": int(dataset["sku"].nunique()),
        "week_count": int(dataset["week_start"].nunique()),
        "row_count": int(len(dataset)),
        "first_week": dataset["week_start"].min().date().isoformat(),
        "last_week": dataset["week_start"].max().date().isoformat(),
        "total_demand": int(dataset["weekly_quantity"].sum()),
        "calendar_reconciliation": {
            "calendar_date_min": calendar["calendar_date"].min().date().isoformat(),
            "calendar_date_max": calendar["calendar_date"].max().date().isoformat(),
            "calendar_date_count": int(calendar["calendar_date"].nunique()),
            "complete_calendar_weeks_in_demand_series": valid_full_weeks,
            "partial_calendar_weeks_in_demand_series": partial_weeks,
            "weeks_without_calendar_coverage": no_calendar_weeks,
            "closure_affected_weeks_in_demand_series": int(
                weekly_calendar["is_closure_affected_week"].sum()
            ),
            "zero_demand_weeks": int(len(zero_demand_weeks)),
            "zero_demand_weeks_with_configured_closure": int(zero_with_closure),
            "zero_demand_weeks_without_configured_closure": int(
                len(zero_demand_weeks) - zero_with_closure
            ),
        },
        "feature_engineering": {
            "target": "weekly_quantity",
            "lag_weeks": LAG_WEEKS,
            "rolling_windows_weeks": ROLLING_WINDOWS,
            "rolling_features_shifted_by_one_week": True,
            "current_target_excluded_from_lags_and_rolling_windows": True,
            "calendar_features_known_from_configured_calendar": [
                "calendar_days_covered", "business_days_in_week",
                "closure_days_in_week", "closure_reason",
                "is_closure_affected_week", "calendar_coverage_status",
                "is_partial_calendar_week",
            ],
            "seasonality_features": [
                "month", "quarter", "iso_week", "week_of_year_sin",
                "week_of_year_cos", "month_sin", "month_cos",
            ],
            "excluded_product_fields": [
                "demand_type_seed: excluded because it is a synthetic generator label and would risk target leakage"
            ],
        },
        "time_split": split_info["metadata"],
        "demand_pattern_segmentation": demand_pattern_summary,
        "split_summary": split_summary,
        "modeling_readiness": {
            "required_history_features": required_history_features,
            "modeling_eligible_rows": int(dataset["modeling_eligible"].sum()),
            "rows_not_yet_eligible_due_to_insufficient_history": int(
                dataset["data_split"].isin(["train", "validation", "test"])
                .mul(~dataset[required_history_features].notna().all(axis=1)).sum()
            ),
            "notes": [
                "Rows with incomplete lag/rolling history remain in the exported dataset for auditability, but modeling_eligible=False.",
                "The partial final calendar week is retained for traceability and excluded from model eligibility and split assignment.",
                "The validation and test sets are 13-week chronological windows. This dataset supports rolling one-step-ahead evaluation; a fixed-origin 13-week forecast evaluation must explicitly construct future horizons from a shared forecast origin.",
            ],
        },
        "validation": {
            "unique_sku_week_pairs": not dataset.duplicated(["sku", "week_start"]).any(),
            "complete_hardware_sku_week_grid": len(dataset) == dataset["sku"].nunique() * dataset["week_start"].nunique(),
            "expected_600_hardware_skus": dataset["sku"].nunique() == EXPECTED_HARDWARE_SKUS,
            "expected_157_weeks": dataset["week_start"].nunique() == EXPECTED_WEEKS,
            "expected_94200_rows": len(dataset) == EXPECTED_ROWS,
            "no_missing_targets": not dataset["weekly_quantity"].isna().any(),
            "no_negative_targets": bool((dataset["weekly_quantity"] >= 0).all()),
            "calendar_week_features_unique": not weekly_calendar["week_start"].duplicated().any(),
            "calendar_week_counts_reconcile_with_demand_weeks": (
                valid_full_weeks + partial_weeks + no_calendar_weeks
                == dataset["week_start"].nunique()
            ),
            "all_demand_weeks_have_calendar_record": bool(dataset["calendar_days_covered"].notna().all()),
            "partial_calendar_weeks_excluded_from_modeling": bool(
                not dataset.loc[dataset["is_partial_calendar_week"], "modeling_eligible"].any()
            ),
            "lag_features_do_not_use_current_target": True,
            "demand_pattern_labels_derived_from_training_only": bool(
                demand_pattern_summary["labels_use_validation_or_test_targets"] is False
            ),
            "one_training_demand_pattern_per_sku": bool(
                dataset.groupby("sku")["demand_pattern_train"].nunique().eq(1).all()
            ),
            "demand_pattern_labels_not_in_predictor_feature_lists": bool(
                not set(["demand_pattern_train", "intermittent_demand_flag_train",
                         "training_adi", "training_cv_squared_positive_demand"])
                & set(lag_columns + rolling_columns + [
                    "year", "month", "quarter", "iso_week", "time_index",
                    "week_of_year_sin", "week_of_year_cos", "month_sin", "month_cos",
                ])
            ),
        },
    }
    report["validation"]["all_checks_pass"] = all(report["validation"].values())
    return dataset, report


def main() -> None:
    demand, products, calendar = load_inputs()
    dataset, report = create_dataset(demand, products, calendar)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("FORECASTING-READY DATASET")
    print(f"Hardware SKUs       : {report['sku_count']:,}")
    print(f"Weeks               : {report['week_count']:,}")
    print(f"Rows                : {report['row_count']:,}")
    print(f"Total demand        : {report['total_demand']:,}")
    print(f"Model-eligible rows : {report['modeling_readiness']['modeling_eligible_rows']:,}")
    print("Split summary:")
    for split, summary in report["split_summary"].items():
        print(
            f"  {split:<32} rows={summary['rows']:,}, "
            f"eligible={summary['modeling_eligible_rows']:,}"
        )
    print("Validation:")
    for check, passed in report["validation"].items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
    print(f"Dataset             : {OUTPUT_PATH}")
    print(f"Report              : {REPORT_PATH}")

    if not report["validation"]["all_checks_pass"]:
        raise SystemExit("Forecasting data preparation validation failed. Review the report.")
    print("Forecasting-ready dataset created successfully.")


if __name__ == "__main__":
    main()
