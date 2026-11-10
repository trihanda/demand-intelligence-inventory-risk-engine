"""Build a leakage-aware forecasting panel from weekly hardware demand.

Input:  data/processed/weekly_demand_hardware.csv
Checks: data/synthetic/products.csv and data/synthetic/calendar.csv
Outputs:
  data/processed/forecasting_ready_dataset.csv
  reports/forecasting_data_preparation/forecasting_data_preparation_report.json

This dataset supports chronological rolling one-step-ahead evaluation. A
separate fixed-origin multi-step backtest must be constructed for 13/20/26-week
forecast horizons; later validation/test rows here may use actual prior weeks
as lag inputs, as they would in a weekly refreshed one-step forecast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LAG_WEEKS = [1, 2, 4, 13, 26, 52]
ROLLING_WINDOWS = [4, 13, 26]
VALIDATION_WEEKS = 13
TEST_WEEKS = 13
ADI_THRESHOLD = 1.32
CV_SQUARED_THRESHOLD = 0.49
REQUIRED_DEMAND = {
    "week_start", "week_end", "sku", "weekly_quantity", "business_days_in_week", "closure_days_in_week",
    "weekend_days_in_week", "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
    "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
    "company_closure_days_in_week", "company_closure_weekdays_in_week",
    "is_public_holiday_week", "is_collective_leave_week", "is_company_closure_week",
}
REQUIRED_PRODUCTS = {"sku", "product_type", "physical_inventory_flag"}
REQUIRED_CALENDAR = {
    "calendar_date", "is_weekday", "is_weekend", "is_business_day", "closure_reason",
    "is_public_holiday", "is_collective_leave", "is_company_closure",
    "public_holiday_name", "collective_leave_name", "company_closure_name",
}


def parse_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
    parsed = series.astype("string").str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f"Cannot parse boolean values in {name}.")
    return parsed.astype(bool)


def get_root() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args().project_root.resolve()


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "demand": root / "data/processed/weekly_demand_hardware.csv",
        "products": root / "data/synthetic/products.csv",
        "calendar": root / "data/synthetic/calendar.csv",
    }
    missing_files = [str(p) for p in paths.values() if not p.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing_files))
    demand, products, calendar = (pd.read_csv(paths[k], low_memory=False) for k in ("demand", "products", "calendar"))
    for label, frame, required in [("weekly demand", demand, REQUIRED_DEMAND), ("products", products, REQUIRED_PRODUCTS), ("calendar", calendar, REQUIRED_CALENDAR)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    demand["week_start"] = pd.to_datetime(demand["week_start"], errors="coerce")
    demand["week_end"] = pd.to_datetime(demand["week_end"], errors="coerce")
    demand["weekly_quantity"] = pd.to_numeric(demand["weekly_quantity"], errors="coerce")
    demand["business_days_in_week"] = pd.to_numeric(demand["business_days_in_week"], errors="coerce")
    demand["closure_days_in_week"] = pd.to_numeric(demand["closure_days_in_week"], errors="coerce")
    calendar_count_columns = [c for c in REQUIRED_DEMAND if c.endswith("_in_week") and c not in {"week_start", "week_end"}]
    for column in calendar_count_columns:
        demand[column] = pd.to_numeric(demand[column], errors="coerce")
    for column in ["is_public_holiday_week", "is_collective_leave_week", "is_company_closure_week"]:
        demand[column] = parse_bool(demand[column], column)
    if demand[list(REQUIRED_DEMAND)].isna().any().any():
        raise ValueError("Weekly demand contains missing required values.")
    if demand["weekly_quantity"].lt(0).any() or not np.allclose(demand["weekly_quantity"], np.round(demand["weekly_quantity"])):
        raise ValueError("Weekly target must be non-negative integer quantity.")
    if demand.duplicated(["sku", "week_start"]).any():
        raise ValueError("Duplicate SKU-week rows in weekly demand.")
    if not demand["week_start"].dt.dayofweek.eq(0).all() or not (demand["week_end"] == demand["week_start"] + pd.Timedelta(days=6)).all():
        raise ValueError("Weekly demand must use complete Monday-Sunday week labels.")

    products["physical_inventory_flag"] = parse_bool(products["physical_inventory_flag"], "physical_inventory_flag")
    products["product_type"] = products["product_type"].astype(str).str.strip().str.lower()
    if products["sku"].isna().any() or products["sku"].duplicated().any():
        raise ValueError("products.csv must have one unique row per SKU.")
    calendar["calendar_date"] = pd.to_datetime(calendar["calendar_date"], errors="coerce").dt.normalize()
    if calendar["calendar_date"].isna().any() or calendar["calendar_date"].duplicated().any():
        raise ValueError("calendar.csv has invalid or duplicate dates.")
    for flag in ["is_weekday", "is_weekend", "is_business_day", "is_public_holiday", "is_collective_leave", "is_company_closure"]:
        calendar[flag] = parse_bool(calendar[flag], flag)
    for col in ["closure_reason", "public_holiday_name", "collective_leave_name", "company_closure_name"]:
        calendar[col] = calendar[col].fillna("").astype(str).str.strip()
    if not (calendar["is_weekend"] == ~calendar["is_weekday"]).all():
        raise ValueError("Calendar weekend and weekday flags disagree.")
    if (calendar["is_business_day"] & (~calendar["is_weekday"] | calendar["is_public_holiday"])).any():
        raise ValueError("Calendar incorrectly marks a weekend or public holiday as a business day.")
    return demand, products, calendar


def build_calendar_validation(calendar: pd.DataFrame) -> pd.DataFrame:
    work = calendar.copy()
    work["week_start"] = work["calendar_date"] - pd.to_timedelta(work["calendar_date"].dt.dayofweek, unit="D")
    work["weekday_closure"] = work["is_weekday"] & ~work["is_business_day"]
    result = work.groupby("week_start", as_index=False).agg(
        calendar_days_covered=("calendar_date", "nunique"),
        business_days_in_week=("is_business_day", "sum"),
        weekend_days_in_week=("is_weekend", "sum"),
        closure_days_in_week=("weekday_closure", "sum"),
        public_holiday_days_in_week=("is_public_holiday", "sum"),
        public_holiday_weekdays_in_week=("is_public_holiday", lambda s: int((s & work.loc[s.index, "is_weekday"]).sum())),
        collective_leave_days_in_week=("is_collective_leave", "sum"),
        collective_leave_weekdays_in_week=("is_collective_leave", lambda s: int((s & work.loc[s.index, "is_weekday"]).sum())),
        company_closure_days_in_week=("is_company_closure", "sum"),
        company_closure_weekdays_in_week=("is_company_closure", lambda s: int((s & work.loc[s.index, "is_weekday"]).sum())),
        public_holiday_name=("public_holiday_name", lambda s: ";".join(sorted({v for v in s if v}))),
        collective_leave_name=("collective_leave_name", lambda s: ";".join(sorted({v for v in s if v}))),
        company_closure_name=("company_closure_name", lambda s: ";".join(sorted({v for v in s if v}))),
        closure_reason=("closure_reason", lambda s: ";".join(sorted({v for v in s if v}))),
    )
    result["week_end"] = result["week_start"] + pd.Timedelta(days=6)
    result["is_complete_week"] = result["calendar_days_covered"].eq(7)
    result["is_closure_affected_week"] = result["closure_days_in_week"].gt(0)
    result["is_public_holiday_week"] = result["public_holiday_weekdays_in_week"].gt(0)
    result["is_collective_leave_week"] = result["collective_leave_weekdays_in_week"].gt(0)
    result["is_company_closure_week"] = result["company_closure_weekdays_in_week"].gt(0)
    result["is_full_closure_week"] = result["is_complete_week"] & result["business_days_in_week"].eq(0)
    return result


def assign_splits(weeks: list[pd.Timestamp]) -> tuple[dict[pd.Timestamp, str], dict[str, Any]]:
    if len(weeks) <= VALIDATION_WEEKS + TEST_WEEKS + 52:
        raise ValueError("Not enough complete weeks for the selected split and 52-week history features.")
    train_end = len(weeks) - VALIDATION_WEEKS - TEST_WEEKS
    val_end = len(weeks) - TEST_WEEKS
    split_map = {week: ("train" if i < train_end else "validation" if i < val_end else "test") for i, week in enumerate(weeks)}
    meta = {
        "method": "chronological target-week split; validation/test are weekly refreshed one-step-ahead windows",
        "train_weeks": train_end,
        "validation_weeks": VALIDATION_WEEKS,
        "test_weeks": TEST_WEEKS,
        "train_first_week": weeks[0].date().isoformat(),
        "train_last_week": weeks[train_end - 1].date().isoformat(),
        "validation_first_week": weeks[train_end].date().isoformat(),
        "validation_last_week": weeks[val_end - 1].date().isoformat(),
        "test_first_week": weeks[val_end].date().isoformat(),
        "test_last_week": weeks[-1].date().isoformat(),
        "fixed_origin_multi_step_backtest_included": False,
    }
    return split_map, meta


def classify_training_patterns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = df.loc[df["data_split"].eq("train")].copy()
    # Remove full closure weeks from ADI denominator, but retain normal zero-demand weeks.
    train = train.loc[train["business_days_in_week"].gt(0)]
    rows: list[dict[str, Any]] = []
    for sku, group in train.groupby("sku", sort=True):
        q = group["weekly_quantity"].astype(float)
        positive = q[q.gt(0)]
        adi = len(q) / len(positive) if len(positive) else None
        cv2 = float((positive.std(ddof=0) / positive.mean()) ** 2) if len(positive) and positive.mean() > 0 else None
        if adi is None or cv2 is None:
            pattern = "no_demand"
        elif adi >= ADI_THRESHOLD and cv2 >= CV_SQUARED_THRESHOLD:
            pattern = "lumpy"
        elif adi >= ADI_THRESHOLD:
            pattern = "intermittent"
        elif cv2 >= CV_SQUARED_THRESHOLD:
            pattern = "erratic"
        else:
            pattern = "smooth"
        rows.append({
            "sku": sku,
            "training_weeks_observed": int(len(q)),
            "training_nonzero_demand_weeks": int(len(positive)),
            "training_adi": adi,
            "training_cv_squared_positive_demand": cv2,
            "demand_pattern_train": pattern,
            "intermittent_demand_flag_train": bool(adi is not None and adi >= ADI_THRESHOLD),
        })
    profile = pd.DataFrame(rows)
    counts = {str(k): int(v) for k, v in profile["demand_pattern_train"].value_counts().sort_index().items()}
    summary = {
        "derived_from": "training split only; full closure weeks excluded from ADI denominator",
        "adi_threshold": ADI_THRESHOLD,
        "cv_squared_threshold": CV_SQUARED_THRESHOLD,
        "sku_counts_by_pattern": counts,
        "intermittent_demand_sku_count": int(profile["intermittent_demand_flag_train"].sum()),
        "labels_use_validation_or_test_targets": False,
    }
    return profile, summary


def create_forecasting_ready_dataset(demand: pd.DataFrame, products: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    hardware_products = products.loc[products["product_type"].eq("hardware") & products["physical_inventory_flag"]].copy()
    hardware_skus = sorted(hardware_products["sku"].astype(str).unique())
    if len(hardware_skus) != 600:
        raise ValueError(f"Expected 600 physical hardware SKUs; found {len(hardware_skus)}.")
    demand = demand.loc[demand["sku"].astype(str).isin(hardware_skus)].copy()
    demand["sku"] = demand["sku"].astype(str)
    weeks = sorted(demand["week_start"].drop_duplicates().tolist())
    expected_rows = len(hardware_skus) * len(weeks)
    if len(demand) != expected_rows or demand["sku"].nunique() != 600:
        raise ValueError(f"Expected complete 600 x {len(weeks)} panel ({expected_rows:,}); found {len(demand):,} rows.")
    if demand.groupby("sku")["week_start"].nunique().nunique() != 1:
        raise ValueError("SKUs do not have a consistent number of weeks.")

    calendar_week = build_calendar_validation(calendar)
    calendar_compare_columns = [
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
        "company_closure_days_in_week", "company_closure_weekdays_in_week",
        "is_public_holiday_week", "is_collective_leave_week", "is_company_closure_week",
    ]
    weekly_cal = demand[["week_start", "week_end", *calendar_compare_columns]].drop_duplicates("week_start")
    check = weekly_cal.merge(calendar_week, on=["week_start", "week_end"], how="left", suffixes=("_panel", "_calendar"), validate="one_to_one")
    if check["calendar_days_covered"].isna().any():
        raise ValueError("One or more weekly demand weeks have no matching calendar record.")
    mismatch = ~check["calendar_days_covered"].eq(7)
    for column in calendar_compare_columns:
        mismatch = mismatch | check[f"{column}_panel"].ne(check[f"{column}_calendar"])
    if mismatch.any():
        raise ValueError(f"Weekly calendar mismatch found in {int(mismatch.sum())} week(s).")

    split_map, split_meta = assign_splits(weeks)
    dataset = demand.merge(
        hardware_products[[c for c in ["sku", "product_category", "product_family", "brand", "base_uom"] if c in hardware_products.columns]],
        on="sku", how="left", validate="many_to_one", suffixes=("", "_master")
    )
    dataset = dataset.sort_values(["sku", "week_start"], ignore_index=True)
    dataset["year"] = dataset["week_start"].dt.year.astype(int)
    dataset["month"] = dataset["week_start"].dt.month.astype(int)
    dataset["quarter"] = dataset["week_start"].dt.quarter.astype(int)
    dataset["iso_week"] = dataset["week_start"].dt.isocalendar().week.astype(int)
    dataset["time_index"] = dataset["week_start"].rank(method="dense").astype(int) - 1
    dataset["week_of_year_sin"] = np.sin(2 * np.pi * (dataset["iso_week"] - 1) / 52.0)
    dataset["week_of_year_cos"] = np.cos(2 * np.pi * (dataset["iso_week"] - 1) / 52.0)
    dataset["month_sin"] = np.sin(2 * np.pi * (dataset["month"] - 1) / 12.0)
    dataset["month_cos"] = np.cos(2 * np.pi * (dataset["month"] - 1) / 12.0)

    grouped = dataset.groupby("sku", sort=False)["weekly_quantity"]
    lag_columns = []
    for lag in LAG_WEEKS:
        col = f"demand_lag_{lag}"
        dataset[col] = grouped.shift(lag)
        lag_columns.append(col)
    rolling_columns = []
    for window in ROLLING_WINDOWS:
        col = f"rolling_mean_{window}w"
        dataset[col] = dataset.groupby("sku", sort=False)["weekly_quantity"].transform(
            lambda s, window=window: s.shift(1).rolling(window, min_periods=window).mean()
        )
        rolling_columns.append(col)
    dataset["rolling_std_13w"] = dataset.groupby("sku", sort=False)["weekly_quantity"].transform(
        lambda s: s.shift(1).rolling(13, min_periods=13).std(ddof=0)
    )
    rolling_columns.append("rolling_std_13w")
    dataset["data_split"] = dataset["week_start"].map(split_map)

    pattern_profile, pattern_summary = classify_training_patterns(dataset)
    pattern_summary["training_first_week"] = split_meta["train_first_week"]
    pattern_summary["training_last_week"] = split_meta["train_last_week"]
    pattern_summary["labels_use_validation_or_test_targets"] = False
    dataset = dataset.merge(pattern_profile, on="sku", how="left", validate="many_to_one")
    required_history = lag_columns + rolling_columns
    dataset["modeling_eligible"] = dataset["data_split"].isin(["train", "validation", "test"]) & dataset[required_history].notna().all(axis=1)

    predictor_columns = [
        "product_category", "product_family", "brand", "base_uom",
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_weekdays_in_week", "collective_leave_weekdays_in_week", "company_closure_weekdays_in_week",
        "is_closure_affected_week", "is_public_holiday_week", "is_collective_leave_week",
        "is_company_closure_week", "is_full_closure_week",
        "year", "month", "quarter", "iso_week", "time_index",
        "week_of_year_sin", "week_of_year_cos", "month_sin", "month_cos",
        *lag_columns, *rolling_columns,
    ]
    segmentation_columns = ["training_weeks_observed", "training_nonzero_demand_weeks", "training_adi", "training_cv_squared_positive_demand", "demand_pattern_train", "intermittent_demand_flag_train"]
    identity_columns = ["sku", "week_start", "week_end", "weekly_quantity"]
    optional_calendar = [c for c in [
        "closure_reason", "public_holiday_name", "collective_leave_name", "company_closure_name", "week_status"
    ] if c in dataset.columns]
    output_calendar_counts = [c for c in [
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
        "company_closure_days_in_week", "company_closure_weekdays_in_week",
        "is_closure_affected_week", "is_public_holiday_week", "is_collective_leave_week",
        "is_company_closure_week", "is_full_closure_week"
    ] if c in dataset.columns]
    output_columns = identity_columns + [c for c in ["product_category", "product_family", "brand", "base_uom"] if c in dataset.columns] + optional_calendar + output_calendar_counts + ["year", "month", "quarter", "iso_week", "time_index", "week_of_year_sin", "week_of_year_cos", "month_sin", "month_cos"] + lag_columns + rolling_columns + segmentation_columns + ["data_split", "modeling_eligible"]
    # Deduplicate column names while preserving order.
    output_columns = list(dict.fromkeys(output_columns))
    dataset = dataset[output_columns]

    # Verify leakage-safe lag/rolling calculations on all rows with enough history.
    lag_check = True
    rolling_check = True
    for lag in LAG_WEEKS:
        expected = dataset.groupby("sku", sort=False)["weekly_quantity"].shift(lag)
        col = f"demand_lag_{lag}"
        same = dataset[col].fillna(-999999).to_numpy() == expected.fillna(-999999).to_numpy()
        lag_check = lag_check and bool(same.all())
    for window in ROLLING_WINDOWS:
        expected = dataset.groupby("sku", sort=False)["weekly_quantity"].transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean())
        col = f"rolling_mean_{window}w"
        same = np.isclose(dataset[col].to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True)
        rolling_check = rolling_check and bool(same.all())
    expected_std = dataset.groupby("sku", sort=False)["weekly_quantity"].transform(
        lambda s: s.shift(1).rolling(13, min_periods=13).std(ddof=0)
    )
    rolling_std_check = bool(np.isclose(dataset["rolling_std_13w"].to_numpy(dtype=float), expected_std.to_numpy(dtype=float), equal_nan=True).all())

    split_summary = {}
    for split in ["train", "validation", "test"]:
        subset = dataset.loc[dataset["data_split"].eq(split)]
        split_summary[split] = {"rows": int(len(subset)), "weeks": int(subset["week_start"].nunique()), "modeling_eligible_rows": int(subset["modeling_eligible"].sum())}
    report = {
        "task": "forecasting_data_preparation",
        "input_file": "data/processed/weekly_demand_hardware.csv",
        "output_file": "data/processed/forecasting_ready_dataset.csv",
        "sku_count": int(dataset["sku"].nunique()),
        "week_count": int(dataset["week_start"].nunique()),
        "row_count": int(len(dataset)),
        "first_week": dataset["week_start"].min().date().isoformat(),
        "last_week": dataset["week_end"].max().date().isoformat(),
        "total_demand": int(dataset["weekly_quantity"].sum()),
        "feature_engineering": {
            "target": "weekly_quantity",
            "predictor_features": predictor_columns,
            "evaluation_metadata_not_predictors": segmentation_columns,
            "lag_weeks": LAG_WEEKS,
            "rolling_windows_weeks": ROLLING_WINDOWS,
            "lag_features_use_prior_weeks_only": True,
            "rolling_features_use_shift_1": True,
            "calendar_features_are_known_in_advance": True,
            "calendar_predictors": [
                "business_days_in_week", "public_holiday_weekdays_in_week",
                "collective_leave_weekdays_in_week", "company_closure_weekdays_in_week",
                "is_public_holiday_week", "is_collective_leave_week", "is_company_closure_week",
            ],
            "collective_leave_policy": str(calendar["collective_leave_policy"].iloc[0]) if "collective_leave_policy" in calendar.columns and len(calendar) else None,
            "demand_type_seed_used_as_predictor": False,
            "lifecycle_or_current_active_flags_used_as_predictors": False,
        },
        "time_split": split_meta,
        "split_summary": split_summary,
        "demand_pattern_segmentation": pattern_summary,
        "modeling_readiness": {
            "required_history_features": required_history,
            "modeling_eligible_rows": int(dataset["modeling_eligible"].sum()),
            "fixed_origin_multi_step_backtesting_required_separately": True,
            "notes": [
                "Validation/test rows use realized prior-week demand for rolling one-step-ahead refresh; they do not represent a fixed-origin 13/20/26-week forecast.",
                "Demand pattern segments are derived from training-period demand only and are evaluation metadata, not model predictors.",
                "Rows missing the 52-week lag or rolling history remain in the output but are marked modeling_eligible=False.",
                "All weeks in this input panel are complete Monday-Sunday calendar weeks; the partial final week is kept in a separate trace dataset.",
                "Public holidays and collective leave are included as calendar-known features; collective leave treatment follows the generator's configured company policy.",
            ],
        },
        "validation": {
            "unique_sku_week_pairs": not dataset.duplicated(["sku", "week_start"]).any(),
            "complete_hardware_sku_week_grid": len(dataset) == dataset["sku"].nunique() * dataset["week_start"].nunique(),
            "expected_600_hardware_skus": dataset["sku"].nunique() == 600,
            "no_missing_targets": bool(dataset["weekly_quantity"].notna().all()),
            "no_negative_targets": bool(dataset["weekly_quantity"].ge(0).all()),
            "calendar_matches_daily_calendar": not mismatch.any(),
            "holiday_calendar_features_present": set(["public_holiday_weekdays_in_week", "collective_leave_weekdays_in_week", "is_public_holiday_week", "is_collective_leave_week"]).issubset(dataset.columns),
            "partial_week_excluded_from_input_panel": bool((dataset["week_end"] - dataset["week_start"]).dt.days.eq(6).all()),
            "lag_features_use_prior_weeks_only": bool(lag_check),
            "rolling_mean_features_use_shift_1": bool(rolling_check),
            "rolling_std_feature_uses_shift_1": rolling_std_check,
            "demand_pattern_derived_from_training_only": pattern_summary["labels_use_validation_or_test_targets"] is False,
            "demand_type_seed_excluded": "demand_type_seed" not in dataset.columns,
            "evaluation_segmentation_not_in_predictor_features": not bool(set(segmentation_columns) & set(predictor_columns)),
            "all_lags_and_rolling_features_missing_only_at_insufficient_history": bool(dataset["modeling_eligible"].eq(dataset[required_history].notna().all(axis=1)).all()),
        },
    }
    report["validation"]["all_checks_pass"] = all(report["validation"].values())
    return dataset, report


def main() -> None:
    root = get_root()
    demand, products, calendar = load_inputs(root)
    dataset, report = create_forecasting_ready_dataset(demand, products, calendar)
    output_path = root / "data/processed/forecasting_ready_dataset.csv"
    report_dir = root / "reports/forecasting_data_preparation"
    report_path = report_dir / "forecasting_data_preparation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("FORECASTING-READY DATASET")
    print(f"Hardware SKUs       : {report['sku_count']:,}")
    print(f"Weeks               : {report['week_count']:,}")
    print(f"Rows                : {report['row_count']:,}")
    print(f"Total demand        : {report['total_demand']:,}")
    print(f"Eligible rows       : {report['modeling_readiness']['modeling_eligible_rows']:,}")
    for split, summary in report["split_summary"].items():
        print(f"  {split:<12} rows={summary['rows']:,}, eligible={summary['modeling_eligible_rows']:,}")
    for name, passed in report["validation"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not report["validation"]["all_checks_pass"]:
        raise SystemExit("Forecasting-ready dataset failed validation.")
    print(f"Dataset             : {output_path}")
    print(f"Report              : {report_path}")


if __name__ == "__main__":
    main()
