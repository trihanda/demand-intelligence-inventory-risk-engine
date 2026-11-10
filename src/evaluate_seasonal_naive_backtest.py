"""Fixed-origin 13-week backtesting for a Seasonal Naive (52-week) baseline.

Run from the repository root:
    python src/evaluate_seasonal_naive_backtest.py

Input:
    data/processed/forecasting_ready_dataset.csv

Outputs:
    data/processed/seasonal_naive_13w_predictions.csv
    reports/forecasting_backtesting/seasonal_naive_13w_backtest_report.json

Development folds use expanding history and remain inside the train split.
Validation forecasts from the end of train; test forecasts from the end of
validation. Actual demand inside each 13-week forecast horizon is never used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "forecasting_ready_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "seasonal_naive_13w_predictions.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "forecasting_backtesting" / "seasonal_naive_13w_backtest_report.json"

HORIZON_WEEKS = 13
SEASONAL_LAG_WEEKS = 52
MIN_TRAIN_WEEKS = 104
DEVELOPMENT_STEP_WEEKS = 13
REQUIRED_COLUMNS = {
    "sku", "week_start", "week_end", "weekly_quantity", "data_split",
    "demand_pattern_train", "is_full_closure_week",
}


def fail(message: str) -> None:
    raise ValueError(message)


def calculate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    actual = frame["actual"].astype(float).to_numpy()
    forecast = frame["forecast"].astype(float).to_numpy()
    error = forecast - actual
    total_actual = float(actual.sum())
    return {
        "forecast_rows": int(len(frame)),
        "total_actual": total_actual,
        "total_forecast": float(forecast.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape_pct": float(np.abs(error).sum() / total_actual * 100) if total_actual > 0 else None,
        "bias_pct": float(error.sum() / total_actual * 100) if total_actual > 0 else None,
    }


def main() -> None:
    if not INPUT_FILE.exists():
        fail(f"Input file not found: {INPUT_FILE}. Run forecasting data preparation first.")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        fail(f"Missing required columns: {missing}")

    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    if df[["week_start", "week_end"]].isna().any().any():
        fail("Invalid or missing week_start/week_end values found.")
    df["weekly_quantity"] = pd.to_numeric(df["weekly_quantity"], errors="coerce")
    if df["weekly_quantity"].isna().any() or (df["weekly_quantity"] < 0).any():
        fail("weekly_quantity contains missing, non-numeric, or negative values.")

    df["data_split"] = df["data_split"].astype(str).str.strip().str.lower()
    allowed_splits = {"train", "validation", "test"}
    unknown = sorted(set(df["data_split"]) - allowed_splits)
    if unknown:
        fail(f"Unexpected data_split values: {unknown}")
    if df.duplicated(["sku", "week_start"]).any():
        fail("Duplicate SKU-week pairs found.")

    all_weeks = pd.DatetimeIndex(sorted(df["week_start"].unique()))
    if len(all_weeks) < MIN_TRAIN_WEEKS + 2 * HORIZON_WEEKS:
        fail(f"Not enough history: found {len(all_weeks)} weeks.")
    expected_weeks = pd.date_range(all_weeks[0], all_weeks[-1], freq="7D")
    if not all_weeks.equals(pd.DatetimeIndex(expected_weeks)):
        fail("Weekly calendar is not a complete continuous 7-day sequence.")

    sku_values = sorted(df["sku"].astype(str).unique())
    sku_count = len(sku_values)
    expected_rows = sku_count * len(all_weeks)
    if len(df) != expected_rows:
        fail(f"Incomplete SKU-week grid: {len(df):,} rows; expected {expected_rows:,}.")

    week_to_idx = {week: idx for idx, week in enumerate(all_weeks)}
    df["week_idx"] = df["week_start"].map(week_to_idx).astype(int)
    split_week = df[["week_idx", "week_start", "data_split"]].drop_duplicates()
    if split_week.groupby("week_idx")["data_split"].nunique().gt(1).any():
        fail("A calendar week is assigned to more than one data_split.")
    split_week = split_week.sort_values("week_idx")
    split_indices = {
        name: split_week.loc[split_week["data_split"] == name, "week_idx"].tolist()
        for name in ("train", "validation", "test")
    }
    for name, indices in split_indices.items():
        if not indices:
            fail(f"No weeks found for split '{name}'.")
        if indices != list(range(indices[0], indices[-1] + 1)):
            fail(f"'{name}' weeks are not contiguous.")

    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    if train_idx[-1] + 1 != val_idx[0] or val_idx[-1] + 1 != test_idx[0]:
        fail("Expected contiguous chronological train/validation/test splits.")
    if len(val_idx) != HORIZON_WEEKS or len(test_idx) != HORIZON_WEEKS:
        fail(f"Expected 13 validation and 13 test weeks; got {len(val_idx)} and {len(test_idx)}.")

    # Fast lookups: one actual demand value per SKU and weekly index.
    demand = df.set_index(["sku", "week_idx"])["weekly_quantity"]
    sku_patterns = (
        df[["sku", "demand_pattern_train"]].drop_duplicates("sku")
        .set_index("sku")["demand_pattern_train"].fillna("unclassified").astype(str)
    )
    calendar_cols = [c for c in [
        "business_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_weekdays_in_week", "company_closure_weekdays_in_week",
        "is_closure_affected_week", "is_public_holiday_week",
        "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week",
    ] if c in df.columns]
    calendar_lookup = df[["week_idx"] + calendar_cols].drop_duplicates("week_idx").set_index("week_idx")

    folds: list[dict[str, Any]] = []
    # First development target is after 104 complete training weeks.
    target_start = train_idx[0] + MIN_TRAIN_WEEKS
    last_dev_start = train_idx[-1] - HORIZON_WEEKS + 1
    fold_num = 1
    while target_start <= last_dev_start:
        folds.append({
            "phase": "development", "fold_id": f"DEV_{fold_num:02d}",
            "origin_idx": target_start - 1,
            "target_indices": list(range(target_start, target_start + HORIZON_WEEKS)),
        })
        target_start += DEVELOPMENT_STEP_WEEKS
        fold_num += 1
    if not folds:
        fail("No development folds possible with the configured minimum history.")

    folds.append({"phase": "validation", "fold_id": "VALIDATION_13W",
                  "origin_idx": train_idx[-1], "target_indices": val_idx})
    folds.append({"phase": "test", "fold_id": "TEST_13W",
                  "origin_idx": val_idx[-1], "target_indices": test_idx})

    prediction_rows: list[dict[str, Any]] = []
    fold_plan: list[dict[str, Any]] = []
    for fold in folds:
        origin_idx = fold["origin_idx"]
        targets = fold["target_indices"]
        if len(targets) != HORIZON_WEEKS or targets[0] != origin_idx + 1:
            fail(f"Invalid origin/horizon setup for {fold['fold_id']}.")
        sources = [target_idx - SEASONAL_LAG_WEEKS for target_idx in targets]
        if min(sources) < 0 or max(sources) > origin_idx:
            fail(f"Insufficient history or seasonal-source leakage in {fold['fold_id']}.")

        origin_start = all_weeks[origin_idx]
        fold_plan.append({
            "phase": fold["phase"], "fold_id": fold["fold_id"],
            "origin_week_start": origin_start.strftime("%Y-%m-%d"),
            "origin_week_end": (origin_start + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
            "target_first_week": all_weeks[targets[0]].strftime("%Y-%m-%d"),
            "target_last_week": all_weeks[targets[-1]].strftime("%Y-%m-%d"),
            "horizon_weeks": HORIZON_WEEKS,
            "forecast_uses_actuals_inside_horizon": False,
        })

        for horizon, (target_idx, source_idx) in enumerate(zip(targets, sources), start=1):
            for sku in sku_values:
                actual = int(demand.loc[(sku, target_idx)])
                forecast = int(demand.loc[(sku, source_idx)])
                row = {
                    "phase": fold["phase"], "fold_id": fold["fold_id"], "sku": sku,
                    "evaluation_demand_segment": (
                        "not_used_for_development"
                        if fold["phase"] == "development"
                        else sku_patterns.loc[sku]
                    ),
                    "origin_week_start": origin_start.strftime("%Y-%m-%d"),
                    "origin_week_end": (origin_start + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
                    "target_week_start": all_weeks[target_idx].strftime("%Y-%m-%d"),
                    "horizon_week": horizon,
                    "seasonal_source_week_start": all_weeks[source_idx].strftime("%Y-%m-%d"),
                    "actual": actual, "forecast": forecast,
                    "error": forecast - actual, "absolute_error": abs(forecast - actual),
                }
                for col in calendar_cols:
                    row[col] = calendar_lookup.loc[target_idx, col]
                prediction_rows.append(row)

    pred = pd.DataFrame(prediction_rows)
    metrics: dict[str, Any] = {"overall": calculate_metrics(pred), "by_phase": {},
                               "by_phase_and_horizon": {}, "by_phase_and_demand_pattern": {}}
    for phase, phase_df in pred.groupby("phase", sort=False):
        metrics["by_phase"][phase] = calculate_metrics(phase_df)
        metrics["by_phase_and_horizon"][phase] = {
            str(int(h)): calculate_metrics(group)
            for h, group in phase_df.groupby("horizon_week", sort=True)
        }
        metrics["by_phase_and_demand_pattern"][phase] = {
            str(pattern): calculate_metrics(group)
            for pattern, group in phase_df.groupby("evaluation_demand_segment", sort=True)
        }

    target_dates = pd.to_datetime(pred["target_week_start"])
    source_dates = pd.to_datetime(pred["seasonal_source_week_start"])
    origin_dates = pd.to_datetime(pred["origin_week_start"])
    checks = {
        "unique_sku_target_origin_fold": not pred.duplicated(["phase", "fold_id", "sku", "target_week_start"]).any(),
        "all_folds_have_13_week_horizon": all(len(f["target_indices"]) == HORIZON_WEEKS for f in folds),
        "seasonal_source_is_52_weeks_before_target": ((target_dates - source_dates) == pd.Timedelta(days=364)).all(),
        "seasonal_source_not_after_forecast_origin": (source_dates <= origin_dates).all(),
        "forecast_targets_start_after_origin": (target_dates > origin_dates).all(),
        "validation_targets_match_validation_split": set(pred.loc[pred.phase == "validation", "target_week_start"]) == set(all_weeks[val_idx].strftime("%Y-%m-%d")),
        "test_targets_match_test_split": set(pred.loc[pred.phase == "test", "target_week_start"]) == set(all_weeks[test_idx].strftime("%Y-%m-%d")),
        "no_negative_forecasts": (pred["forecast"] >= 0).all(),
        "no_missing_actuals_or_forecasts": not pred[["actual", "forecast"]].isna().any().any(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        fail(f"Backtesting validation failed: {[k for k, v in checks.items() if not v]}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUTPUT_FILE, index=False)
    report = {
        "task": "fixed_origin_13_week_seasonal_naive_backtest",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "methodology": {
            "baseline": "Seasonal Naive (52-week lag)",
            "forecast_horizon_weeks": HORIZON_WEEKS,
            "seasonal_lag_weeks": SEASONAL_LAG_WEEKS,
            "development_window": "expanding window; folds remain inside train split",
            "development_minimum_history_weeks": MIN_TRAIN_WEEKS,
            "development_fold_step_weeks": DEVELOPMENT_STEP_WEEKS,
            "validation": "one fixed-origin 13-week forecast from end of train split",
            "test": "one fixed-origin 13-week forecast from end of validation split",
            "within_horizon_actual_demand_used": False,
            "development_segment_labels": "not applied; demand_pattern_train uses the full train split and would leak future information into earlier development folds",
            "forecast_rule": "forecast[t+h] = actual[t+h-52] for h=1..13",
            "interpretation": "Validation origin is the final train week. Test origin is the final validation week. Actual demand within each 13-week target horizon is not used to generate its predictions.",
        },
        "dataset": {
            "sku_count": sku_count, "week_count": len(all_weeks), "row_count": int(len(df)),
            "first_week": all_weeks[0].strftime("%Y-%m-%d"), "last_week": all_weeks[-1].strftime("%Y-%m-%d"),
            "train_weeks": len(train_idx), "validation_weeks": len(val_idx), "test_weeks": len(test_idx),
        },
        "fold_plan": fold_plan,
        "prediction_row_count": int(len(pred)),
        "metrics": metrics,
        "validation": checks,
    }
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== FIXED-ORIGIN 13-WEEK SEASONAL NAIVE BACKTEST ===")
    print(f"SKUs                     : {sku_count:,}")
    print(f"Weeks                    : {len(all_weeks):,}")
    print(f"Development folds        : {sum(f['phase'] == 'development' for f in folds)}")
    print(f"Validation horizon       : {len(val_idx)} weeks")
    print(f"Test horizon             : {len(test_idx)} weeks")
    print(f"Prediction rows          : {len(pred):,}")
    print("\nFold plan:")
    for f in fold_plan:
        print(f"  {f['fold_id']:<16} origin={f['origin_week_end']} target={f['target_first_week']} to {f['target_last_week']}")
    print("\nMetrics by phase:")
    for phase, m in metrics["by_phase"].items():
        print(f"  {phase:<12} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} WAPE={m['wape_pct']:.2f}% Bias={m['bias_pct']:.2f}%")
    print("\nValidation:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nPredictions : {OUTPUT_FILE}")
    print(f"Report      : {REPORT_FILE}")
    print("\nBacktesting completed.")


if __name__ == "__main__":
    main()
