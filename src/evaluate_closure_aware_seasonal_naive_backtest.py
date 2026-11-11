"""Fixed-origin 13-week backtesting for a closure-aware Seasonal Naive baseline.

Run from the repository root:
    python src/evaluate_closure_aware_seasonal_naive_backtest.py

Input:
    data/processed/forecasting_ready_dataset.csv

Outputs:
    data/processed/closure_aware_seasonal_naive_13w_predictions.csv
    reports/forecasting_backtesting/closure_aware_seasonal_naive_13w_backtest_report.json

Rules:
1. If the target week is a known full-closure week, forecast zero.
2. Otherwise, if the 52-week seasonal source is not a full closure, use its SKU demand.
3. If the seasonal source is a full closure, use the mean demand from the nearest
   non-full-closure week(s) within +/- 4 weeks of that source. Prefer candidates
   with the same number of business days as the target; if none exist, choose
   the smallest business-day difference. Ties at the nearest distance are averaged.

All fallback source weeks must be on or before the forecast origin. Actual demand
inside each 13-week forecast horizon is never used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "forecasting_ready_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "closure_aware_seasonal_naive_13w_predictions.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "forecasting_backtesting" / "closure_aware_seasonal_naive_13w_backtest_report.json"

HORIZON_WEEKS = 13
SEASONAL_LAG_WEEKS = 52
MIN_TRAIN_WEEKS = 104
DEVELOPMENT_STEP_WEEKS = 13
FALLBACK_RADIUS_WEEKS = 4
REQUIRED_COLUMNS = {
    "sku", "week_start", "week_end", "weekly_quantity", "data_split",
    "demand_pattern_train", "is_full_closure_week", "business_days_in_week",
}


def fail(message: str) -> None:
    raise ValueError(message)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def choose_fallback_sources(
    source_idx: int,
    origin_idx: int,
    target_business_days: int,
    calendar_lookup: pd.DataFrame,
) -> tuple[list[int], str]:
    """Choose nearest comparable non-full-closure week(s), using past data only."""
    candidates: list[dict[str, int]] = []
    for distance in range(1, FALLBACK_RADIUS_WEEKS + 1):
        for candidate_idx in (source_idx - distance, source_idx + distance):
            if candidate_idx < 0 or candidate_idx > origin_idx:
                continue
            candidate = calendar_lookup.loc[candidate_idx]
            if as_bool(candidate["is_full_closure_week"]):
                continue
            business_days = int(candidate["business_days_in_week"])
            if business_days <= 0:
                continue
            candidates.append({"idx": candidate_idx, "distance": distance, "business_days": business_days})

    if not candidates:
        fail(
            f"No eligible non-full-closure fallback week found within +/-{FALLBACK_RADIUS_WEEKS} "
            f"weeks of seasonal source index {source_idx}, at origin index {origin_idx}."
        )

    exact_matches = [c for c in candidates if c["business_days"] == target_business_days]
    if exact_matches:
        pool = exact_matches
        rule = "nearest_non_full_closure_weeks_matching_target_business_days"
    else:
        min_gap = min(abs(c["business_days"] - target_business_days) for c in candidates)
        pool = [c for c in candidates if abs(c["business_days"] - target_business_days) == min_gap]
        rule = "nearest_non_full_closure_weeks_with_minimum_business_day_gap"

    min_distance = min(c["distance"] for c in pool)
    selected = sorted(c["idx"] for c in pool if c["distance"] == min_distance)
    return selected, rule


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
    df["business_days_in_week"] = pd.to_numeric(df["business_days_in_week"], errors="coerce")
    if df["weekly_quantity"].isna().any() or (df["weekly_quantity"] < 0).any():
        fail("weekly_quantity contains missing, non-numeric, or negative values.")
    if df["business_days_in_week"].isna().any():
        fail("business_days_in_week contains missing or non-numeric values.")
    if not df["business_days_in_week"].between(0, 7).all():
        fail("business_days_in_week must be between 0 and 7.")

    df["data_split"] = df["data_split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["data_split"]) - {"train", "validation", "test"})
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
    calendar_consistency = df.groupby("week_idx")[calendar_cols].nunique(dropna=False)
    if calendar_consistency.gt(1).any().any():
        fail("Calendar features are inconsistent across SKUs for the same week.")
    calendar_week = df[["week_idx"] + calendar_cols].drop_duplicates("week_idx")
    calendar_lookup = calendar_week.set_index("week_idx")
    calendar_lookup["is_full_closure_week"] = calendar_lookup["is_full_closure_week"].map(as_bool)
    calendar_lookup["business_days_in_week"] = calendar_lookup["business_days_in_week"].astype(int)

    folds: list[dict[str, Any]] = []
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
    fallback_target_details: list[dict[str, Any]] = []
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
            target_date = all_weeks[target_idx]
            source_date = all_weeks[source_idx]
            target_calendar = calendar_lookup.loc[target_idx]
            source_calendar = calendar_lookup.loc[source_idx]
            target_is_full_closure = as_bool(target_calendar["is_full_closure_week"])
            source_is_full_closure = as_bool(source_calendar["is_full_closure_week"])
            target_business_days = int(target_calendar["business_days_in_week"])

            fallback_indices: list[int] = []
            fallback_rule: str | None = None
            if target_is_full_closure:
                forecast_method = "zero_due_to_target_full_closure"
            elif source_is_full_closure:
                fallback_indices, fallback_rule = choose_fallback_sources(
                    source_idx, origin_idx, target_business_days, calendar_lookup
                )
                forecast_method = "business_day_matched_fallback_mean" if fallback_rule == "nearest_non_full_closure_weeks_matching_target_business_days" else "nearest_business_day_fallback_mean"
                fallback_target_details.append({
                    "phase": fold["phase"], "fold_id": fold["fold_id"],
                    "target_week_start": target_date.strftime("%Y-%m-%d"),
                    "seasonal_source_week_start": source_date.strftime("%Y-%m-%d"),
                    "target_business_days": target_business_days,
                    "fallback_source_week_starts": [all_weeks[i].strftime("%Y-%m-%d") for i in fallback_indices],
                    "fallback_rule": fallback_rule,
                })
            else:
                forecast_method = "seasonal_naive_52w"

            fallback_dates = [all_weeks[i].strftime("%Y-%m-%d") for i in fallback_indices]
            fallback_business_days = [int(calendar_lookup.loc[i, "business_days_in_week"]) for i in fallback_indices]

            for sku in sku_values:
                actual = float(demand.loc[(sku, target_idx)])
                if forecast_method == "zero_due_to_target_full_closure":
                    forecast = 0.0
                elif forecast_method == "seasonal_naive_52w":
                    forecast = float(demand.loc[(sku, source_idx)])
                else:
                    forecast = float(np.mean([demand.loc[(sku, i)] for i in fallback_indices]))

                row = {
                    "phase": fold["phase"], "fold_id": fold["fold_id"], "sku": sku,
                    "evaluation_demand_segment": (
                        "not_used_for_development" if fold["phase"] == "development" else sku_patterns.loc[sku]
                    ),
                    "origin_week_start": origin_start.strftime("%Y-%m-%d"),
                    "origin_week_end": (origin_start + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
                    "target_week_start": target_date.strftime("%Y-%m-%d"),
                    "horizon_week": horizon,
                    "seasonal_source_week_start": source_date.strftime("%Y-%m-%d"),
                    "source_is_full_closure_week": source_is_full_closure,
                    "target_is_full_closure_week": target_is_full_closure,
                    "forecast_method": forecast_method,
                    "fallback_source_week_starts": ";".join(fallback_dates) if fallback_dates else "none",
                    "fallback_source_count": len(fallback_indices),
                    "fallback_source_business_days_mean": float(np.mean(fallback_business_days)) if fallback_business_days else None,
                    "fallback_rule": fallback_rule or "none",
                    "actual": actual, "forecast": forecast,
                    "error": forecast - actual, "absolute_error": abs(forecast - actual),
                }
                for col in calendar_cols:
                    row[col] = calendar_lookup.loc[target_idx, col]
                prediction_rows.append(row)

    pred = pd.DataFrame(prediction_rows)
    metrics: dict[str, Any] = {
        "overall": calculate_metrics(pred), "by_phase": {}, "by_phase_and_horizon": {},
        "by_phase_and_demand_pattern": {}, "by_phase_and_forecast_method": {},
    }
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
        metrics["by_phase_and_forecast_method"][phase] = {
            str(method): calculate_metrics(group)
            for method, group in phase_df.groupby("forecast_method", sort=True)
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
        "full_closure_targets_have_zero_forecast": (pred.loc[pred["target_is_full_closure_week"], "forecast"] == 0).all(),
        "nonclosure_targets_with_closure_source_use_fallback": pred.loc[
            (~pred["target_is_full_closure_week"]) & pred["source_is_full_closure_week"],
            "forecast_method"
        ].isin(["business_day_matched_fallback_mean", "nearest_business_day_fallback_mean"]).all(),
        "fallback_source_weeks_are_nonclosure": all(
            not as_bool(calendar_lookup.loc[i, "is_full_closure_week"])
            for detail in fallback_target_details
            for date in detail["fallback_source_week_starts"]
            for i in [week_to_idx[pd.Timestamp(date)]]
        ),
        "fallback_business_days_match_target_when_available": all(
            all(int(calendar_lookup.loc[week_to_idx[pd.Timestamp(date)], "business_days_in_week"]) == detail["target_business_days"]
                for date in detail["fallback_source_week_starts"])
            for detail in fallback_target_details
            if detail["fallback_rule"] == "nearest_non_full_closure_weeks_matching_target_business_days"
        ),
    }
    # Explicitly validate every selected fallback source against its fold's origin.
    fold_origin_by_id = {f["fold_id"]: f["origin_idx"] for f in folds}
    checks["fallback_sources_not_after_origin"] = all(
        week_to_idx[pd.Timestamp(date)] <= fold_origin_by_id[detail["fold_id"]]
        for detail in fallback_target_details
        for date in detail["fallback_source_week_starts"]
    )
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        fail(f"Backtesting validation failed: {[k for k, v in checks.items() if not v]}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUTPUT_FILE, index=False)
    method_counts = {
        str(method): int(count)
        for method, count in pred["forecast_method"].value_counts().items()
    }
    report = {
        "task": "fixed_origin_13_week_closure_aware_seasonal_naive_backtest",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "methodology": {
            "baseline": "Closure-aware Seasonal Naive (52-week seasonal reference)",
            "forecast_horizon_weeks": HORIZON_WEEKS,
            "seasonal_lag_weeks": SEASONAL_LAG_WEEKS,
            "development_window": "expanding window; folds remain inside train split",
            "development_minimum_history_weeks": MIN_TRAIN_WEEKS,
            "development_fold_step_weeks": DEVELOPMENT_STEP_WEEKS,
            "validation": "one fixed-origin 13-week forecast from end of train split",
            "test": "one fixed-origin 13-week forecast from end of validation split",
            "within_horizon_actual_demand_used": False,
            "development_segment_labels": "not applied; demand_pattern_train uses the full train split and would leak future information into earlier development folds",
            "forecast_rules": {
                "target_full_closure": "forecast zero because closure is known in advance",
                "normal_target_and_normal_seasonal_source": "use SKU demand from target week minus 52 weeks",
                "normal_target_and_full_closure_seasonal_source": f"average the nearest non-full-closure week(s) within +/-{FALLBACK_RADIUS_WEEKS} weeks of the seasonal source; prefer same business_days_in_week as target; average ties",
                "fallback_history_constraint": "all selected fallback weeks must be on or before forecast origin",
            },
            "interpretation": "Forecast origin is fixed for each 13-week horizon. Actual demand within each target horizon is not used to generate predictions.",
        },
        "dataset": {
            "sku_count": sku_count, "week_count": len(all_weeks), "row_count": int(len(df)),
            "first_week": all_weeks[0].strftime("%Y-%m-%d"), "last_week": all_weeks[-1].strftime("%Y-%m-%d"),
            "train_weeks": len(train_idx), "validation_weeks": len(val_idx), "test_weeks": len(test_idx),
        },
        "fold_plan": fold_plan,
        "prediction_row_count": int(len(pred)),
        "forecast_method_counts": method_counts,
        "fallback_target_details": fallback_target_details,
        "metrics": metrics,
        "validation": checks,
    }
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== FIXED-ORIGIN 13-WEEK CLOSURE-AWARE SEASONAL NAIVE BACKTEST ===")
    print(f"SKUs                     : {sku_count:,}")
    print(f"Weeks                    : {len(all_weeks):,}")
    print(f"Development folds        : {sum(f['phase'] == 'development' for f in folds)}")
    print(f"Validation horizon       : {len(val_idx)} weeks")
    print(f"Test horizon             : {len(test_idx)} weeks")
    print(f"Prediction rows          : {len(pred):,}")
    print("\nForecast method counts:")
    for method, count in method_counts.items():
        print(f"  {method:<44} {count:,}")
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
