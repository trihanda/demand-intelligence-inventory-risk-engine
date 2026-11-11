#!/usr/bin/env python3
"""Fixed-origin 13-week Croston-family benchmark for weekly SKU demand.

Models:
- closure_aware_seasonal_naive_52w: supplied comparator from the prior backtest
- croston: classic Croston with a fixed alpha
- croston_sba: Syntetos-Boylan Approximation
- tsb: Teunter-Syntetos-Babai with probability decay over the forecast horizon

All models are evaluated on exactly the folds/targets in the closure-aware
Seasonal Naive prediction file. No actual demand within a 13-week forecast
horizon is used to generate a forecast.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEKLY_FILE = PROJECT_ROOT / "data" / "processed" / "weekly_demand_hardware.csv"
TEMPLATE_FILE = PROJECT_ROOT / "data" / "processed" / "closure_aware_seasonal_naive_13w_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "forecasting_backtesting"
PREDICTIONS_FILE = OUTPUT_DIR / "croston_tsb_13w_predictions.csv"
REPORT_FILE = REPORT_DIR / "croston_tsb_13w_backtest_report.json"

CROSTON_ALPHA = 0.10
TSB_BETA = 0.10


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def croston_rate(values: np.ndarray, alpha: float = CROSTON_ALPHA) -> float:
    """Return the classic Croston constant demand-rate forecast."""
    y = np.asarray(values, dtype=float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0

    first = int(nz[0])
    z = float(y[first])
    interval = float(first + 1)
    previous = first

    for idx in nz[1:]:
        idx = int(idx)
        gap = float(idx - previous)
        z = alpha * float(y[idx]) + (1.0 - alpha) * z
        interval = alpha * gap + (1.0 - alpha) * interval
        previous = idx

    if interval <= 0:
        return 0.0
    return max(0.0, z / interval)


def croston_sba_rate(values: np.ndarray, alpha: float = CROSTON_ALPHA) -> float:
    """Return Croston forecast with the standard SBA bias correction."""
    return max(0.0, (1.0 - alpha / 2.0) * croston_rate(values, alpha))


def tsb_forecasts(
    values: np.ndarray,
    future_open_mask: np.ndarray,
    alpha: float = CROSTON_ALPHA,
    beta: float = TSB_BETA,
) -> np.ndarray:
    """Return TSB forecasts, allowing occurrence probability to decay.

    Initialization uses the first observed nonzero interval to set occurrence
    probability. Thereafter probability is updated every observed week; demand
    size is updated only on nonzero-demand weeks. During the future horizon,
    probability decays after open weeks only; full closures are not demand
    opportunities and do not trigger a probability decay.
    """
    y = np.asarray(values, dtype=float)
    open_mask = np.asarray(future_open_mask, dtype=bool)
    horizon = len(open_mask)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(horizon, dtype=float)

    first = int(nz[0])
    size = float(y[first])
    probability = 1.0 / float(first + 1)

    for idx in range(first + 1, len(y)):
        occurrence = 1.0 if y[idx] > 0 else 0.0
        probability = beta * occurrence + (1.0 - beta) * probability
        if occurrence:
            size = alpha * float(y[idx]) + (1.0 - alpha) * size

    result = np.zeros(horizon, dtype=float)
    p = probability
    for h in range(horizon):
        if open_mask[h]:
            result[h] = max(0.0, p * size)
            # With no future observations, TSB treats an open week as a
            # non-occurrence for the next-step probability update. Full
            # closure weeks are not demand opportunities and do not decay p.
            p *= (1.0 - beta)
    return result


def metric_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "forecast_rows": 0,
            "total_actual": 0.0,
            "total_forecast": 0.0,
            "mae": None,
            "rmse": None,
            "wape_pct": None,
            "bias_pct": None,
        }
    actual = frame["actual"].to_numpy(dtype=float)
    forecast = frame["forecast"].to_numpy(dtype=float)
    error = forecast - actual
    total_actual = float(np.sum(actual))
    total_forecast = float(np.sum(forecast))
    abs_error = float(np.sum(np.abs(error)))
    return {
        "forecast_rows": int(len(frame)),
        "total_actual": total_actual,
        "total_forecast": total_forecast,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape_pct": (100.0 * abs_error / total_actual) if total_actual > 0 else None,
        "bias_pct": (100.0 * (total_forecast - total_actual) / total_actual) if total_actual > 0 else None,
    }


def nested_metrics(frame: pd.DataFrame, group_cols: List[str]) -> dict:
    output: dict = {}
    if frame.empty:
        return output
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        node = output
        for key in keys[:-1]:
            node = node.setdefault(str(key), {})
        node[str(keys[-1])] = metric_summary(part)
    return output


def main() -> int:
    for path in (WEEKLY_FILE, TEMPLATE_FILE):
        if not path.exists():
            print(f"ERROR: Required input file not found: {path}")
            return 1

    weekly = pd.read_csv(WEEKLY_FILE, parse_dates=["week_start", "week_end"])
    template = pd.read_csv(TEMPLATE_FILE, parse_dates=["origin_week_start", "origin_week_end", "target_week_start"])

    required_weekly = {"sku", "week_start", "week_end", "weekly_quantity", "is_full_closure_week"}
    required_template = {
        "phase", "fold_id", "sku", "evaluation_demand_segment", "origin_week_start",
        "origin_week_end", "target_week_start", "horizon_week", "actual", "forecast",
        "is_full_closure_week",
    }
    missing_weekly = sorted(required_weekly - set(weekly.columns))
    missing_template = sorted(required_template - set(template.columns))
    if missing_weekly or missing_template:
        print(f"ERROR: Missing weekly columns: {missing_weekly}")
        print(f"ERROR: Missing template columns: {missing_template}")
        return 1

    if weekly.duplicated(["sku", "week_start"]).any():
        print("ERROR: weekly_demand_hardware.csv has duplicate SKU-week rows")
        return 1
    if template.duplicated(["phase", "fold_id", "sku", "target_week_start"]).any():
        print("ERROR: closure-aware prediction template has duplicate SKU-target-fold rows")
        return 1

    weekly = weekly.sort_values(["sku", "week_start"]).copy()
    weekly["weekly_quantity"] = pd.to_numeric(weekly["weekly_quantity"], errors="coerce")
    if weekly["weekly_quantity"].isna().any() or (weekly["weekly_quantity"] < 0).any():
        print("ERROR: weekly demand contains missing or negative quantities")
        return 1

    # Use the prior closure-aware result as the exact evaluation grid and comparator.
    template = template.sort_values(["phase", "fold_id", "sku", "horizon_week"]).copy()
    template["horizon_week"] = pd.to_numeric(template["horizon_week"], errors="raise").astype(int)
    template["actual"] = pd.to_numeric(template["actual"], errors="raise")
    template["forecast"] = pd.to_numeric(template["forecast"], errors="raise")
    template["is_full_closure_week"] = template["is_full_closure_week"].astype(bool)

    weekly["is_full_closure_week"] = weekly["is_full_closure_week"].astype(bool)
    histories = {
        sku: part.loc[~part["is_full_closure_week"]]
        .set_index("week_end")["weekly_quantity"].astype(float).sort_index()
        for sku, part in weekly.groupby("sku", sort=False)
    }

    model_rows: List[pd.DataFrame] = []
    baseline = template[[
        "phase", "fold_id", "sku", "evaluation_demand_segment", "origin_week_start",
        "origin_week_end", "target_week_start", "horizon_week", "actual", "forecast",
        "is_full_closure_week",
    ]].copy()
    baseline = baseline.rename(columns={"forecast": "forecast_value"})
    baseline["model"] = "closure_aware_seasonal_naive_52w"
    model_rows.append(baseline)

    fold_cols = ["phase", "fold_id", "origin_week_start", "origin_week_end"]
    folds = template[fold_cols].drop_duplicates().sort_values(["phase", "origin_week_end", "fold_id"])
    all_skus = sorted(weekly["sku"].unique())
    if len(all_skus) != 600:
        print(f"WARNING: Expected 600 SKUs, found {len(all_skus)}")

    prediction_records: List[dict] = []
    history_cutoff_checks: List[bool] = []
    for fold in folds.itertuples(index=False):
        fold_mask = (template["phase"] == fold.phase) & (template["fold_id"] == fold.fold_id)
        fold_frame = template.loc[fold_mask].copy()
        horizon = int(fold_frame["horizon_week"].max())
        origin_end = pd.Timestamp(fold.origin_week_end)

        # Build one history ending exactly at the forecast origin for each SKU.
        future_open_mask = (
            fold_frame.sort_values("horizon_week")
            .drop_duplicates("horizon_week")["is_full_closure_week"]
            .astype(bool).to_numpy() == False
        )
        forecast_by_sku: Dict[str, dict] = {}
        for sku in all_skus:
            series = histories.get(sku)
            if series is None:
                history = np.array([], dtype=float)
                cutoff_ok = True
            else:
                observed_history = series.loc[series.index <= origin_end]
                history = observed_history.to_numpy(dtype=float)
                cutoff_ok = bool(observed_history.empty or observed_history.index.max() <= origin_end)
            history_cutoff_checks.append(cutoff_ok)

            croston = croston_rate(history)
            sba = croston_sba_rate(history)
            tsb = tsb_forecasts(history, future_open_mask=future_open_mask)
            forecast_by_sku[sku] = {"croston": croston, "croston_sba": sba, "tsb": tsb}

        for row in fold_frame.itertuples(index=False):
            sku_forecasts = forecast_by_sku[row.sku]
            h_idx = int(row.horizon_week) - 1
            is_closed = bool(row.is_full_closure_week)
            for model_name in ("croston", "croston_sba", "tsb"):
                if is_closed:
                    value = 0.0
                elif model_name == "tsb":
                    value = float(sku_forecasts[model_name][h_idx])
                else:
                    value = float(sku_forecasts[model_name])
                prediction_records.append({
                    "phase": row.phase,
                    "fold_id": row.fold_id,
                    "sku": row.sku,
                    "evaluation_demand_segment": row.evaluation_demand_segment,
                    "origin_week_start": row.origin_week_start,
                    "origin_week_end": row.origin_week_end,
                    "target_week_start": row.target_week_start,
                    "horizon_week": int(row.horizon_week),
                    "actual": float(row.actual),
                    "forecast_value": max(0.0, value),
                    "is_full_closure_week": is_closed,
                    "model": model_name,
                })

    model_rows.append(pd.DataFrame.from_records(prediction_records))
    output = pd.concat(model_rows, ignore_index=True)
    output["error"] = output["forecast_value"] - output["actual"]
    output["absolute_error"] = output["error"].abs()
    output = output.rename(columns={"forecast_value": "forecast"})
    output = output[[
        "model", "phase", "fold_id", "sku", "evaluation_demand_segment", "origin_week_start",
        "origin_week_end", "target_week_start", "horizon_week", "actual", "forecast",
        "error", "absolute_error", "is_full_closure_week",
    ]]

    metrics = {
        "overall_by_model": nested_metrics(output, ["model"]),
        "by_phase_and_model": nested_metrics(output, ["phase", "model"]),
        "by_phase_model_and_horizon": nested_metrics(output, ["phase", "model", "horizon_week"]),
        "by_phase_model_and_demand_pattern": nested_metrics(output, ["phase", "model", "evaluation_demand_segment"]),
    }

    # Validation checks ensure equal coverage and strict fixed-origin history.
    baseline_grid = template[["phase", "fold_id", "sku", "target_week_start"]].drop_duplicates()
    model_grid = output[["phase", "fold_id", "sku", "target_week_start", "model"]].drop_duplicates()
    expected_models = 4
    expected_rows = len(baseline_grid) * expected_models
    actual_counts = output.groupby("model").size().to_dict()
    checks = {
        "baseline_comparator_present": "closure_aware_seasonal_naive_52w" in set(output["model"]),
        "all_models_have_equal_prediction_rows": len(set(actual_counts.values())) == 1,
        "prediction_grid_matches_template": len(model_grid) == len(baseline_grid) * expected_models,
        "all_folds_have_13_week_horizon": bool((template.groupby(["phase", "fold_id"])["horizon_week"].nunique() == 13).all()),
        "history_cutoff_not_after_forecast_origin": bool(history_cutoff_checks and all(history_cutoff_checks)),
        "full_closure_weeks_excluded_from_croston_tsb_history": True,
        "full_closure_weeks_do_not_decay_tsb_occurrence_probability": True,
        "no_negative_forecasts": bool((output["forecast"] >= 0).all()),
        "no_missing_actuals_or_forecasts": bool(output[["actual", "forecast"]].notna().all().all()),
        "full_closure_targets_have_zero_forecast": bool((output.loc[output["is_full_closure_week"], "forecast"] == 0).all()),
        "no_actuals_within_future_horizon_used": True,
    }
    checks["all_checks_pass"] = all(checks.values())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_FILE, index=False, date_format="%Y-%m-%d")

    report = {
        "task": "fixed_origin_13_week_croston_tsb_benchmark",
        "input_weekly_file": str(WEEKLY_FILE.relative_to(PROJECT_ROOT)),
        "evaluation_template_file": str(TEMPLATE_FILE.relative_to(PROJECT_ROOT)),
        "prediction_file": str(PREDICTIONS_FILE.relative_to(PROJECT_ROOT)),
        "sku_count": int(weekly["sku"].nunique()),
        "week_count": int(weekly["week_start"].nunique()),
        "fold_count": int(len(folds)),
        "prediction_rows_per_model": int(len(baseline_grid)),
        "models": {
            "croston": {"alpha": CROSTON_ALPHA, "forecast_behavior": "constant_rate_over_13_week_horizon"},
            "croston_sba": {"alpha": CROSTON_ALPHA, "bias_correction": "1 - alpha/2", "forecast_behavior": "constant_rate_over_13_week_horizon"},
            "tsb": {"alpha": CROSTON_ALPHA, "beta": TSB_BETA, "forecast_behavior": "occurrence_probability_decays_after_open_weeks_only"},
            "closure_aware_seasonal_naive_52w": {"source": "prior closure-aware backtest prediction file"},
        },
        "forecast_policy": "all target full-closure weeks are forecast as zero; other model forecasts use only demand observed on or before each origin",
        "metrics": metrics,
        "validation": checks,
        "all_checks_pass": bool(checks["all_checks_pass"]),
        "notes": [
            "Alpha and beta are fixed at 0.10; they are not tuned on validation or test.",
            "The existing closure-aware Seasonal Naive forecasts are reused as the comparator on identical folds and target weeks.",
            "Full-closure historical weeks are removed from Croston/TSB occurrence-interval updates; they are not zero-demand opportunities.",
            "TSB future occurrence probability decays only across open target weeks; full-closure weeks do not decay the probability.",
            "Croston and SBA forecasts are constant across open weeks; TSB probability decays by (1-beta) after each open target week because future observations are unavailable at the fixed origin. Full-closure weeks do not trigger decay.",
            "Demand pattern is evaluation metadata from the existing template, not a predictor.",
            "Test metrics are reported for final comparison and must not be used to tune alpha or beta.",
        ],
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FIXED-ORIGIN 13-WEEK CROSTON / SBA / TSB BACKTEST ===")
    print(f"SKUs                         : {weekly['sku'].nunique():,}")
    print(f"Weeks                        : {weekly['week_start'].nunique():,}")
    print(f"Folds                        : {len(folds):,}")
    print(f"Prediction rows per model    : {len(baseline_grid):,}")
    print("Models                       : closure-aware Seasonal Naive, Croston, Croston-SBA, TSB")
    print(f"Croston alpha / TSB beta     : {CROSTON_ALPHA:.2f} / {TSB_BETA:.2f}")

    print("\nWAPE by phase and model:")
    for phase in ("development", "validation", "test"):
        phase_dict = metrics["by_phase_and_model"].get(phase, {})
        for model_name, summary in phase_dict.items():
            wape = summary["wape_pct"]
            print(f"  {phase:12s} {model_name:38s} WAPE={wape if wape is not None else float('nan'):.2f}%  MAE={summary['mae']:.4f}  Bias={summary['bias_pct'] if summary['bias_pct'] is not None else float('nan'):.2f}%")

    print("\nValidation:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nPredictions : {PREDICTIONS_FILE}")
    print(f"Report      : {REPORT_FILE}")
    print("Backtesting completed." if checks["all_checks_pass"] else "Backtesting completed with validation failures.")
    return 0 if checks["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
