"""Analyze demand occurrence and size variability for hardware SKU-week series.

Input:  data/processed/weekly_demand_hardware.csv
Outputs:
  reports/demand_pattern_analysis/sku_demand_profile.csv
  reports/demand_pattern_analysis/weekly_demand_trend.csv
  reports/demand_pattern_analysis/demand_pattern_analysis_report.json

ADI and CV-squared are descriptive diagnostics, not model predictors. Weeks with
zero configured business days are excluded from occurrence/size diagnostics so
business closures are not mistaken for SKU-level intermittency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ADI_THRESHOLD = 1.32
CV_SQUARED_THRESHOLD = 0.49
REQUIRED = {"week_start", "week_end", "sku", "weekly_quantity", "business_days_in_week"}


def get_root() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args().project_root.resolve()


def load_weekly_demand(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weekly demand file not found: {path}\nRun build_weekly_demand_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Weekly demand missing columns: {sorted(missing)}")
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df["weekly_quantity"] = pd.to_numeric(df["weekly_quantity"], errors="coerce")
    df["business_days_in_week"] = pd.to_numeric(df["business_days_in_week"], errors="coerce")
    if df[["week_start", "week_end", "sku", "weekly_quantity", "business_days_in_week"]].isna().any().any():
        raise ValueError("Weekly demand has missing or invalid required values.")
    if (df["weekly_quantity"] < 0).any():
        raise ValueError("Weekly demand contains negative quantities.")
    if not df["week_start"].dt.dayofweek.eq(0).all():
        raise ValueError("week_start must be Monday.")
    if not (df["week_end"] == df["week_start"] + pd.Timedelta(days=6)).all():
        raise ValueError("week_end must equal week_start + 6 days.")
    if df.duplicated(["sku", "week_start"]).any():
        raise ValueError("Duplicate SKU-week combinations found.")
    return df.sort_values(["sku", "week_start"], ignore_index=True)


def classify_pattern(adi: float | None, cv2: float | None) -> str:
    if adi is None or cv2 is None:
        return "no_demand"
    intermittent = adi >= ADI_THRESHOLD
    erratic = cv2 >= CV_SQUARED_THRESHOLD
    if intermittent and erratic:
        return "lumpy"
    if intermittent:
        return "intermittent"
    if erratic:
        return "erratic"
    return "smooth"


def build_sku_profile(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sku, group in df.groupby("sku", sort=True):
        # Closure-only weeks are excluded from ADI/CV²; regular zero-demand weeks remain.
        valid = group.loc[group["business_days_in_week"].gt(0), "weekly_quantity"].astype(float)
        positive = valid.loc[valid.gt(0)]
        n_weeks, n_positive = int(len(valid)), int(len(positive))
        adi = n_weeks / n_positive if n_positive else None
        if n_positive:
            mean_positive = float(positive.mean())
            cv2 = float((positive.std(ddof=0) / mean_positive) ** 2) if mean_positive else None
            positive_std = float(positive.std(ddof=0))
        else:
            cv2, positive_std = None, None
        records.append({
            "sku": sku,
            "total_demand_all_weeks": int(group["weekly_quantity"].sum()),
            "weeks_in_panel": int(len(group)),
            "full_closure_weeks_excluded": int(group["business_days_in_week"].eq(0).sum()),
            "valid_business_weeks": n_weeks,
            "nonzero_demand_weeks": n_positive,
            "zero_demand_business_weeks": n_weeks - n_positive,
            "demand_occurrence_rate": n_positive / n_weeks if n_weeks else 0.0,
            "average_demand_per_business_week": float(valid.mean()) if n_weeks else None,
            "average_positive_demand": float(positive.mean()) if n_positive else None,
            "std_positive_demand": positive_std,
            "average_demand_interval": adi,
            "cv_squared_positive_demand": cv2,
            "demand_pattern": classify_pattern(adi, cv2),
        })
    return pd.DataFrame(records)


def build_weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    calendar_cols = [c for c in [
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
        "company_closure_days_in_week", "company_closure_weekdays_in_week",
        "public_holiday_name", "collective_leave_name", "company_closure_name",
        "closure_reason", "is_closure_affected_week", "is_public_holiday_week",
        "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week",
    ] if c in df.columns]
    for col in calendar_cols:
        counts = df.groupby("week_start")[col].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(f"Calendar feature '{col}' is inconsistent across SKUs within a week.")
    agg = df.groupby(["week_start", "week_end"], as_index=False).agg(
        total_weekly_demand=("weekly_quantity", "sum"),
        active_sku_count=("weekly_quantity", lambda x: int(x.gt(0).sum())),
        zero_demand_sku_count=("weekly_quantity", lambda x: int(x.eq(0).sum())),
    )
    calendar = df[["week_start", *calendar_cols]].drop_duplicates("week_start")
    trend = agg.merge(calendar, on="week_start", how="left", validate="one_to_one")
    trend = trend.sort_values("week_start", ignore_index=True)
    trend["rolling_13_week_avg"] = trend["total_weekly_demand"].rolling(13, min_periods=13).mean()
    trend["year"] = trend["week_start"].dt.year
    trend["quarter"] = trend["week_start"].dt.quarter
    trend["iso_week"] = trend["week_start"].dt.isocalendar().week.astype(int)
    return trend


def main() -> None:
    root = get_root()
    input_path = root / "data/processed/weekly_demand_hardware.csv"
    out_dir = root / "reports/demand_pattern_analysis"
    demand = load_weekly_demand(input_path)
    sku_profile = build_sku_profile(demand)
    weekly_trend = build_weekly_trend(demand)

    sku_count = int(demand["sku"].nunique())
    week_count = int(demand["week_start"].nunique())
    expected_rows = sku_count * week_count
    pattern_counts = {str(k): int(v) for k, v in sku_profile["demand_pattern"].value_counts().sort_index().items()}
    full_closure = weekly_trend.get("is_full_closure_week", pd.Series(False, index=weekly_trend.index)).astype(bool)
    validation = {
        "complete_sku_week_grid": len(demand) == expected_rows,
        "no_duplicate_sku_week_pairs": not demand.duplicated(["sku", "week_start"]).any(),
        "no_missing_or_negative_demand": bool(demand["weekly_quantity"].notna().all() and demand["weekly_quantity"].ge(0).all()),
        "expected_600_hardware_skus": sku_count == 600,
        "calendar_features_consistent_per_week": bool(all(
            (demand.groupby("week_start")[col].nunique(dropna=False) <= 1).all()
            for col in [c for c in [
                "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
                "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
                "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
                "company_closure_days_in_week", "company_closure_weekdays_in_week",
                "public_holiday_name", "collective_leave_name", "company_closure_name",
                "closure_reason", "is_closure_affected_week", "is_public_holiday_week",
                "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week",
            ] if c in demand.columns]
        )),
        "full_closure_weeks_have_zero_demand": bool(weekly_trend.loc[full_closure, "total_weekly_demand"].eq(0).all()),
        "adi_cv2_exclude_full_closure_weeks": bool((sku_profile["valid_business_weeks"] + sku_profile["full_closure_weeks_excluded"] == sku_profile["weeks_in_panel"]).all()),
    }
    report = {
        "task": "demand_pattern_analysis",
        "input_file": "data/processed/weekly_demand_hardware.csv",
        "analysis_grain": "hardware SKU-week",
        "source_rows": int(len(demand)),
        "sku_count": sku_count,
        "week_count": week_count,
        "first_week_start": demand["week_start"].min().date().isoformat(),
        "last_week_end": demand["week_end"].max().date().isoformat(),
        "total_demand": int(demand["weekly_quantity"].sum()),
        "zero_demand_sku_week_rows": int(demand["weekly_quantity"].eq(0).sum()),
        "zero_demand_sku_week_share": float(demand["weekly_quantity"].eq(0).mean()),
        "public_holiday_affected_week_count": int(weekly_trend.get("is_public_holiday_week", pd.Series(False, index=weekly_trend.index)).astype(bool).sum()),
        "collective_leave_affected_week_count": int(weekly_trend.get("is_collective_leave_week", pd.Series(False, index=weekly_trend.index)).astype(bool).sum()),
        "total_public_holiday_weekdays": int(weekly_trend.get("public_holiday_weekdays_in_week", pd.Series(0, index=weekly_trend.index)).sum()),
        "total_collective_leave_weekdays": int(weekly_trend.get("collective_leave_weekdays_in_week", pd.Series(0, index=weekly_trend.index)).sum()),
        "demand_pattern_counts": pattern_counts,
        "adi_threshold": ADI_THRESHOLD,
        "cv_squared_threshold": CV_SQUARED_THRESHOLD,
        "cv_squared_definition": "CV squared of positive weekly quantities; population standard deviation ddof=0.",
        "closure_treatment": "Weeks with zero configured business days are excluded from ADI and CV-squared diagnostics; the target panel retains their zero demand rows.",
        "validation": validation,
        "all_checks_pass": all(validation.values()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    sku_profile.to_csv(out_dir / "sku_demand_profile.csv", index=False)
    weekly_trend.to_csv(out_dir / "weekly_demand_trend.csv", index=False, date_format="%Y-%m-%d")
    (out_dir / "demand_pattern_analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DEMAND PATTERN ANALYSIS")
    print(f"Input rows             : {len(demand):,}")
    print(f"Hardware SKUs          : {sku_count:,}")
    print(f"Complete weeks         : {week_count:,}")
    print(f"Total demand           : {report['total_demand']:,}")
    print("Pattern counts:")
    for name, count in pattern_counts.items():
        print(f"  {name:<14}: {count:,}")
    for name, passed in validation.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not report["all_checks_pass"]:
        raise SystemExit("Demand pattern analysis failed validation.")
    print(f"Outputs                : {out_dir}")


if __name__ == "__main__":
    main()
