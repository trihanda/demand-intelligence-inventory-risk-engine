"""Analyze weekly hardware demand trend and descriptive seasonality.

Input:  data/processed/weekly_demand_hardware.csv
Outputs: reports/seasonality_trend/*.csv, *.json, and PNG charts.

Weekly values are assigned to calendar periods using Monday week_start. The
13-week rolling average is descriptive and includes the current observed week;
it is not a leakage-safe predictive feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = {"week_start", "week_end", "sku", "weekly_quantity", "business_days_in_week"}


def get_root() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args().project_root.resolve()


def load_demand(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weekly demand file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Weekly demand missing columns: {sorted(missing)}")
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df["weekly_quantity"] = pd.to_numeric(df["weekly_quantity"], errors="coerce")
    df["business_days_in_week"] = pd.to_numeric(df["business_days_in_week"], errors="coerce")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Weekly demand contains missing required values.")
    if df["weekly_quantity"].lt(0).any() or df.duplicated(["sku", "week_start"]).any():
        raise ValueError("Negative demand or duplicate SKU-week pair found.")
    return df.sort_values(["week_start", "sku"], ignore_index=True)


def build_weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    calendar_fields = [c for c in [
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
        "company_closure_days_in_week", "company_closure_weekdays_in_week",
        "public_holiday_name", "collective_leave_name", "company_closure_name",
        "closure_reason", "is_closure_affected_week", "is_public_holiday_week",
        "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week",
    ] if c in df.columns]
    for col in calendar_fields:
        if (df.groupby("week_start")[col].nunique(dropna=False) > 1).any():
            raise ValueError(f"Calendar field {col} is inconsistent across SKUs.")
    weekly = df.groupby(["week_start", "week_end"], as_index=False).agg(
        total_demand=("weekly_quantity", "sum"),
        active_sku_count=("weekly_quantity", lambda x: int(x.gt(0).sum())),
        zero_demand_sku_count=("weekly_quantity", lambda x: int(x.eq(0).sum())),
    )
    cal = df[["week_start", *calendar_fields]].drop_duplicates("week_start")
    weekly = weekly.merge(cal, on="week_start", how="left", validate="one_to_one")
    weekly = weekly.sort_values("week_start", ignore_index=True)
    weekly["rolling_13_week_avg"] = weekly["total_demand"].rolling(13, min_periods=13).mean()
    weekly["year"] = weekly["week_start"].dt.year
    weekly["quarter"] = weekly["week_start"].dt.quarter
    weekly["month"] = weekly["week_start"].dt.month
    weekly["iso_week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["demand_per_business_day"] = np.where(
        weekly["business_days_in_week"].gt(0),
        weekly["total_demand"] / weekly["business_days_in_week"],
        np.nan,
    )
    return weekly


def summarize_period(weekly: pd.DataFrame, keys: list[str], labels: list[str]) -> pd.DataFrame:
    grouped = weekly.groupby(keys, dropna=False, as_index=False).agg(
        total_demand=("total_demand", "sum"),
        average_weekly_demand=("total_demand", "mean"),
        weeks=("total_demand", "size"),
        business_days=("business_days_in_week", "sum"),
        closure_affected_weeks=("is_closure_affected_week", "sum") if "is_closure_affected_week" in weekly.columns else ("total_demand", lambda x: 0),
        public_holiday_weekdays=("public_holiday_weekdays_in_week", "sum") if "public_holiday_weekdays_in_week" in weekly.columns else ("total_demand", lambda x: 0),
        collective_leave_weekdays=("collective_leave_weekdays_in_week", "sum") if "collective_leave_weekdays_in_week" in weekly.columns else ("total_demand", lambda x: 0),
        company_closure_weekdays=("company_closure_weekdays_in_week", "sum") if "company_closure_weekdays_in_week" in weekly.columns else ("total_demand", lambda x: 0),
    )
    grouped["average_units_per_business_day"] = np.where(
        grouped["business_days"].gt(0), grouped["total_demand"] / grouped["business_days"], np.nan
    )
    for key, label in zip(keys, labels):
        if key != label:
            grouped.rename(columns={key: label}, inplace=True)
    return grouped


def main() -> None:
    root = get_root()
    input_path = root / "data/processed/weekly_demand_hardware.csv"
    out_dir = root / "reports/seasonality_trend"
    df = load_demand(input_path)
    weekly = build_weekly_trend(df)
    sku_count = int(df["sku"].nunique())
    week_count = int(df["week_start"].nunique())
    expected_rows = sku_count * week_count

    monthly = weekly.assign(month_period=weekly["week_start"].dt.to_period("M").astype(str)).groupby("month_period", as_index=False).agg(
        total_demand=("total_demand", "sum"),
        average_weekly_demand=("total_demand", "mean"),
        weeks=("total_demand", "size"),
        business_days=("business_days_in_week", "sum"),
        closure_affected_weeks=("is_closure_affected_week", "sum") if "is_closure_affected_week" in weekly.columns else ("total_demand", lambda x: 0),
        public_holiday_weekdays=("public_holiday_weekdays_in_week", "sum") if "public_holiday_weekdays_in_week" in weekly.columns else ("total_demand", lambda x: 0),
        collective_leave_weekdays=("collective_leave_weekdays_in_week", "sum") if "collective_leave_weekdays_in_week" in weekly.columns else ("total_demand", lambda x: 0),
        company_closure_weekdays=("company_closure_weekdays_in_week", "sum") if "company_closure_weekdays_in_week" in weekly.columns else ("total_demand", lambda x: 0),
    )
    monthly["average_units_per_business_day"] = np.where(monthly["business_days"].gt(0), monthly["total_demand"] / monthly["business_days"], np.nan)
    monthly.rename(columns={"month_period": "month"}, inplace=True)
    quarterly = summarize_period(weekly, ["quarter"], ["quarter"])
    annual_weekly = weekly.groupby(["year", "iso_week"], as_index=False).agg(
        average_demand=("total_demand", "mean"),
        median_demand=("total_demand", "median"),
        observations=("total_demand", "size"),
    )

    profile = weekly.pivot_table(index="iso_week", columns="year", values="total_demand", aggfunc="mean")
    correlations: dict[str, float | None] = {}
    overlap_counts: dict[str, int] = {}
    years = sorted(weekly["year"].unique().tolist())
    for i, year_a in enumerate(years):
        for year_b in years[i + 1:]:
            pair = profile[[year_a, year_b]].dropna()
            key = f"{year_a}_vs_{year_b}"
            overlap_counts[key] = int(len(pair))
            if len(pair) >= 10 and pair[year_a].nunique() > 1 and pair[year_b].nunique() > 1:
                value = pair[year_a].corr(pair[year_b])
                correlations[key] = float(value) if pd.notna(value) else None
            else:
                correlations[key] = None

    y = weekly["total_demand"].to_numpy(dtype=float)
    slope = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) > 1 else 0.0
    positive_days = weekly["business_days_in_week"].gt(0)
    weekly_business = weekly.loc[positive_days]
    quarter_means = weekly_business.groupby("quarter")["demand_per_business_day"].mean()
    highest_quarter = int(quarter_means.idxmax()) if not quarter_means.empty else None
    lowest_quarter = int(quarter_means.idxmin()) if not quarter_means.empty else None

    calendar_fields = [c for c in [
        "business_days_in_week", "weekend_days_in_week", "closure_days_in_week",
        "public_holiday_days_in_week", "public_holiday_weekdays_in_week",
        "collective_leave_days_in_week", "collective_leave_weekdays_in_week",
        "company_closure_days_in_week", "company_closure_weekdays_in_week",
        "public_holiday_name", "collective_leave_name", "company_closure_name",
        "closure_reason", "is_closure_affected_week", "is_public_holiday_week",
        "is_collective_leave_week", "is_company_closure_week", "is_full_closure_week",
    ] if c in df.columns]
    calendar_constant = all(
        (df.groupby("week_start")[col].nunique(dropna=False) <= 1).all()
        for col in calendar_fields
    )
    validation = {
        "complete_sku_week_grid": len(df) == expected_rows,
        "expected_600_hardware_skus": sku_count == 600,
        "unique_weekly_trend_rows": not weekly["week_start"].duplicated().any(),
        "no_missing_or_negative_demand": bool(df["weekly_quantity"].notna().all() and df["weekly_quantity"].ge(0).all()),
        "calendar_features_constant_per_week": bool(calendar_constant),
        "full_closure_weeks_have_zero_demand": bool(weekly.loc[weekly.get("is_full_closure_week", pd.Series(False, index=weekly.index)).astype(bool), "total_demand"].eq(0).all()),
        "partial_weeks_not_in_input_panel": bool((weekly["week_end"] - weekly["week_start"]).dt.days.eq(6).all()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_dir / "weekly_hardware_demand.csv", index=False, date_format="%Y-%m-%d")
    monthly.to_csv(out_dir / "monthly_hardware_demand.csv", index=False)
    quarterly.to_csv(out_dir / "quarterly_demand_summary.csv", index=False)
    annual_weekly.to_csv(out_dir / "annual_weekly_seasonality.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly["week_start"], weekly["total_demand"], linewidth=1, label="Weekly demand")
    ax.plot(weekly["week_start"], weekly["rolling_13_week_avg"], linewidth=2, label="13-week rolling average")
    ax.set_title("Weekly Hardware Demand Trend")
    ax.set_xlabel("Week (Monday)")
    ax.set_ylabel("Demand (units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "weekly_demand_trend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    quarter_means.reindex([1, 2, 3, 4]).plot(kind="bar", ax=ax)
    ax.set_title("Average Hardware Demand per Business Day by Quarter")
    ax.set_xlabel("Calendar quarter (assigned by Monday week_start)")
    ax.set_ylabel("Units per business day")
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "quarterly_demand_comparison.png", dpi=150)
    plt.close(fig)

    report: dict[str, Any] = {
        "task": "seasonality_and_trend_analysis",
        "input_file": "data/processed/weekly_demand_hardware.csv",
        "scope": "600 physical inventory-managed hardware SKUs",
        "week_definition": "Monday-Sunday; weeks assigned to month/quarter by Monday week_start",
        "rolling_average_note": "13-week rolling average includes current observed week; descriptive only, not a leakage-safe model feature.",
        "sku_count": sku_count,
        "week_count": week_count,
        "row_count": int(len(df)),
        "first_week": weekly["week_start"].min().date().isoformat(),
        "last_week": weekly["week_end"].max().date().isoformat(),
        "total_demand": int(df["weekly_quantity"].sum()),
        "mean_total_weekly_demand": float(weekly["total_demand"].mean()),
        "median_total_weekly_demand": float(weekly["total_demand"].median()),
        "linear_trend_slope_units_per_week": slope,
        "first_13_week_average": float(weekly["total_demand"].head(13).mean()),
        "last_13_week_average": float(weekly["total_demand"].tail(13).mean()),
        "highest_average_demand_per_business_day_quarter": highest_quarter,
        "lowest_average_demand_per_business_day_quarter": lowest_quarter,
        "annual_weekly_profile_correlations": correlations,
        "comparable_weeks_by_year_pair": overlap_counts,
        "full_closure_week_count": int(weekly.get("is_full_closure_week", pd.Series(False, index=weekly.index)).astype(bool).sum()),
        "public_holiday_affected_week_count": int(weekly.get("is_public_holiday_week", pd.Series(False, index=weekly.index)).astype(bool).sum()),
        "collective_leave_affected_week_count": int(weekly.get("is_collective_leave_week", pd.Series(False, index=weekly.index)).astype(bool).sum()),
        "total_public_holiday_weekdays": int(weekly.get("public_holiday_weekdays_in_week", pd.Series(0, index=weekly.index)).sum()),
        "total_collective_leave_weekdays": int(weekly.get("collective_leave_weekdays_in_week", pd.Series(0, index=weekly.index)).sum()),
        "validation": validation,
        "all_checks_pass": all(validation.values()),
    }
    (out_dir / "seasonality_trend_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SEASONALITY AND TREND ANALYSIS")
    print(f"Hardware SKUs       : {sku_count:,}")
    print(f"Weeks               : {week_count:,}")
    print(f"SKU-week rows       : {len(df):,}")
    print(f"Total demand        : {report['total_demand']:,}")
    print(f"Highest avg/day Q   : Q{highest_quarter}")
    print(f"Lowest avg/day Q    : Q{lowest_quarter}")
    for name, passed in validation.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not report["all_checks_pass"]:
        raise SystemExit("Seasonality analysis failed validation.")
    print(f"Outputs              : {out_dir}")


if __name__ == "__main__":
    main()
