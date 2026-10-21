"""
Seasonality and Trend Analysis

Purpose:
    Analyze weekly demand trends and potential seasonal patterns for
    inventory-managed hardware SKUs to support forecasting preparation.

Input:
    - data/processed/weekly_demand.csv
    - data/synthetic/products.csv
    - data/synthetic/calendar.csv

Outputs:
    - reports/seasonality_trend/weekly_hardware_demand.csv
    - reports/seasonality_trend/monthly_hardware_demand.csv
    - reports/seasonality_trend/quarterly_demand_summary.csv
    - reports/seasonality_trend/annual_weekly_seasonality.csv
    - reports/seasonality_trend/seasonality_trend_report.json
    - reports/seasonality_trend/weekly_demand_trend.png
    - reports/seasonality_trend/quarterly_demand_comparison.png

Usage:
    python src/analyze_seasonality_trend.py

Notes:
    - Analyzes hardware products with physical_inventory_flag = True.
    - Aggregates demand across hardware SKUs at weekly, monthly,
      quarterly, and annual-weekly levels.
    - Uses a 13-week rolling average to summarize short-term trends.
    - Compares weekly demand profiles across years to explore
      potential annual seasonality.
    - Quarterly differences and profile correlations are descriptive
      indicators, not proof of stable seasonality.
    - Joins daily synthetic calendar information to weekly demand, including
      business days, closure days, closure reasons, and calendar coverage.
    - The JSON report includes data-derived interpretations and calendar-aware
      caveats for zero-demand weeks and partial calendar weeks.
    - Does not modify source datasets or train forecasting models.
"""


from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEMAND_PATH = ROOT / "data/processed/weekly_demand.csv"
PRODUCT_PATH = ROOT / "data/synthetic/products.csv"
CALENDAR_PATH = ROOT / "data/synthetic/calendar.csv"
OUTPUT_DIR = ROOT / "reports/seasonality_trend"



def load_calendar(calendar_path: Path) -> pd.DataFrame:
    """Load and validate the generated daily business calendar."""
    if not calendar_path.exists():
        raise FileNotFoundError(
            f"Calendar file not found: {calendar_path}. "
            "Expected data/synthetic/calendar.csv."
        )

    calendar = pd.read_csv(calendar_path)
    required = {"calendar_date", "is_business_day", "closure_reason"}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"Calendar is missing required columns: {sorted(missing)}")

    calendar["calendar_date"] = pd.to_datetime(
        calendar["calendar_date"], errors="coerce"
    )
    if calendar["calendar_date"].isna().any():
        raise ValueError("Calendar contains missing or invalid calendar_date values.")
    if calendar["calendar_date"].duplicated().any():
        raise ValueError("Calendar contains duplicate calendar_date values.")

    business_day_text = calendar["is_business_day"].astype(str).str.strip().str.lower()
    bool_map = {
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    }
    invalid_boolean = ~business_day_text.isin(bool_map)
    if invalid_boolean.any():
        values = sorted(business_day_text[invalid_boolean].unique().tolist())
        raise ValueError(
            "Calendar is_business_day contains unrecognized values: "
            f"{values}"
        )
    calendar["is_business_day"] = business_day_text.map(bool_map).astype(bool)
    calendar["closure_reason"] = (
        calendar["closure_reason"].fillna("").astype(str).str.strip()
    )
    calendar["week_start"] = (
        calendar["calendar_date"]
        - pd.to_timedelta(calendar["calendar_date"].dt.weekday, unit="D")
    ).dt.normalize()
    return calendar.sort_values("calendar_date").reset_index(drop=True)


def build_weekly_calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily calendar attributes into Monday-Sunday week features."""
    weekly_calendar = (
        calendar.groupby("week_start", as_index=False)
        .agg(
            calendar_days_covered=("calendar_date", "nunique"),
            business_days_in_week=("is_business_day", "sum"),
            closure_days_in_week=(
                "closure_reason", lambda values: int(values.ne("").sum())
            ),
            closure_reason=(
                "closure_reason",
                lambda values: ", ".join(sorted(set(values) - {""})),
            ),
        )
    )
    weekly_calendar["is_closure_affected_week"] = (
        weekly_calendar["closure_days_in_week"] > 0
    )
    weekly_calendar["calendar_coverage_status"] = np.where(
        weekly_calendar["calendar_days_covered"] == 7,
        "complete_week",
        "partial_week",
    )
    weekly_calendar["is_partial_calendar_week"] = (
        weekly_calendar["calendar_days_covered"].between(1, 6)
    )
    return weekly_calendar


def attach_calendar_features(
    weekly: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Join weekly calendar features and summarize zero-demand/closure overlap."""
    daily_min = calendar["calendar_date"].min().date().isoformat()
    daily_max = calendar["calendar_date"].max().date().isoformat()
    calendar_features = build_weekly_calendar_features(calendar)
    enriched = weekly.merge(
        calendar_features, on="week_start", how="left", validate="one_to_one"
    )

    # Preserve missingness when the calendar has no dates for a demand week.
    enriched["calendar_days_covered"] = (
        enriched["calendar_days_covered"].fillna(0).astype(int)
    )
    for column in ("business_days_in_week", "closure_days_in_week"):
        enriched[column] = enriched[column].astype("Int64")
    enriched["closure_reason"] = enriched["closure_reason"].where(
        enriched["calendar_days_covered"] > 0, pd.NA
    )
    enriched["is_closure_affected_week"] = enriched[
        "is_closure_affected_week"
    ].astype("boolean")
    enriched.loc[enriched["calendar_days_covered"] == 0, "is_closure_affected_week"] = pd.NA
    enriched["calendar_coverage_status"] = enriched[
        "calendar_coverage_status"
    ].fillna("no_calendar_coverage")
    enriched["is_partial_calendar_week"] = enriched[
        "is_partial_calendar_week"
    ].fillna(False).astype(bool)

    zero = enriched["total_demand"].eq(0)
    affected = enriched["is_closure_affected_week"]
    zero_with_closure = int((zero & affected.fillna(False)).sum())
    zero_without_closure = int((zero & affected.eq(False).fillna(False)).sum())
    zero_unassessed = int((zero & affected.isna()).sum())

    summary = {
        "source_file": str(calendar_path_relative(CALENDAR_PATH)),
        "calendar_date_min": daily_min,
        "calendar_date_max": daily_max,
        "calendar_date_count": int(calendar["calendar_date"].nunique()),
        "complete_calendar_weeks_in_demand_series": int(
            enriched["calendar_coverage_status"].eq("complete_week").sum()
        ),
        "partial_calendar_weeks_in_demand_series": int(
            enriched["calendar_coverage_status"].eq("partial_week").sum()
        ),
        "weeks_without_calendar_coverage": int(
            enriched["calendar_coverage_status"].eq("no_calendar_coverage").sum()
        ),
        "closure_affected_weeks_in_demand_series": int(
            enriched["is_closure_affected_week"].fillna(False).sum()
        ),
        "zero_demand_weeks": int(zero.sum()),
        "zero_demand_weeks_with_configured_closure": zero_with_closure,
        "zero_demand_weeks_without_configured_closure": zero_without_closure,
        "zero_demand_weeks_without_calendar_coverage": zero_unassessed,
        "interpretation": (
            f"Of {int(zero.sum())} zero-demand week(s), {zero_with_closure} "
            "overlap a configured closure period, "
            f"{zero_without_closure} do not overlap a configured closure, and "
            f"{zero_unassessed} cannot be assessed because calendar coverage is absent. "
            "Overlap is descriptive and does not establish that closure alone caused zero demand."
        ),
    }
    return enriched, summary


def calendar_path_relative(path: Path) -> str:
    """Return a project-relative path for report output."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_interpretations(
    weekly: pd.DataFrame,
    quarterly_avg: pd.Series,
    slope: float,
    first_13_avg: float,
    last_13_avg: float,
    highest_quarter: int,
    lowest_quarter: int,
    correlations: dict,
    comparable_week_counts: dict,
    calendar_summary: dict,
) -> dict:
    """Create data-derived interpretations and caveats for the JSON report."""
    weekly_mean = float(weekly["total_demand"].mean())
    change_units = last_13_avg - first_13_avg
    change_pct = (
        (change_units / first_13_avg) * 100
        if first_13_avg != 0
        else None
    )
    slope_pct_of_mean = (
        (slope / weekly_mean) * 100
        if weekly_mean != 0
        else None
    )

    if slope > 0:
        slope_direction = "positive"
    elif slope < 0:
        slope_direction = "negative"
    else:
        slope_direction = "flat"

    if change_pct is None:
        endpoint_comparison = (
            "The first 13-week average is zero, so percentage change "
            "between the first and last 13-week periods is undefined."
        )
    elif change_pct > 0:
        endpoint_comparison = (
            f"The last 13-week average is {change_pct:.1f}% higher than "
            "the first 13-week average. This compares two windows and does "
            "not by itself establish a sustained increase."
        )
    elif change_pct < 0:
        endpoint_comparison = (
            f"The last 13-week average is {abs(change_pct):.1f}% lower than "
            "the first 13-week average. This compares two windows and does "
            "not by itself establish a sustained decrease."
        )
    else:
        endpoint_comparison = (
            "The first and last 13-week averages are equal; this endpoint "
            "comparison does not rule out fluctuations within the period."
        )

    highest_avg = float(quarterly_avg.loc[highest_quarter])
    lowest_avg = float(quarterly_avg.loc[lowest_quarter])
    quarter_gap = highest_avg - lowest_avg
    quarter_gap_pct = (
        (quarter_gap / lowest_avg) * 100
        if lowest_avg != 0
        else None
    )
    if quarter_gap_pct is None:
        quarter_gap_description = (
            "The lowest-average quarter has zero demand, so a relative "
            "percentage difference cannot be calculated."
        )
    else:
        quarter_gap_description = (
            f"Q{highest_quarter}'s average is {quarter_gap_pct:.1f}% above "
            f"Q{lowest_quarter}'s average in this observed dataset."
        )

    pairwise_interpretations = {}
    valid_correlations = []
    for pair_key, correlation in correlations.items():
        overlap = int(comparable_week_counts.get(pair_key, 0))
        if correlation is None:
            interpretation = (
                "Not interpreted as a reliable correlation because the pair "
                "has fewer than 10 overlapping ISO weeks or insufficient "
                "variation in one or both profiles."
            )
        else:
            valid_correlations.append(float(correlation))
            direction = (
                "positive" if correlation > 0 else
                "negative" if correlation < 0 else "near-zero"
            )
            interpretation = (
                f"The paired annual weekly profiles have a {direction} "
                f"correlation (r={correlation:.3f}) across {overlap} "
                "overlapping ISO weeks. This is descriptive similarity, "
                "not proof of stable or causal seasonality."
            )
        pairwise_interpretations[pair_key] = {
            "correlation": correlation,
            "comparable_iso_weeks": overlap,
            "interpretation": interpretation,
        }

    if valid_correlations:
        min_corr = min(valid_correlations)
        max_corr = max(valid_correlations)
        annual_summary = (
            f"Valid year-pair correlations range from {min_corr:.3f} to "
            f"{max_corr:.3f}. Interpret them alongside the number of "
            "overlapping weeks; partial years are not directly comparable "
            "to full years. Correlation alone does not confirm repeatable "
            "seasonality."
        )
    else:
        annual_summary = (
            "No year-pair correlation met the minimum overlap and variation "
            "requirements; annual seasonality cannot be assessed reliably "
            "from these correlations."
        )

    zero_demand_weeks = int((weekly["total_demand"] == 0).sum())
    caveats = [
        "All observations and calendar closure periods are synthetic; they represent generator assumptions, not verified real-world market behavior or official holiday declarations.",
        "Quarterly averages and annual profile correlations are descriptive diagnostics; neither proves that seasonality is stable or will persist in the forecast horizon.",
        "Calendar closure overlap helps explain recorded zero-demand weeks but does not prove closure alone caused zero demand or capture latent/unfulfilled customer demand.",
        "The analysis uses Monday-Sunday weekly bins and assigns each bin to a quarter using week_start; first or last bins may be partial relative to the configured data window.",
        "The fitted linear slope summarizes an overall linear direction and does not mean demand changes by the same amount every week. The first-versus-last 13-week comparison is not a substitute for time-based model evaluation.",
    ]
    if zero_demand_weeks:
        caveats.append(
            f"Calendar reconciliation found {calendar_summary['zero_demand_weeks_with_configured_closure']} zero-demand week(s) overlapping configured closure periods, {calendar_summary['zero_demand_weeks_without_configured_closure']} without a configured closure, and {calendar_summary['zero_demand_weeks_without_calendar_coverage']} not assessable due to missing calendar coverage. Review these distinctions before interpreting zero demand."
        )
    if calendar_summary["partial_calendar_weeks_in_demand_series"]:
        caveats.append(
            f"The calendar covers {calendar_summary['calendar_date_min']} through {calendar_summary['calendar_date_max']}; {calendar_summary['partial_calendar_weeks_in_demand_series']} demand week(s) have partial daily-calendar coverage. Do not interpret missing dates in a partial week as non-business days or closure days."
        )

    return {
        "trend_interpretation": {
            "linear_slope_direction": slope_direction,
            "linear_trend_slope_units_per_week": round(float(slope), 4),
            "linear_slope_as_percent_of_mean_weekly_demand_per_week": (
                round(float(slope_pct_of_mean), 4)
                if slope_pct_of_mean is not None else None
            ),
            "first_13_week_average": round(float(first_13_avg), 4),
            "last_13_week_average": round(float(last_13_avg), 4),
            "first_to_last_13_week_change_units": round(float(change_units), 4),
            "first_to_last_13_week_change_percent": (
                round(float(change_pct), 4) if change_pct is not None else None
            ),
            "summary": (
                f"The fitted linear trend is {slope_direction} at "
                f"{slope:.2f} units per week, equivalent to "
                f"{slope_pct_of_mean:.4f}% of mean weekly demand per week. "
                f"{endpoint_comparison}"
                if slope_pct_of_mean is not None
                else f"The fitted linear trend is {slope_direction} at {slope:.2f} units per week. {endpoint_comparison}"
            ),
        },
        "quarterly_seasonality_interpretation": {
            "highest_average_demand_quarter": highest_quarter,
            "highest_average_weekly_demand": round(highest_avg, 4),
            "lowest_average_demand_quarter": lowest_quarter,
            "lowest_average_weekly_demand": round(lowest_avg, 4),
            "high_minus_low_average_weekly_demand": round(quarter_gap, 4),
            "high_vs_low_percent_above_lowest": (
                round(quarter_gap_pct, 4) if quarter_gap_pct is not None else None
            ),
            "summary": (
                f"Q{highest_quarter} has the highest average weekly demand and "
                f"Q{lowest_quarter} the lowest among the observed calendar "
                f"quarters. {quarter_gap_description} This is an observed "
                "quarter-level difference, not confirmation of stable seasonality."
            ),
        },
        "annual_profile_interpretation": {
            "summary": annual_summary,
            "year_pair_details": pairwise_interpretations,
        },
        "caveats": caveats,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    demand = pd.read_csv(DEMAND_PATH, parse_dates=["week_start"])
    products = pd.read_csv(PRODUCT_PATH)
    calendar = load_calendar(CALENDAR_PATH)

    required_demand = {"sku", "week_start", "weekly_quantity"}
    if not required_demand.issubset(demand.columns):
        raise ValueError(
            f"Missing columns: {required_demand - set(demand.columns)}"
        )

    # 2. Select inventory-managed hardware
    physical = products["physical_inventory_flag"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    hardware = products.loc[
        products["product_type"].eq("hardware") & physical, "sku"
    ].drop_duplicates()

    data = demand[demand["sku"].isin(hardware)].copy()
    data["weekly_quantity"] = pd.to_numeric(
        data["weekly_quantity"], errors="raise"
    )

    if data["weekly_quantity"].isna().any():
        raise ValueError("Missing demand values found.")
    if (data["weekly_quantity"] < 0).any():
        raise ValueError("Negative weekly demand found.")
    if data.duplicated(["sku", "week_start"]).any():
        raise ValueError("Duplicate SKU-week pairs found.")

    # 3. Validate the complete SKU-week grid
    week_count = data["week_start"].nunique()
    sku_count = data["sku"].nunique()
    expected_rows = sku_count * week_count

    if len(data) != expected_rows:
        raise ValueError(
            f"Incomplete SKU-week grid: expected {expected_rows:,}, "
            f"found {len(data):,}"
        )

    # 4. Aggregate demand across hardware SKUs
    weekly = (
        data.groupby("week_start", as_index=False)["weekly_quantity"]
        .sum()
        .rename(columns={"weekly_quantity": "total_demand"})
        .sort_values("week_start")
    )
    weekly, calendar_summary = attach_calendar_features(weekly, calendar)

    weekly["rolling_13_week_avg"] = (
        weekly["total_demand"].rolling(13, min_periods=13).mean()
    )
    weekly["year"] = weekly["week_start"].dt.year
    weekly["quarter"] = weekly["week_start"].dt.quarter
    weekly["month"] = weekly["week_start"].dt.month
    weekly["iso_week"] = (
        weekly["week_start"].dt.isocalendar().week.astype(int)
    )

    # 5. Monthly and quarterly summaries
    monthly = (
        weekly.assign(month_period=weekly["week_start"].dt.to_period("M"))
        .groupby("month_period", as_index=False)
        .agg(
            total_demand=("total_demand", "sum"),
            average_weekly_demand=("total_demand", "mean"),
            weeks=("total_demand", "size"),
        )
    )
    monthly["month"] = monthly["month_period"].astype(str)
    monthly.drop(columns="month_period", inplace=True)

    quarterly = (
        weekly.groupby("quarter", as_index=False)
        .agg(
            total_demand=("total_demand", "sum"),
            average_weekly_demand=("total_demand", "mean"),
            weeks=("total_demand", "size"),
        )
    )

    annual_weekly = (
        weekly.groupby(["year", "iso_week"], as_index=False)
        .agg(average_demand=("total_demand", "mean"))
    )

    # 6. Compare annual weekly profiles pairwise.
    # Partial years may not share enough weeks for a reliable comparison.
    profile = weekly.pivot_table(
        index="iso_week",
        columns="year",
        values="total_demand",
        aggfunc="mean",
    )

    correlations = {}
    comparable_week_counts = {}
    years = sorted(weekly["year"].unique())

    for i, year_a in enumerate(years):
        for year_b in years[i + 1:]:
            pair = profile[[year_a, year_b]].dropna()
            pair_key = f"{year_a}_vs_{year_b}"
            comparable_week_counts[pair_key] = int(len(pair))

            # Require at least 10 overlapping weeks and variation in both series.
            if (
                len(pair) >= 10
                and pair[year_a].nunique() > 1
                and pair[year_b].nunique() > 1
            ):
                value = pair[year_a].corr(pair[year_b])
                correlations[pair_key] = (
                    float(value) if pd.notna(value) else None
                )
            else:
                correlations[pair_key] = None
                print(
                    f"Skipping correlation for {pair_key}: "
                    f"only {len(pair)} comparable weeks or insufficient variation."
                )

    
    # 7. Quantify linear trend and compare early vs late demand
    y = weekly["total_demand"].to_numpy(dtype=float)
    slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])

    first_13_avg = float(weekly["total_demand"].head(13).mean())
    last_13_avg = float(weekly["total_demand"].tail(13).mean())

    quarterly_avg = quarterly.set_index("quarter")["average_weekly_demand"]
    highest_quarter = int(quarterly_avg.idxmax())
    lowest_quarter = int(quarterly_avg.idxmin())

    # 8. Save tables
    weekly.to_csv(OUTPUT_DIR / "weekly_hardware_demand.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_hardware_demand.csv", index=False)
    quarterly.to_csv(OUTPUT_DIR / "quarterly_demand_summary.csv", index=False)
    annual_weekly.to_csv(
        OUTPUT_DIR / "annual_weekly_seasonality.csv", index=False
    )

    # 9. Plot weekly trend
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        weekly["week_start"], weekly["total_demand"],
        linewidth=1, label="Weekly demand"
    )
    ax.plot(
        weekly["week_start"], weekly["rolling_13_week_avg"],
        linewidth=2, label="13-week rolling average"
    )
    ax.set_title("Weekly Hardware Demand Trend")
    ax.set_xlabel("Week")
    ax.set_ylabel("Demand (units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weekly_demand_trend.png", dpi=150)
    plt.close(fig)

    # 10. Plot average demand by quarter
    fig, ax = plt.subplots(figsize=(8, 5))
    quarterly_avg.reindex([1, 2, 3, 4]).plot(kind="bar", ax=ax)
    ax.set_title("Average Weekly Demand by Calendar Quarter")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Average weekly demand (units)")
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "quarterly_demand_comparison.png", dpi=150)
    plt.close(fig)

    # 11. Save report with data-derived interpretation and caveats
    interpretation = build_interpretations(
        weekly=weekly,
        quarterly_avg=quarterly_avg,
        slope=slope,
        first_13_avg=first_13_avg,
        last_13_avg=last_13_avg,
        highest_quarter=highest_quarter,
        lowest_quarter=lowest_quarter,
        correlations=correlations,
        comparable_week_counts=comparable_week_counts,
        calendar_summary=calendar_summary,
    )

    report = {
        "scope": "Inventory-managed hardware",
        "sku_count": int(sku_count),
        "week_count": int(week_count),
        "row_count": int(len(data)),
        "first_week": str(weekly["week_start"].min().date()),
        "last_week": str(weekly["week_start"].max().date()),
        "total_demand": int(data["weekly_quantity"].sum()),
        "weekly_demand_mean": float(weekly["total_demand"].mean()),
        "weekly_demand_std": float(weekly["total_demand"].std()),
        "linear_trend_slope_units_per_week": slope,
        "first_13_week_average": first_13_avg,
        "last_13_week_average": last_13_avg,
        "highest_average_demand_quarter": highest_quarter,
        "lowest_average_demand_quarter": lowest_quarter,
        "annual_weekly_profile_correlations": correlations,
        "comparable_weeks_by_year_pair": comparable_week_counts,
        "calendar_reconciliation": calendar_summary,
        **interpretation,
        "validation": {
            "complete_sku_week_grid": len(data) == expected_rows,
            "no_duplicate_sku_week_pairs": not data.duplicated(
                ["sku", "week_start"]
            ).any(),
            "no_missing_demand": not data["weekly_quantity"].isna().any(),
            "no_negative_demand": bool(
                (data["weekly_quantity"] >= 0).all()
            ),
            "expected_600_hardware_skus": sku_count == 600,
            "expected_157_weeks": week_count == 157,
            "expected_94200_rows": len(data) == 94200,
            "calendar_date_unique": not calendar["calendar_date"].duplicated().any(),
            "calendar_week_features_unique": not weekly["week_start"].duplicated().any(),
            "calendar_overlaps_demand_period": bool(
                calendar["calendar_date"].max().normalize() >= weekly["week_start"].min().normalize()
                and calendar["calendar_date"].min().normalize() <= weekly["week_start"].max().normalize()
            ),
        },
    }
    report["validation"]["all_checks_pass"] = all(
        report["validation"].values()
    )

    with open(
        OUTPUT_DIR / "seasonality_trend_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("SEASONALITY AND TREND ANALYSIS")
    print(f"Hardware SKUs: {sku_count:,}")
    print(f"Weeks: {week_count:,}")
    print(f"SKU-week rows: {len(data):,}")
    print(f"Total demand: {int(data['weekly_quantity'].sum()):,}")
    print(f"Linear trend slope: {slope:.3f} units/week")
    print(f"Highest average quarter: Q{highest_quarter}")
    print(f"Lowest average quarter: Q{lowest_quarter}")
    print(
        "Zero-demand weeks overlapping closures: "
        f"{calendar_summary['zero_demand_weeks_with_configured_closure']}"
    )
    print(
        "Partial calendar weeks: "
        f"{calendar_summary['partial_calendar_weeks_in_demand_series']}"
    )
    print("\nValidation:")
    for name, passed in report["validation"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")

    if not report["validation"]["all_checks_pass"]:
        raise SystemExit("Validation failed. Review the report.")


if __name__ == "__main__":
    main()
