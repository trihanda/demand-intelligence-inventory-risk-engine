"""
Purpose:
    Analyze weekly demand patterns by SKU and summarize overall demand trends.

Input:
    data/processed/weekly_demand.csv

Outputs:
    reports/demand_pattern_analysis/sku_demand_profile.csv
    reports/demand_pattern_analysis/weekly_demand_trend.csv
    reports/demand_pattern_analysis/demand_pattern_analysis_report.json

Usage:
    python src/analyze_demand_patterns.py

Notes:
    - The input is the SKU-week dataset produced by aggregate_weekly_demand.py.
    - Demand pattern classification uses common ADI and squared-CV thresholds:
      ADI = 1.32 and CV squared = 0.49.
    - Classification is descriptive and should be reviewed before selecting
      a forecasting model.
    - Results inherit the synthetic dataset's assumption that absent SKU-week
      transactions represent zero demand.
    - The source dataset is not modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "weekly_demand.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "demand_pattern_analysis"

SKU_PROFILE_FILE = REPORT_DIR / "sku_demand_profile.csv"
WEEKLY_TREND_FILE = REPORT_DIR / "weekly_demand_trend.csv"
REPORT_FILE = REPORT_DIR / "demand_pattern_analysis_report.json"

REQUIRED_COLUMNS = {"week_start", "week_end", "sku", "weekly_quantity"}
ADI_THRESHOLD = 1.32
CV_SQUARED_THRESHOLD = 0.49


# ---------------------------------------------------------------------------
# Data loading and input validation
# ---------------------------------------------------------------------------

def load_weekly_demand(input_file: Path) -> pd.DataFrame:
    """Load weekly demand and validate required columns and values."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Run src/aggregate_weekly_demand.py first."
        )

    demand = pd.read_csv(input_file)

    missing_columns = REQUIRED_COLUMNS - set(demand.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input is missing required columns: {missing}")

    if demand.empty:
        raise ValueError("Weekly demand dataset is empty.")

    demand["week_start"] = pd.to_datetime(
        demand["week_start"],
        errors="coerce",
    )
    demand["week_end"] = pd.to_datetime(
        demand["week_end"],
        errors="coerce",
    )
    demand["weekly_quantity"] = pd.to_numeric(
        demand["weekly_quantity"],
        errors="coerce",
    )

    invalid_mask = (
        demand["week_start"].isna()
        | demand["week_end"].isna()
        | demand["sku"].isna()
        | demand["weekly_quantity"].isna()
        | (demand["weekly_quantity"] < 0)
    )

    if invalid_mask.any():
        raise ValueError(
            f"Input contains {int(invalid_mask.sum())} rows with invalid dates, "
            "missing SKU/quantity, or negative weekly quantity."
        )

    if demand.duplicated(["sku", "week_end"]).any():
        raise ValueError("Input contains duplicate SKU-week combinations.")

    return demand


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def classify_demand_pattern(adi: float, cv_squared: float) -> str:
    """Classify demand as smooth, intermittent, erratic, or lumpy."""
    intermittent = adi >= ADI_THRESHOLD
    erratic = cv_squared >= CV_SQUARED_THRESHOLD

    if intermittent and erratic:
        return "lumpy"
    if intermittent:
        return "intermittent"
    if erratic:
        return "erratic"
    return "smooth"


def build_sku_demand_profile(demand: pd.DataFrame) -> pd.DataFrame:
    """Calculate demand volume, occurrence, variability, and intermittency."""
    records = []

    for sku, sku_demand in demand.groupby("sku", sort=True):
        quantities = sku_demand["weekly_quantity"].astype(float)
        weeks_observed = int(len(quantities))
        nonzero_quantities = quantities[quantities > 0]
        nonzero_weeks = int(len(nonzero_quantities))
        zero_weeks = weeks_observed - nonzero_weeks

        mean_demand = float(quantities.mean())
        std_demand = float(quantities.std(ddof=0))

        if nonzero_weeks > 0:
            average_demand_interval = weeks_observed / nonzero_weeks

            # Calculate demand-size variability only in weeks with positive demand.
            positive_mean = float(nonzero_quantities.mean())
            positive_std = float(nonzero_quantities.std(ddof=0))

            coefficient_of_variation = (
                positive_std / positive_mean
                if positive_mean > 0
                else None
            )
            cv_squared = (
                coefficient_of_variation ** 2
                if coefficient_of_variation is not None
                else None
            )
        else:
            average_demand_interval = None
            coefficient_of_variation = None
            cv_squared = None

        if average_demand_interval is None or cv_squared is None:
            demand_pattern = "no_demand"
        else:
            demand_pattern = classify_demand_pattern(
                average_demand_interval,
                cv_squared,
            )

        records.append(
            {
                "sku": sku,
                "total_demand": int(quantities.sum()),
                "weeks_observed": weeks_observed,
                "nonzero_demand_weeks": nonzero_weeks,
                "zero_demand_weeks": zero_weeks,
                "demand_occurrence_rate": (
                    nonzero_weeks / weeks_observed if weeks_observed else 0.0
                ),
                "average_weekly_demand": mean_demand,
                "std_weekly_demand": std_demand,
                "coefficient_of_variation": coefficient_of_variation,
                "average_demand_interval": average_demand_interval,
                "cv_squared": cv_squared,
                "demand_pattern": demand_pattern,
            }
        )

    return pd.DataFrame(records)


def build_weekly_demand_trend(demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize total demand across all SKUs for each week."""
    trend = (
        demand.groupby(["week_start", "week_end"], as_index=False)
        .agg(
            total_weekly_demand=("weekly_quantity", "sum"),
            active_sku_count=(
                "weekly_quantity",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values("week_start", ignore_index=True)
    )

    return trend


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    demand: pd.DataFrame,
    sku_profile: pd.DataFrame,
    weekly_trend: pd.DataFrame,
    input_file: Path,
) -> dict:
    """Build a machine-readable summary of demand patterns."""
    pattern_counts = {
        str(pattern): int(count)
        for pattern, count in sku_profile["demand_pattern"]
        .value_counts()
        .sort_index()
        .items()
    }

    peak_week = weekly_trend.loc[
        weekly_trend["total_weekly_demand"].idxmax()
    ]
    lowest_positive_week = weekly_trend.loc[
        weekly_trend["total_weekly_demand"].gt(0)
    ]["total_weekly_demand"].idxmin()

    return {
        "task": "demand_pattern_analysis",
        "input_file": str(input_file.relative_to(PROJECT_ROOT)),
        "analysis_grain": "SKU-week",
        "source_rows": int(len(demand)),
        "sku_count": int(demand["sku"].nunique()),
        "week_count": int(demand["week_end"].nunique()),
        "date_min": demand["week_start"].min().date().isoformat(),
        "date_max": demand["week_end"].max().date().isoformat(),
        "total_demand": int(demand["weekly_quantity"].sum()),
        "sku_demand_pattern_counts": pattern_counts,
        "mean_total_weekly_demand": float(
            weekly_trend["total_weekly_demand"].mean()
        ),
        "median_total_weekly_demand": float(
            weekly_trend["total_weekly_demand"].median()
        ),
        "peak_demand_week": {
            "week_start": peak_week["week_start"].date().isoformat(),
            "week_end": peak_week["week_end"].date().isoformat(),
            "total_weekly_demand": int(peak_week["total_weekly_demand"]),
        },
        "lowest_positive_demand_week": {
            "week_start": weekly_trend.loc[
                lowest_positive_week, "week_start"
            ].date().isoformat(),
            "week_end": weekly_trend.loc[
                lowest_positive_week, "week_end"
            ].date().isoformat(),
            "total_weekly_demand": int(
                weekly_trend.loc[
                    lowest_positive_week, "total_weekly_demand"
                ]
            ),
        },
        "classification_method": {
            "adi_threshold": ADI_THRESHOLD,
            "cv_squared_threshold": CV_SQUARED_THRESHOLD,
            "categories": [
                "smooth",
                "intermittent",
                "erratic",
                "lumpy",
                "no_demand",
            ],
        },
        "assumptions": [
            "Zero-demand weeks inherit the synthetic dataset's complete "
            "transaction-capture assumption.",
            "ADI/CV-squared classes are descriptive diagnostics, not automatic "
            "model-selection decisions.",
            "The first and last calendar weeks may be partial relative to the "
            "historical date range.",
        ],
        "validation": {
            "one_profile_row_per_sku": bool(
                not sku_profile["sku"].duplicated().any()
            ),
            "one_trend_row_per_week": bool(
                not weekly_trend["week_end"].duplicated().any()
            ),
            "profile_total_reconciles": bool(
                sku_profile["total_demand"].sum()
                == demand["weekly_quantity"].sum()
            ),
            "trend_total_reconciles": bool(
                weekly_trend["total_weekly_demand"].sum()
                == demand["weekly_quantity"].sum()
            ),
            "no_negative_demand": bool(
                (demand["weekly_quantity"] >= 0).all()
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """Run demand pattern analysis and save results."""
    demand = load_weekly_demand(INPUT_FILE)
    sku_profile = build_sku_demand_profile(demand)
    weekly_trend = build_weekly_demand_trend(demand)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = build_report(
        demand=demand,
        sku_profile=sku_profile,
        weekly_trend=weekly_trend,
        input_file=INPUT_FILE,
    )

    sku_profile.to_csv(SKU_PROFILE_FILE, index=False)
    weekly_trend.to_csv(WEEKLY_TREND_FILE, index=False)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("DEMAND PATTERN ANALYSIS")
    print(f"Input rows             : {report['source_rows']:,}")
    print(f"SKUs                   : {report['sku_count']:,}")
    print(f"Weeks                  : {report['week_count']:,}")
    print(f"Total demand           : {report['total_demand']:,}")
    print("Demand pattern counts  :")
    for pattern, count in report["sku_demand_pattern_counts"].items():
        print(f"  {pattern:<14}: {count:,}")
    print("Validation:")
    for check, passed in report["validation"].items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
    print(f"SKU profile            : {SKU_PROFILE_FILE}")
    print(f"Weekly trend           : {WEEKLY_TREND_FILE}")
    print(f"Report                 : {REPORT_FILE}")

    if not all(report["validation"].values()):
        raise ValueError(
            "One or more demand analysis checks failed. Review "
            f"{REPORT_FILE}."
        )

    print("Demand pattern analysis completed successfully.")


if __name__ == "__main__":
    main()
