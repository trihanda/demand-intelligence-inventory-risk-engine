"""
Standardize sales transactions for demand analysis.

Purpose:
    Apply documented inclusion and exclusion rules to raw sales transactions
    before weekly demand aggregation.

Input:
    data/synthetic/sales_transactions.csv
    data/synthetic/anomaly_log.csv (optional; used for traceability)

Outputs:
    data/processed/standardized_sales_transactions.csv
    data/processed/excluded_sales_transactions.csv
    reports/data_standardization/standardization_report.json

Usage:
    python src/standardize_sales_data.py

Notes:
    - Source files are never modified.
    - Negative quantities are excluded pending business clarification.
    - Missing customer_id does not exclude a row from SKU-level demand analysis.
    - The first eligible occurrence of a duplicate business key is retained.
    - Only posted transactions are included in the standardized output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# 1. Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "synthetic" / "sales_transactions.csv"
ANOMALY_LOG_FILE = PROJECT_ROOT / "data" / "synthetic" / "anomaly_log.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_standardization"

OUTPUT_FILE = PROCESSED_DIR / "standardized_sales_transactions.csv"
EXCLUDED_FILE = PROCESSED_DIR / "excluded_sales_transactions.csv"
REPORT_FILE = REPORT_DIR / "standardization_report.json"

DATE_COL = "transaction_date"
LINE_ID_COL = "invoice_line_id"
INVOICE_ID_COL = "invoice_id"
SKU_COL = "sku"
WAREHOUSE_COL = "warehouse_id"
CUSTOMER_COL = "customer_id"
QUANTITY_COL = "quantity"
UNIT_PRICE_COL = "unit_price"
LINE_AMOUNT_COL = "line_amount"
STATUS_COL = "transaction_status"

REQUIRED_COLUMNS = [
    LINE_ID_COL, INVOICE_ID_COL, DATE_COL, SKU_COL, WAREHOUSE_COL,
    CUSTOMER_COL, QUANTITY_COL, UNIT_PRICE_COL, LINE_AMOUNT_COL, STATUS_COL,
]
BUSINESS_KEY = [
    INVOICE_ID_COL, SKU_COL, WAREHOUSE_COL, DATE_COL, QUANTITY_COL,
]
AMOUNT_TOLERANCE = 0.01
INCLUDED_STATUS = "posted"


# 2. Input loading and validation
def load_sales_data(path: Path) -> pd.DataFrame:
    """Load sales data and validate the expected schema."""
    if not path.exists():
        raise FileNotFoundError(f"Sales input file was not found: {path}")

    data = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Sales input is missing required columns: "
            + ", ".join(missing_columns)
        )

    data = data.copy()
    data["source_row_number"] = np.arange(2, len(data) + 2)
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")

    for column in [QUANTITY_COL, UNIT_PRICE_COL, LINE_AMOUNT_COL]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def load_anomaly_log(path: Path) -> pd.DataFrame:
    """Load the anomaly log when available; otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# 3. Standardization rules
def standardize_sales_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Separate included and excluded rows and record exclusion reasons."""
    working = data.copy()
    exclusion_reasons = pd.Series("", index=working.index, dtype="string")

    def add_exclusion(mask: pd.Series, reason: str) -> None:
        """Append an exclusion reason without overwriting prior reasons."""
        mask = mask.fillna(False)
        exclusion_reasons.loc[mask] = exclusion_reasons.loc[mask].map(
            lambda value: reason if not value else f"{value}; {reason}"
        )

    for column in [LINE_ID_COL, INVOICE_ID_COL, SKU_COL, WAREHOUSE_COL]:
        missing = (
            working[column].isna()
            | working[column].astype("string").str.strip().eq("")
        )
        add_exclusion(missing, f"missing_required_{column}")

    add_exclusion(
        working[DATE_COL].isna(), "invalid_or_missing_transaction_date"
    )
    add_exclusion(
        working[QUANTITY_COL].isna(), "missing_or_invalid_quantity"
    )
    add_exclusion(
        working[QUANTITY_COL] < 0,
        "negative_quantity_requires_business_review",
    )
    add_exclusion(
        working[UNIT_PRICE_COL].isna() | (working[UNIT_PRICE_COL] < 0),
        "missing_invalid_or_negative_unit_price",
    )
    add_exclusion(
        working[LINE_AMOUNT_COL].isna() | (working[LINE_AMOUNT_COL] < 0),
        "missing_invalid_or_negative_line_amount",
    )

    expected_amount = working[QUANTITY_COL] * working[UNIT_PRICE_COL]
    amount_matches = np.isclose(
        expected_amount.to_numpy(dtype=float),
        working[LINE_AMOUNT_COL].to_numpy(dtype=float),
        atol=AMOUNT_TOLERANCE,
        rtol=0,
        equal_nan=False,
    )
    amount_mismatch = (
        working[QUANTITY_COL].notna()
        & (working[QUANTITY_COL] >= 0)
        & working[UNIT_PRICE_COL].notna()
        & working[LINE_AMOUNT_COL].notna()
        & ~amount_matches
    )
    add_exclusion(
        pd.Series(amount_mismatch, index=working.index),
        "line_amount_mismatch",
    )

    normalized_status = (
        working[STATUS_COL].astype("string").str.strip().str.lower()
    )
    add_exclusion(
        normalized_status.isna() | normalized_status.ne(INCLUDED_STATUS),
        f"transaction_status_not_{INCLUDED_STATUS}",
    )

    # Detect duplicate business records among otherwise eligible rows.
    eligible_before_dedup = exclusion_reasons.eq("")
    duplicate_candidates = pd.Series(False, index=working.index)
    eligible_rows = working.loc[eligible_before_dedup]
    duplicate_candidates.loc[eligible_before_dedup] = eligible_rows.duplicated(
        subset=BUSINESS_KEY, keep="first"
    )
    add_exclusion(
        duplicate_candidates, "duplicate_business_record_candidate"
    )

    excluded_mask = exclusion_reasons.ne("")
    excluded = working.loc[excluded_mask].copy()
    excluded["exclusion_reason"] = exclusion_reasons.loc[excluded_mask]

    standardized = working.loc[~excluded_mask].copy()
    standardized = standardized.sort_values(
        [DATE_COL, SKU_COL, WAREHOUSE_COL, LINE_ID_COL]
    ).reset_index(drop=True)
    standardized = standardized.drop(columns=["source_row_number"])
    excluded = excluded.sort_values("source_row_number").reset_index(drop=True)

    missing_customer_mask = (
        standardized[CUSTOMER_COL].isna()
        | standardized[CUSTOMER_COL].astype("string").str.strip().eq("")
    )
    reason_counts: dict[str, int] = {}
    for value in exclusion_reasons:
        if value:
            for reason in value.split("; "):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    report = {
        "input_file": INPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "standardized_output": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "excluded_output": EXCLUDED_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "input_rows": int(len(working)),
        "standardized_rows": int(len(standardized)),
        "excluded_rows": int(len(excluded)),
        "missing_customer_id_rows_retained_if_otherwise_valid": int(
            missing_customer_mask.sum()
        ),
        "standardized_negative_quantity_rows": int(
            (standardized[QUANTITY_COL] < 0).sum()
        ),
        "standardized_amount_mismatch_rows": 0,
        "included_status": INCLUDED_STATUS,
        "business_key_for_duplicate_detection": BUSINESS_KEY,
        "exclusion_reason_counts": reason_counts,
        "decisions": [
            "Source data was not modified.",
            "Negative quantities were excluded pending business clarification.",
            "Missing customer_id was not an exclusion criterion for SKU-level demand.",
            "The first eligible occurrence of a duplicate business key was retained.",
            f"Only '{INCLUDED_STATUS}' transactions were included.",
        ],
    }
    return standardized, excluded, report


# 4. Output reporting
def save_outputs(
    standardized: pd.DataFrame,
    excluded: pd.DataFrame,
    report: dict[str, Any],
    anomaly_log: pd.DataFrame,
) -> None:
    """Write standardized data, excluded records, and a JSON report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    standardized.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")
    excluded.to_csv(EXCLUDED_FILE, index=False, date_format="%Y-%m-%d")
    report["anomaly_log_available"] = not anomaly_log.empty
    if not anomaly_log.empty:
        report["anomaly_log_rows"] = int(len(anomaly_log))

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)


# 5. Main workflow
def main() -> None:
    """Run the sales data standardization workflow."""
    sales_data = load_sales_data(INPUT_FILE)
    anomaly_log = load_anomaly_log(ANOMALY_LOG_FILE)
    standardized, excluded, report = standardize_sales_data(sales_data)
    save_outputs(standardized, excluded, report, anomaly_log)

    print("\n" + "=" * 65)
    print("SALES DATA STANDARDIZATION")
    print("=" * 65)
    print(f"Input rows          : {report['input_rows']:,}")
    print(f"Standardized rows   : {report['standardized_rows']:,}")
    print(f"Excluded rows       : {report['excluded_rows']:,}")
    print(
        "Missing customer IDs retained: "
        f"{report['missing_customer_id_rows_retained_if_otherwise_valid']:,}"
    )
    print(f"Output file         : {OUTPUT_FILE}")
    print(f"Excluded records    : {EXCLUDED_FILE}")
    print(f"Report              : {REPORT_FILE}")
    print("\nExclusion reasons:")
    for reason, count in report["exclusion_reason_counts"].items():
        print(f"- {reason}: {count:,}")
    print("\nStandardization completed. Source data was not modified.")


if __name__ == "__main__":
    main()
