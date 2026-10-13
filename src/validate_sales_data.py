"""
Sales Data Quality Assessment

Purpose:
    Identify missing values, duplicate identifiers and business records,
    invalid values, calculation inconsistencies, and transaction-status issues.
    Reconcile detected sales issues with the controlled anomaly log.

Input:
    data/synthetic/sales_transactions.csv
    data/synthetic/anomaly_log.csv (optional)

Outputs:
    reports/data_quality_assessment/
        data_quality_report.json
        column_missingness.csv
        data_quality_issues.csv
        transaction_status_distribution.csv
        anomaly_reconciliation.csv

Usage:
    python src/validate_sales_data.py

Notes:
    This script identifies and documents potential data-quality issues.
    It does not modify the source datasets or automatically remove records.
    Repeated invoice_id values are not automatically considered duplicates,
    because an invoice may legitimately contain multiple lines.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# 1. Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "synthetic" / "sales_transactions.csv"
ANOMALY_LOG_FILE = PROJECT_ROOT / "data" / "synthetic" / "anomaly_log.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "data_quality_assessment"

DATE_COL = "transaction_date"
SKU_COL = "sku"
QUANTITY_COL = "quantity"
UNIT_PRICE_COL = "unit_price"
LINE_AMOUNT_COL = "line_amount"
LINE_ID_COL = "invoice_line_id"
INVOICE_ID_COL = "invoice_id"
STATUS_COL = "transaction_status"

REQUIRED_COLUMNS = [
    LINE_ID_COL,
    INVOICE_ID_COL,
    DATE_COL,
    SKU_COL,
    "warehouse_id",
    "customer_id",
    QUANTITY_COL,
    UNIT_PRICE_COL,
    LINE_AMOUNT_COL,
    STATUS_COL,
]

EXPECTED_STATUSES = {"posted", "pending", "cancelled", "returned"}
AMOUNT_TOLERANCE = 0.01


# 2. Data loading and validation
def load_sales_data(file_path: Path) -> pd.DataFrame:
    """Load the sales dataset and validate required columns."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Sales dataset not found: {file_path}\n"
            "Check that the synthetic data generator has been run."
        )

    data = pd.read_csv(file_path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}\n"
            f"Available columns: {data.columns.tolist()}"
        )

    return data


def load_anomaly_log(file_path: Path) -> pd.DataFrame:
    """Load the controlled anomaly log when it is available."""
    if not file_path.exists():
        return pd.DataFrame(
            columns=[
                "anomaly_id",
                "dataset",
                "record_id",
                "field",
                "anomaly_type",
                "description",
            ]
        )

    return pd.read_csv(file_path)


# 3. Issue collection helpers
def add_issue(
    issues: list[dict[str, Any]],
    data: pd.DataFrame,
    mask: pd.Series,
    rule: str,
    severity: str,
    description: str,
    field: str,
) -> None:
    """Append one issue record for every row matching a rule."""
    matched_indices = data.index[mask.fillna(False)]

    for index in matched_indices:
        row = data.loc[index]
        issues.append(
            {
                "source_row_number": int(index) + 2,
                "invoice_line_id": row.get(LINE_ID_COL),
                "invoice_id": row.get(INVOICE_ID_COL),
                "sku": row.get(SKU_COL),
                "field": field,
                "rule": rule,
                "severity": severity,
                "description": description,
            }
        )


def build_missingness_report(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values and original data types for each column."""
    row_count = len(data)
    report = pd.DataFrame(
        {
            "column": data.columns,
            "data_type": [str(data[column].dtype) for column in data.columns],
            "missing_count": [int(data[column].isna().sum()) for column in data.columns],
        }
    )
    report["missing_percent"] = (
        (report["missing_count"] / row_count * 100).round(2)
        if row_count
        else 0.0
    )
    return report.sort_values(
        ["missing_count", "column"], ascending=[False, True]
    ).reset_index(drop=True)


# 4. Data-quality assessment
def assess_data_quality(data: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect data-quality issues without changing the input DataFrame."""
    issues: list[dict[str, Any]] = []
    working = data.copy()

    parsed_dates = pd.to_datetime(working[DATE_COL], errors="coerce")
    numeric_columns = [QUANTITY_COL, UNIT_PRICE_COL, LINE_AMOUNT_COL]
    numeric_values = {
        column: pd.to_numeric(working[column], errors="coerce")
        for column in numeric_columns
    }

    # Missing identifiers and required business fields
    for column in [LINE_ID_COL, INVOICE_ID_COL, SKU_COL, "warehouse_id", "customer_id"]:
        add_issue(
            issues,
            working,
            working[column].isna() | working[column].astype("string").str.strip().eq(""),
            "missing_identifier",
            "high",
            f"Required identifier '{column}' is missing or blank.",
            column,
        )

    add_issue(
        issues,
        working,
        parsed_dates.isna(),
        "invalid_or_missing_date",
        "high",
        "Transaction date is missing or cannot be parsed as a date.",
        DATE_COL,
    )

    for column, values in numeric_values.items():
        add_issue(
            issues,
            working,
            values.isna(),
            "missing_or_invalid_numeric",
            "high",
            f"'{column}' is missing or cannot be converted to a numeric value.",
            column,
        )

    # Identifier duplicates: invoice_line_id is expected to be unique.
    duplicate_line_id = (
        working[LINE_ID_COL].notna()
        & working[LINE_ID_COL].duplicated(keep=False)
    )
    add_issue(
        issues,
        working,
        duplicate_line_id,
        "duplicate_invoice_line_id",
        "high",
        "Invoice-line identifier appears more than once.",
        LINE_ID_COL,
    )

    # A copied business record can have a new line ID, so check key business fields too.
    business_key = [INVOICE_ID_COL, SKU_COL, "warehouse_id", DATE_COL, QUANTITY_COL]
    duplicate_business_record = working.duplicated(
        subset=business_key, keep=False
    )
    add_issue(
        issues,
        working,
        duplicate_business_record,
        "duplicate_business_record_candidate",
        "medium",
        "Business fields match another row. Review before treating it as a duplicate.",
        ", ".join(business_key),
    )

    # Quantity, price, and amount rules
    quantity = numeric_values[QUANTITY_COL]
    unit_price = numeric_values[UNIT_PRICE_COL]
    line_amount = numeric_values[LINE_AMOUNT_COL]

    add_issue(
        issues,
        working,
        quantity < 0,
        "negative_quantity",
        "medium",
        "Quantity is negative; determine whether this represents a return or invalid value.",
        QUANTITY_COL,
    )
    add_issue(
        issues,
        working,
        quantity == 0,
        "zero_quantity",
        "medium",
        "Quantity is zero; confirm whether zero-quantity transaction lines are valid.",
        QUANTITY_COL,
    )
    add_issue(
        issues,
        working,
        unit_price < 0,
        "negative_unit_price",
        "high",
        "Unit price is negative.",
        UNIT_PRICE_COL,
    )
    add_issue(
        issues,
        working,
        line_amount < 0,
        "negative_line_amount",
        "high",
        "Line amount is negative.",
        LINE_AMOUNT_COL,
    )

    expected_amount = quantity * unit_price
    valid_quantity_mask = quantity >= 0
    amount_mismatch = (
        quantity.notna()
        & unit_price.notna()
        & line_amount.notna()
        & valid_quantity_mask
        & ~np.isclose(
            expected_amount.to_numpy(dtype=float),
            line_amount.to_numpy(dtype=float),
            atol=AMOUNT_TOLERANCE,
            rtol=0,
            equal_nan=False,
        )
    )
    
    add_issue(
        issues,
        working,
        pd.Series(amount_mismatch, index=working.index),
        "line_amount_mismatch",
        "high",
        "line_amount differs from quantity multiplied by unit_price beyond the allowed tolerance.",
        LINE_AMOUNT_COL,
    )

    # Status values: report unexpected values but do not assume business meaning.
    normalized_status = working[STATUS_COL].astype("string").str.strip().str.lower()
    missing_status = normalized_status.isna() | normalized_status.eq("")
    add_issue(
        issues,
        working,
        missing_status,
        "missing_transaction_status",
        "high",
        "Transaction status is missing or blank.",
        STATUS_COL,
    )

    unexpected_status = (
        normalized_status.notna()
        & normalized_status.ne("")
        & ~normalized_status.isin(EXPECTED_STATUSES)
    )
    add_issue(
        issues,
        working,
        unexpected_status,
        "unexpected_transaction_status",
        "medium",
        f"Status is outside the configured reference set: {sorted(EXPECTED_STATUSES)}.",
        STATUS_COL,
    )

    # Entire-row duplicates are a separate diagnostic from business-key candidates.
    full_duplicate = working.duplicated(keep=False)
    add_issue(
        issues,
        working,
        full_duplicate,
        "duplicate_full_row",
        "high",
        "All source columns match another row.",
        "all_columns",
    )

    issue_columns = [
        "source_row_number",
        LINE_ID_COL,
        INVOICE_ID_COL,
        SKU_COL,
        "field",
        "rule",
        "severity",
        "description",
    ]
    issues_df = pd.DataFrame(issues, columns=issue_columns)

    status_distribution = (
        normalized_status.fillna("<MISSING>")
        .replace("", "<BLANK>")
        .value_counts(dropna=False)
        .rename_axis("transaction_status")
        .reset_index(name="row_count")
    )
    status_distribution["percentage"] = (
        status_distribution["row_count"] / len(working) * 100
    ).round(2) if len(working) else 0.0

    summary = {
        "dataset": INPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "row_count": int(len(working)),
        "column_count": int(len(working.columns)),
        "missing_cells": int(working.isna().sum().sum()),
        "duplicate_invoice_line_id_rows": int(duplicate_line_id.sum()),
        "duplicate_business_record_candidate_rows": int(duplicate_business_record.sum()),
        "duplicate_full_row_count": int(full_duplicate.sum()),
        "invalid_or_missing_date_rows": int(parsed_dates.isna().sum()),
        "missing_or_invalid_quantity_rows": int(quantity.isna().sum()),
        "negative_quantity_rows": int((quantity < 0).sum()),
        "zero_quantity_rows": int((quantity == 0).sum()),
        "negative_unit_price_rows": int((unit_price < 0).sum()),
        "negative_line_amount_rows": int((line_amount < 0).sum()),
        "line_amount_mismatch_rows": int(np.asarray(amount_mismatch).sum()),
        "missing_or_blank_status_rows": int(missing_status.sum()),
        "unexpected_status_rows": int(unexpected_status.sum()),
        "issue_record_count": int(len(issues_df)),
        "expected_status_values": sorted(EXPECTED_STATUSES),
        "note": (
            "Issue counts are rule-level counts; one source row may trigger "
            "more than one rule. Review context before removing or correcting records."
        ),
    }

    if not issues_df.empty:
        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues_df["_severity_order"] = issues_df["severity"].map(severity_order)
        issues_df = (
            issues_df.sort_values(
                ["_severity_order", "rule", "source_row_number"]
            )
            .drop(columns="_severity_order")
            .reset_index(drop=True)
        )

    return issues_df.to_dict(orient="records"), {
        "summary": summary,
        "status_distribution": status_distribution,
        "issues_df": issues_df,
    }


# 5. Controlled anomaly reconciliation
def reconcile_anomaly_log(
    issues_df: pd.DataFrame, anomaly_log: pd.DataFrame
) -> pd.DataFrame:
    """Compare logged sales anomalies with detected sales issues by record ID."""
    columns = [
        "anomaly_id",
        "dataset",
        "record_id",
        "field",
        "anomaly_type",
        "description",
        "detected_rules",
        "reconciliation_status",
    ]

    if anomaly_log.empty:
        return pd.DataFrame(columns=columns)

    sales_anomalies = anomaly_log.loc[
        anomaly_log["dataset"].eq("sales_transactions")
    ].copy()

    if sales_anomalies.empty:
        return pd.DataFrame(columns=columns)

    detected_by_id: dict[str, list[str]] = {}
    if not issues_df.empty:
        for _, issue in issues_df.iterrows():
            record_id = issue.get(LINE_ID_COL)
            if pd.notna(record_id):
                detected_by_id.setdefault(str(record_id), []).append(str(issue["rule"]))

    rows = []
    for _, anomaly in sales_anomalies.iterrows():
        record_id = str(anomaly.get("record_id", ""))
        detected_rules = sorted(set(detected_by_id.get(record_id, [])))
        rows.append(
            {
                "anomaly_id": anomaly.get("anomaly_id"),
                "dataset": anomaly.get("dataset"),
                "record_id": record_id,
                "field": anomaly.get("field"),
                "anomaly_type": anomaly.get("anomaly_type"),
                "description": anomaly.get("description"),
                "detected_rules": "; ".join(detected_rules),
                "reconciliation_status": (
                    "detected" if detected_rules else "not_detected_by_current_rules"
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns)


# 6. Report generation
def save_reports(
    missingness: pd.DataFrame,
    issues_df: pd.DataFrame,
    status_distribution: pd.DataFrame,
    anomaly_reconciliation: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Save data-quality outputs without modifying source data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missingness.to_csv(OUTPUT_DIR / "column_missingness.csv", index=False)
    issues_df.to_csv(OUTPUT_DIR / "data_quality_issues.csv", index=False)
    status_distribution.to_csv(
        OUTPUT_DIR / "transaction_status_distribution.csv", index=False
    )
    anomaly_reconciliation.to_csv(
        OUTPUT_DIR / "anomaly_reconciliation.csv", index=False
    )

    report = {
        "assessment": "Sales Data Quality Assessment",
        "synthetic_data": True,
        "summary": summary,
        "outputs": [
            "column_missingness.csv",
            "data_quality_issues.csv",
            "transaction_status_distribution.csv",
            "anomaly_reconciliation.csv",
        ],
        "source_data_modified": False,
    }
    report_path = OUTPUT_DIR / "data_quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("SALES DATA QUALITY ASSESSMENT")
    print("=" * 65)
    print(f"Input file             : {INPUT_FILE}")
    print(f"Rows assessed          : {summary['row_count']:,}")
    print(f"Columns assessed       : {summary['column_count']:,}")
    print(f"Missing cells          : {summary['missing_cells']:,}")
    print(f"Issue records          : {summary['issue_record_count']:,}")
    print(f"Negative quantity rows : {summary['negative_quantity_rows']:,}")
    print(f"Amount mismatch rows   : {summary['line_amount_mismatch_rows']:,}")
    print(f"Output directory       : {OUTPUT_DIR}")
    print("\nOutput files:")
    for filename in [
        "data_quality_report.json",
        "column_missingness.csv",
        "data_quality_issues.csv",
        "transaction_status_distribution.csv",
        "anomaly_reconciliation.csv",
    ]:
        print(f"- {filename}")
    print("\nAssessment completed. Source data was not modified.")


# 7. Main workflow
def main() -> None:
    """Run the complete sales data-quality assessment."""
    data = load_sales_data(INPUT_FILE)
    anomaly_log = load_anomaly_log(ANOMALY_LOG_FILE)

    missingness = build_missingness_report(data)
    _, assessment = assess_data_quality(data)

    issues_df = assessment["issues_df"]
    anomaly_reconciliation = reconcile_anomaly_log(issues_df, anomaly_log)

    summary = assessment["summary"]
    summary["anomaly_log_available"] = not anomaly_log.empty
    summary["sales_anomalies_logged"] = int(
        anomaly_log["dataset"].eq("sales_transactions").sum()
    ) if not anomaly_log.empty and "dataset" in anomaly_log.columns else 0
    summary["logged_sales_anomalies_detected"] = int(
        anomaly_reconciliation["reconciliation_status"].eq("detected").sum()
    ) if not anomaly_reconciliation.empty else 0

    save_reports(
        missingness=missingness,
        issues_df=issues_df,
        status_distribution=assessment["status_distribution"],
        anomaly_reconciliation=anomaly_reconciliation,
        summary=summary,
    )


if __name__ == "__main__":
    main()
