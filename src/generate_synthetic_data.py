"""
Generate reproducible synthetic datasets for the
Demand Intelligence & Inventory Risk Engine.

Purpose:
    Generate relational datasets for demand forecasting, inventory risk,
    quotation intelligence, and subscription renewal analysis. Controlled
    data-quality anomalies are injected and documented in anomaly_log.csv.

Input:
    No external input files are required.

Outputs:
    data/synthetic/
        products.csv
        warehouses.csv
        customers.csv
        sales_transactions.csv
        inventory_snapshots.csv
        purchase_orders.csv
        sales_quotations.csv
        quotation_lines.csv
        subscription_records.csv
        anomaly_log.csv
        generation_report.json

Usage:
    python src/generate_synthetic_data.py
    python src/generate_synthetic_data.py --output-dir data/synthetic --seed 42

Notes:
    All records are synthetic. Review anomaly_log.csv before cleaning or
    modeling. The generator validates referential integrity before saving.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Configuration
DEFAULT_SEED = 42
DEFAULT_START_DATE = "2023-10-01"
DEFAULT_END_DATE = "2026-09-30"
DEFAULT_PRODUCTS = 100
DEFAULT_WAREHOUSES = 3
DEFAULT_CUSTOMERS = 250
CATEGORIES = ["Networking", "Cabling", "Network Security", "Load Balancer"]
DEMAND_TYPES = ["stable", "trend", "seasonal", "intermittent", "declining"]
DEMAND_TYPE_PROBS = [0.30, 0.20, 0.22, 0.18, 0.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory (relative to current working directory by default).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCTS)
    parser.add_argument("--warehouses", type=int, default=DEFAULT_WAREHOUSES)
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMERS)
    return parser.parse_args()


def as_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def make_master_data(rng: np.random.Generator, n_products: int, n_warehouses: int, n_customers: int):
    product_ids = [f"SKU-{i:04d}" for i in range(1, n_products + 1)]
    products = pd.DataFrame({
        "sku": product_ids,
        "product_name": [f"Network Product {i:03d}" for i in range(1, n_products + 1)],
        "category": rng.choice(CATEGORIES, n_products),
        "product_type": rng.choice(["hardware", "software_license", "subscription"], n_products, p=[0.70, 0.18, 0.12]),
        "unit_cost": np.round(rng.uniform(25, 3500, n_products), 2),
        "unit_price": np.round(rng.uniform(50, 7000, n_products), 2),
        "demand_type": rng.choice(DEMAND_TYPES, n_products, p=DEMAND_TYPE_PROBS),
        "active_flag": True,
    })
    products["unit_price"] = np.maximum(products["unit_price"], products["unit_cost"] * 1.15).round(2)
    warehouses = pd.DataFrame({
        "warehouse_id": [
            f"WH-{i:02d}" for i in range(1, n_warehouses + 1)
        ],
        "warehouse_name": [
            f"Warehouse {i}" for i in range(1, n_warehouses + 1)
        ],
    })

    customer_segments = ["SMB", "Mid-Market", "Enterprise", "Public Sector"]
    industries = ["Technology", "Manufacturing", "Financial Services", "Retail", "Education", "Healthcare"]
    customers = pd.DataFrame({
        "customer_id": [f"CUS-{i:05d}" for i in range(1, n_customers + 1)],
        "customer_name": [f"Synthetic Customer {i:03d}" for i in range(1, n_customers + 1)],
        "segment": rng.choice(customer_segments, n_customers, p=[0.35, 0.30, 0.25, 0.10]),
        "industry": rng.choice(industries, n_customers),
        "country": "Indonesia",
    })
    return products, warehouses, customers


def generate_sales(rng: np.random.Generator, products: pd.DataFrame, warehouses: pd.DataFrame, customers: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Weekly demand is simulated, then split into invoice lines. Weeks with no transaction
    # remain absent in raw transactions and are represented later during weekly aggregation.
    week_ends = pd.date_range(start=start, end=end, freq="W-SUN")
    rows: list[dict[str, Any]] = []
    invoice_line_counter = 1
    customer_ids = customers["customer_id"].to_numpy()
    warehouse_ids = warehouses["warehouse_id"].to_numpy()
    for p in products.itertuples(index=False):
        base = float(rng.uniform(4, 35))
        for wi, week_end in enumerate(week_ends):
            elapsed_years = wi / 52.0
            trend = 1 + (0.16 * elapsed_years if p.demand_type == "trend" else 0)
            if p.demand_type == "declining":
                trend = max(0.45, 1 - 0.18 * elapsed_years)
            seasonal = 1 + 0.30 * np.sin(2 * np.pi * (wi - 8) / 52)
            promo_factor = 1.0
            # Controlled business events are deterministic for reproducibility.
            if week_end.month in (11, 12):
                promo_factor = 1.18
            expected = base
            if p.demand_type in ("trend", "declining"):
                expected *= trend
            if p.demand_type == "seasonal":
                expected *= seasonal
            if p.demand_type == "intermittent":
                if rng.random() > 0.28:
                    continue
                expected *= rng.uniform(0.35, 1.7)
            expected *= promo_factor
            weekly_qty = int(rng.poisson(max(expected, 0.2)))
            if weekly_qty <= 0:
                continue
            # Allocate weekly demand across one to four invoice lines.
            n_lines = int(rng.integers(1, min(4, weekly_qty) + 1))
            quantities = rng.multinomial(weekly_qty, np.repeat(1 / n_lines, n_lines))
            for qty in quantities:
                if qty <= 0:
                    continue
                tx_date = week_end - pd.Timedelta(days=int(rng.integers(0, 7)))
                tx_date = max(start, tx_date)
                warehouse_id = str(rng.choice(warehouse_ids))
                customer_id = str(rng.choice(customer_ids))                
                unit_price = round(
                    float(p.unit_price * rng.uniform(0.96, 1.04)),
                    2,
                )

                rows.append({
                    "invoice_line_id": f"SIL-{invoice_line_counter:08d}",
                    "invoice_id": f"INV-{invoice_line_counter:07d}",
                    "transaction_date": tx_date,
                    "sku": p.sku,
                    "warehouse_id": warehouse_id,
                    "customer_id": customer_id,
                    "quantity": int(qty),
                    "unit_price": unit_price,
                    "line_amount": round(unit_price * int(qty), 2),
                    "transaction_status": "posted",
                })
                invoice_line_counter += 1
    cols = ["invoice_line_id", "invoice_id", "transaction_date", "sku", "warehouse_id", "customer_id", "quantity", "unit_price", "line_amount", "transaction_status"]
    return pd.DataFrame(rows, columns=cols).sort_values("transaction_date").reset_index(drop=True)


def generate_inventory(rng: np.random.Generator, products: pd.DataFrame, warehouses: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    # Monthly snapshots support history-level exploration; current stock is a synthetic snapshot,
    # not a claim of live stock availability.
    snapshot_dates = pd.date_range(start=end - pd.DateOffset(months=11), end=end, freq="MS")
    snapshot_dates = snapshot_dates.union(pd.DatetimeIndex([end])).sort_values()
    rows = []
    for sku in products["sku"]:
        for wh in warehouses["warehouse_id"]:
            starting = int(rng.integers(25, 500))
            for snap_date in snapshot_dates:
                on_hand = max(0, int(starting + rng.normal(0, max(5, starting * 0.12))))
                reserved = int(rng.integers(0, min(on_hand, 30) + 1)) if on_hand > 0 else 0
                rows.append({
                    "snapshot_date": snap_date,
                    "sku": sku,
                    "warehouse_id": wh,
                    "on_hand_quantity": on_hand,
                    "reserved_quantity": reserved,
                })
                starting = max(0, on_hand + int(rng.integers(-35, 36)))
    return pd.DataFrame(rows)


def generate_purchase_orders(rng: np.random.Generator, products: pd.DataFrame, warehouses: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    n = max(180, len(products) * 3)
    rows = []
    sku_ids = products["sku"].to_numpy()
    wh_ids = warehouses["warehouse_id"].to_numpy()
    statuses = ["open", "partial", "received", "cancelled"]
    probs = [0.28, 0.16, 0.46, 0.10]
    for i in range(1, n + 1):
        order_date = start + pd.Timedelta(days=int(rng.integers(0, max(1, (end - start).days - 45))))
        lead_days = int(rng.integers(14, 61))
        expected = order_date + pd.Timedelta(days=lead_days)
        status = str(rng.choice(statuses, p=probs))
        received_qty = 0
        if status == "received":
            received_qty = int(rng.integers(10, 301))
        elif status == "partial":
            received_qty = int(rng.integers(1, 100))
        quantity = int(rng.integers(max(20, received_qty + 1), max(21, received_qty + 251)))
        actual = expected + pd.Timedelta(days=int(rng.integers(-7, 22))) if status == "received" else pd.NaT
        rows.append({
            "po_line_id": f"POL-{i:06d}",
            "po_id": f"PO-{i:05d}",
            "order_date": order_date,
            "sku": str(rng.choice(sku_ids)),
            "warehouse_id": str(rng.choice(wh_ids)),
            "ordered_quantity": quantity,
            "received_quantity": min(received_qty, quantity),
            "status": status,
            "expected_receipt_date": expected,
            "actual_receipt_date": actual,
        })
    # Include an explicit overdue open PO to ensure this business condition is represented.
    rows[0]["status"] = "open"
    rows[0]["expected_receipt_date"] = end - pd.Timedelta(days=10)
    rows[0]["actual_receipt_date"] = pd.NaT
    return pd.DataFrame(rows)


def generate_quotations(rng: np.random.Generator, products: pd.DataFrame, customers: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    n_quotes = 500
    customer_ids = customers["customer_id"].to_numpy()
    status_values = ["won", "lost", "open"]
    status_probs = [0.42, 0.33, 0.25]
    quote_rows, line_rows = [], []
    line_counter = 1
    product_records = list(products.itertuples(index=False))
    for i in range(1, n_quotes + 1):
        quote_date = start + pd.Timedelta(days=int(rng.integers(0, (end - start).days + 1)))
        status = str(rng.choice(status_values, p=status_probs))
        submitted = quote_date + pd.Timedelta(hours=int(rng.integers(0, 9)))
        approval_hours = int(max(1, rng.lognormal(mean=2.3, sigma=0.7)))
        # Missing approval timestamp is injected later as a controlled anomaly, not in base data.
        approval_ts = submitted + pd.Timedelta(hours=approval_hours)
        close_date = quote_date + pd.Timedelta(days=int(rng.integers(1, 45))) if status != "open" else pd.NaT
        quote_rows.append({
            "quotation_id": f"QT-{i:05d}",
            "customer_id": str(rng.choice(customer_ids)),
            "quotation_date": quote_date,
            "submitted_at": submitted,
            "approval_at": approval_ts,
            "status": status,
            "close_date": close_date,
            "currency": "USD",
        })
        n_lines = int(rng.integers(1, 5))
        chosen_indices = rng.choice(len(product_records), size=n_lines, replace=False)
        chosen = [product_records[int(idx)] for idx in chosen_indices]
        for p in chosen:
            qty = int(rng.integers(1, 31))
            price = round(float(p.unit_price * rng.uniform(0.90, 1.10)), 2)
            line_rows.append({
                "quotation_line_id": f"QTL-{line_counter:06d}",
                "quotation_id": f"QT-{i:05d}",
                "sku": p.sku,
                "quantity": qty,
                "unit_price": price,
                "line_amount": round(qty * price, 2),
            })
            line_counter += 1
    return pd.DataFrame(quote_rows), pd.DataFrame(line_rows)


def generate_subscriptions(rng: np.random.Generator, products: pd.DataFrame, customers: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sub_products = products.loc[products["product_type"].isin(["software_license", "subscription"]), "sku"].to_numpy()
    if len(sub_products) == 0:
        sub_products = products["sku"].to_numpy()
    n = 300
    statuses = ["active", "renewed", "cancelled", "expired"]
    probs = [0.52, 0.20, 0.12, 0.16]
    rows = []
    for i in range(1, n + 1):
        start_date = start + pd.Timedelta(days=int(rng.integers(0, max(1, (end - start).days - 365))))
        term_months = int(rng.choice([12, 24, 36], p=[0.70, 0.20, 0.10]))
        renewal_date = start_date + pd.DateOffset(months=term_months)
        status = str(rng.choice(statuses, p=probs))
        rows.append({
            "subscription_id": f"SUB-{i:05d}",
            "customer_id": str(rng.choice(customers["customer_id"].to_numpy())),
            "sku": str(rng.choice(sub_products)),
            "contract_start_date": start_date,
            "renewal_date": renewal_date,
            "contract_term_months": term_months,
            "annual_contract_value": round(float(rng.uniform(500, 25000)), 2),
            "status": status,
        })
    return pd.DataFrame(rows)


def inject_controlled_anomalies(datasets: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    """Inject known data-quality anomalies and record exactly what was changed."""
    log: list[dict[str, Any]] = []
    def record(dataset: str, record_id: str, field: str, anomaly_type: str, description: str):
        log.append({
            "anomaly_id": f"ANOM-{len(log) + 1:04d}",
            "dataset": dataset,
            "record_id": record_id,
            "field": field,
            "anomaly_type": anomaly_type,
            "description": description,
        })
    # Duplicate candidate: append a copy of an existing invoice line with its own original ID retained.
    sales = datasets["sales_transactions"].copy()
    if len(sales) > 10:
        source = sales.iloc[5].copy()
        duplicate = source.copy()
        duplicate["invoice_line_id"] = f"SIL-DUP-{source['invoice_line_id']}"
        sales = pd.concat([sales, pd.DataFrame([duplicate])], ignore_index=True)
        record("sales_transactions", duplicate["invoice_line_id"], "invoice_id,sku,warehouse_id,transaction_date,quantity", "duplicate_candidate", "Copied an existing transaction's business fields with a new line ID; review before removal.")
    # Missing customer reference on one transaction (a data-quality anomaly).
    if len(sales) > 20:
        idx = int(sales.index[20])
        sales.loc[idx, "customer_id"] = pd.NA
        record("sales_transactions", sales.loc[idx, "invoice_line_id"], "customer_id", "missing_value", "Customer reference intentionally missing.")
    # Invalid negative quantity on one separate row.
    if len(sales) > 30:
        idx = int(sales.index[30])
        sales.loc[idx, "quantity"] = -abs(int(sales.loc[idx, "quantity"]))
        record("sales_transactions", sales.loc[idx, "invoice_line_id"], "quantity", "invalid_value", "Negative quantity intentionally injected; determine business treatment before modeling.")
    datasets["sales_transactions"] = sales
    # Missing approval timestamp for one quotation.
    quotes = datasets["sales_quotations"].copy()
    if len(quotes) > 5:
        idx = 4
        quotes.loc[idx, "approval_at"] = pd.NaT
        record("sales_quotations", quotes.loc[idx, "quotation_id"], "approval_at", "missing_value", "Approval timestamp intentionally missing; exclude from direct turnaround calculation until resolved.")
    datasets["sales_quotations"] = quotes
    # Invalid reserved quantity above on-hand for one inventory snapshot.
    inventory = datasets["inventory_snapshots"].copy()
    if len(inventory) > 5:
        idx = 3
        inventory.loc[idx, "reserved_quantity"] = int(inventory.loc[idx, "on_hand_quantity"]) + 25
        record("inventory_snapshots", f"{inventory.loc[idx, 'snapshot_date']}|{inventory.loc[idx, 'sku']}|{inventory.loc[idx, 'warehouse_id']}", "reserved_quantity", "inconsistent_value", "Reserved quantity intentionally exceeds on-hand quantity.")
    datasets["inventory_snapshots"] = inventory
    # Missing renewal date for one subscription.
    subscriptions = datasets["subscription_records"].copy()
    if len(subscriptions) > 5:
        idx = 2
        subscriptions.loc[idx, "renewal_date"] = pd.NaT
        record("subscription_records", subscriptions.loc[idx, "subscription_id"], "renewal_date", "missing_value", "Renewal date intentionally missing.")
    datasets["subscription_records"] = subscriptions
    return pd.DataFrame(log, columns=["anomaly_id", "dataset", "record_id", "field", "anomaly_type", "description"])


def validate_base_relations(datasets: dict[str, pd.DataFrame]) -> list[str]:
    """Validate identifiers/foreign keys that should remain valid even after anomaly injection.
    Missing foreign keys are permitted only for the documented sales customer anomaly.
    Duplicate business records and value anomalies are logged separately.
    """
    errors: list[str] = []
    products = set(datasets["products"]["sku"])
    warehouses = set(datasets["warehouses"]["warehouse_id"])
    customers = set(datasets["customers"]["customer_id"])
    sales = datasets["sales_transactions"]
    inventory = datasets["inventory_snapshots"]
    po = datasets["purchase_orders"]
    quotes = datasets["sales_quotations"]
    quote_lines = datasets["quotation_lines"]
    subs = datasets["subscription_records"]
    checks = [
        ("sales_transactions.sku", sales["sku"].dropna(), products),
        ("sales_transactions.warehouse_id", sales["warehouse_id"].dropna(), warehouses),
        ("sales_transactions.customer_id", sales["customer_id"].dropna(), customers),
        ("inventory_snapshots.sku", inventory["sku"].dropna(), products),
        ("inventory_snapshots.warehouse_id", inventory["warehouse_id"].dropna(), warehouses),
        ("purchase_orders.sku", po["sku"].dropna(), products),
        ("purchase_orders.warehouse_id", po["warehouse_id"].dropna(), warehouses),
        ("sales_quotations.customer_id", quotes["customer_id"].dropna(), customers),
        ("quotation_lines.sku", quote_lines["sku"].dropna(), products),
        ("quotation_lines.quotation_id", quote_lines["quotation_id"].dropna(), set(quotes["quotation_id"])),
        ("subscription_records.sku", subs["sku"].dropna(), products),
        ("subscription_records.customer_id", subs["customer_id"].dropna(), customers),
    ]
    for name, values, valid in checks:
        unknown = set(values) - valid
        if unknown:
            errors.append(f"{name}: {len(unknown)} unknown foreign key value(s), e.g. {next(iter(unknown))}")
    # Primary identifiers should remain unique. Duplicate candidate intentionally has unique line ID.
    unique_keys = {
        "products.sku": (datasets["products"], "sku"),
        "warehouses.warehouse_id": (datasets["warehouses"], "warehouse_id"),
        "customers.customer_id": (datasets["customers"], "customer_id"),
        "sales_transactions.invoice_line_id": (sales, "invoice_line_id"),
        "purchase_orders.po_line_id": (po, "po_line_id"),
        "sales_quotations.quotation_id": (quotes, "quotation_id"),
        "quotation_lines.quotation_line_id": (quote_lines, "quotation_line_id"),
        "subscription_records.subscription_id": (subs, "subscription_id"),
    }
    for name, (df, col) in unique_keys.items():
        if df[col].duplicated().any():
            errors.append(f"{name}: duplicate primary key(s) found")
    return errors


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def main() -> None:
    args = parse_args()
    start, end = as_date(args.start_date), as_date(args.end_date)
    if end <= start:
        raise ValueError("end-date must be later than start-date")
    if min(args.products, args.warehouses, args.customers) < 1:
        raise ValueError("products, warehouses, and customers must all be >= 1")
    rng = np.random.default_rng(args.seed)
    products, warehouses, customers = make_master_data(rng, args.products, args.warehouses, args.customers)
    datasets: dict[str, pd.DataFrame] = {
        "products": products,
        "warehouses": warehouses,
        "customers": customers,
    }
    datasets["sales_transactions"] = generate_sales(rng, products, warehouses, customers, start, end)
    datasets["inventory_snapshots"] = generate_inventory(rng, products, warehouses, end)
    datasets["purchase_orders"] = generate_purchase_orders(rng, products, warehouses, start, end)
    quotes, quote_lines = generate_quotations(rng, products, customers, start, end)
    datasets["sales_quotations"] = quotes
    datasets["quotation_lines"] = quote_lines
    datasets["subscription_records"] = generate_subscriptions(rng, products, customers, start, end)
    anomaly_log = inject_controlled_anomalies(datasets, rng)
    relation_errors = validate_base_relations(datasets)
    if relation_errors:
        raise RuntimeError("Referential integrity validation failed:\n- " + "\n- ".join(relation_errors))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for name, df in datasets.items():
        filename = f"{name}.csv"
        df.to_csv(output_dir / filename, index=False, date_format="%Y-%m-%d %H:%M:%S")
        row_counts[filename] = int(len(df))
    anomaly_log.to_csv(output_dir / "anomaly_log.csv", index=False)
    row_counts["anomaly_log.csv"] = int(len(anomaly_log))
    sales = datasets["sales_transactions"]
    report = {
        "project": "Demand Intelligence & Inventory Risk Engine",
        "synthetic_data": True,
        "seed": args.seed,
        "configured_start_date": start.date().isoformat(),
        "configured_end_date": end.date().isoformat(),
        "product_count": int(len(products)),
        "warehouse_count": int(len(warehouses)),
        "customer_count": int(len(customers)),
        "sales_transaction_rows_including_controlled_anomalies": int(len(sales)),
        "sales_date_min": sales["transaction_date"].min().date().isoformat() if len(sales) else None,
        "sales_date_max": sales["transaction_date"].max().date().isoformat() if len(sales) else None,
        "sales_unique_skus": int(sales["sku"].nunique()),
        "sales_total_quantity_including_invalid_anomaly": int(sales["quantity"].sum()),
        "controlled_anomaly_count": int(len(anomaly_log)),
        "referential_integrity_validation": "PASS",
        "referential_integrity_checks": "All non-null foreign keys reference existing parent records; primary identifiers are unique.",
        "note": "A few controlled anomalies are intentional. Consult anomaly_log.csv before cleaning or modeling.",
        "row_counts": row_counts,
    }
    with open(output_dir / "generation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=json_safe)
    print("\n=== SYNTHETIC DATA GENERATION REPORT ===")
    print(f"Output directory       : {output_dir}")
    print(f"Random seed            : {args.seed}")
    print(f"Configured date range  : {start.date()} to {end.date()}")
    print(f"Products               : {len(products):,}")
    print(f"Warehouses             : {len(warehouses):,}")
    print(f"Customers              : {len(customers):,}")
    print(f"Sales transaction rows : {len(sales):,}")
    print(f"Sales date coverage    : {report['sales_date_min']} to {report['sales_date_max']}")
    print(f"Controlled anomalies   : {len(anomaly_log):,}")
    print("Referential integrity  : PASS")
    print("\nRows written:")
    for filename, count in row_counts.items():
        print(f"  {filename:<32} {count:>8,}")
    print("  generation_report.json          written")
    print("\nGeneration completed. Review anomaly_log.csv before data cleaning.")
if __name__ == "__main__":
    main()
