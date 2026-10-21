"""
Generate reproducible synthetic datasets for the Demand Intelligence & Inventory Risk Engine.

Purpose:
    Generate relational synthetic data for a fictional Indonesia-based B2B technology
    provider. The data supports demand forecasting, inventory risk, quotation analysis,
    procurement lead-time analysis, and subscription-renewal visibility.

Input:
    No external files required.

Outputs:
    CSV files under data/synthetic/ and generation_report.json.

Usage:
    python src/generate_synthetic_data.py
    python src/generate_synthetic_data.py --seed 42
    python src/generate_synthetic_data.py --products 100 --customers 250 --sales-target-rows 30000
    python src/generate_synthetic_data.py --output-dir data/synthetic

Notes:
    All records are synthetic. Brand names are product-reference examples only and do
    not imply authorization or an actual supplier relationship. Currency rates and
    closure periods are simulated assumptions, not official financial or holiday data.
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
DEFAULT_PRODUCTS = 1000
DEFAULT_WAREHOUSES = 3
DEFAULT_CUSTOMERS = 5000
DEFAULT_SALES_TARGET_ROWS = 500_000

COMPANY_COUNTRY = "Indonesia"
REPORTING_CURRENCY = "USD"
TRANSACTION_CURRENCIES = ["IDR", "USD"]
TRANSACTION_CURRENCY_PROBS = [0.80, 0.20]
FX_IDR_PER_USD_START = 15_500.0
FX_RATE_LOOKUP: dict[str, float] = {}

CATEGORY_PLAN = {
    "Networking Equipment": {
        "count": 180, "brands": ["Cisco", "Juniper Networks"], "product_type": "hardware",
        "families": ["Switch", "Router", "Wireless Access Point", "Network Interface"],
    },
    "Servers & Storage": {
        "count": 150, "brands": ["Dell", "HPE", "Lenovo"], "product_type": "hardware",
        "families": ["Rack Server", "Storage Array", "Backup Appliance", "Server Memory"],
    },
    "Cybersecurity Products": {
        "count": 100, "brands": ["Fortinet", "Palo Alto Networks", "CrowdStrike"], "product_type": "hardware",
        "families": ["Firewall Appliance", "Security Gateway", "Security Appliance"],
    },
    "Cabling, Power & Accessories": {
        "count": 170, "brands": ["CommScope", "APC"], "product_type": "hardware",
        "families": ["Fiber Cable", "Copper Cable", "Transceiver", "Rack Accessory", "UPS"],
    },
    "Software Licenses": {
        "count": 200, "brands": ["Microsoft", "Red Hat"], "product_type": "software_license",
        "families": ["Operating System", "Database License", "Productivity License", "Virtualization License"],
    },
    "Subscription & Support Services": {
        "count": 200, "brands": ["Microsoft", "AWS"], "product_type": "subscription",
        "families": ["Cloud Subscription", "Security Subscription", "Support Plan", "Backup Subscription"],
    },
}

DEMAND_TYPES = ["smooth", "intermittent", "erratic", "lumpy"]
DEMAND_TYPE_PROBS = [0.48, 0.20, 0.17, 0.15]
WAREHOUSE_CITIES = ["Jakarta", "Surabaya", "Bandung"]

# Explicit synthetic business closure periods, not official holiday declarations.
CLOSURE_RANGES = [
    ("2023-12-25", "2024-01-02", "year_end_closure"),
    ("2024-04-08", "2024-04-15", "eid_closure"),
    ("2024-12-24", "2025-01-02", "year_end_closure"),
    ("2025-03-28", "2025-04-07", "eid_closure"),
    ("2025-12-24", "2026-01-02", "year_end_closure"),
    ("2026-03-18", "2026-03-27", "eid_closure"),
]


def parse_args() -> argparse.Namespace:
    """Parse generator configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCTS)
    parser.add_argument("--warehouses", type=int, default=DEFAULT_WAREHOUSES)
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMERS)
    parser.add_argument("--sales-target-rows", type=int, default=DEFAULT_SALES_TARGET_ROWS)
    return parser.parse_args()


def as_date(value: str) -> pd.Timestamp:
    """Parse a date and normalize it to midnight."""
    return pd.Timestamp(value).normalize()


def make_calendar(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """Build a calendar with weekdays and explicit synthetic closure periods."""
    closure_map: dict[pd.Timestamp, str] = {}
    for range_start, range_end, reason in CLOSURE_RANGES:
        for date in pd.date_range(range_start, range_end, freq="D"):
            if start <= date <= end:
                closure_map[date] = reason

    calendar = pd.DataFrame({"calendar_date": pd.date_range(start, end, freq="D")})
    calendar["day_of_week"] = calendar["calendar_date"].dt.day_name()
    calendar["is_weekday"] = calendar["calendar_date"].dt.dayofweek < 5
    calendar["closure_reason"] = calendar["calendar_date"].map(closure_map).fillna("")
    calendar["is_business_day"] = calendar["is_weekday"] & calendar["closure_reason"].eq("")
    calendar["quarter"] = "Q" + calendar["calendar_date"].dt.quarter.astype(str)
    calendar["year"] = calendar["calendar_date"].dt.year
    return calendar, set(closure_map)


def make_master_data(
    rng: np.random.Generator, n_products: int, n_warehouses: int, n_customers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create product, warehouse, customer, supplier, and replacement master data."""
    if min(n_products, n_warehouses, n_customers) < 1:
        raise ValueError("products, warehouses, and customers must all be >= 1")

    categories = list(CATEGORY_PLAN)
    planned = np.array([CATEGORY_PLAN[c]["count"] for c in categories], dtype=float)
    scaled = np.floor(planned / planned.sum() * n_products).astype(int)
    for i in range(n_products - int(scaled.sum())):
        scaled[i % len(scaled)] += 1
    category_counts = dict(zip(categories, scaled.tolist()))

    product_rows: list[dict[str, Any]] = []
    product_number = 1
    for category in categories:
        plan = CATEGORY_PLAN[category]
        for _ in range(category_counts[category]):
            product_type = plan["product_type"]
            brand = str(rng.choice(plan["brands"]))
            family = str(rng.choice(plan["families"]))
            if product_type == "hardware":
                unit_cost = float(np.exp(rng.uniform(np.log(12), np.log(9000))))
            elif product_type == "software_license":
                unit_cost = float(np.exp(rng.uniform(np.log(40), np.log(5000))))
            else:
                unit_cost = float(np.exp(rng.uniform(np.log(60), np.log(12000))))
            unit_price = unit_cost * float(rng.uniform(1.12, 1.55))
            product_rows.append({
                "sku": f"SKU-{product_number:05d}",
                "product_name": f"{brand} {family} Model {product_number:04d}",
                "product_category": category,
                "product_family": f"FAM-{product_number:05d}",
                "brand": brand,
                "product_type": product_type,
                "unit_cost": round(unit_cost, 2),
                "unit_price": round(unit_price, 2),
                "cost_currency": "USD",
                "list_price_currency": "USD",
                "demand_type_seed": str(rng.choice(DEMAND_TYPES, p=DEMAND_TYPE_PROBS)),
                "lifecycle_status": str(rng.choice(["active", "active", "active", "phase_out"])),
                "physical_inventory_flag": product_type == "hardware",
                "digital_fulfillment_flag": product_type != "hardware",
                "active_flag": True,
            })
            product_number += 1
    products = pd.DataFrame(product_rows)

    warehouse_rows = [{
        "warehouse_id": f"WH-{i:02d}",
        "warehouse_name": f"Warehouse {i}",
        "country": COMPANY_COUNTRY,
        "city": WAREHOUSE_CITIES[i - 1] if i <= len(WAREHOUSE_CITIES) else f"City {i}",
        "reporting_currency": REPORTING_CURRENCY,
    } for i in range(1, n_warehouses + 1)]
    warehouses = pd.DataFrame(warehouse_rows)

    segments = ["SMB", "Mid-Market", "Enterprise", "System Integrator", "Reseller", "Public Sector"]
    segment_probs = [0.12, 0.20, 0.28, 0.18, 0.17, 0.05]
    industries = [
        "Technology", "Manufacturing", "Financial Services", "Retail",
        "Education", "Healthcare", "Public Sector", "Telecommunications",
    ]
    customers = pd.DataFrame({
        "customer_id": [f"CUS-{i:06d}" for i in range(1, n_customers + 1)],
        "customer_name": [f"Synthetic Customer {i:05d}" for i in range(1, n_customers + 1)],
        "segment": rng.choice(segments, n_customers, p=segment_probs),
        "industry": rng.choice(industries, n_customers),
        "country": COMPANY_COUNTRY,
        "ship_to_country": COMPANY_COUNTRY,
        "customer_currency_preference": rng.choice(TRANSACTION_CURRENCIES, n_customers, p=TRANSACTION_CURRENCY_PROBS),
        "customer_status": rng.choice(["active", "active", "active", "inactive"], n_customers),
    })

    suppliers = pd.DataFrame([
        # Synthetic portfolio suppliers; these do not represent real vendor agreements.
        {"supplier_id": "SUP-0001", "supplier_name": "Synthetic Asia Hardware Principal", "supplier_country": "China", "procurement_channel": "direct_import", "default_currency": "USD", "supplier_type": "hardware_vendor"},
        {"supplier_id": "SUP-0002", "supplier_name": "Synthetic Regional Hardware Vendor", "supplier_country": "Singapore", "procurement_channel": "direct_import", "default_currency": "USD", "supplier_type": "hardware_vendor"},
        {"supplier_id": "SUP-0003", "supplier_name": "Synthetic Indonesian IT Distributor A", "supplier_country": "Indonesia", "procurement_channel": "local_distributor", "default_currency": "IDR", "supplier_type": "hardware_distributor"},
        {"supplier_id": "SUP-0004", "supplier_name": "Synthetic Indonesian IT Importer", "supplier_country": "Indonesia", "procurement_channel": "local_importer", "default_currency": "IDR", "supplier_type": "hardware_distributor"},
        {"supplier_id": "SUP-0005", "supplier_name": "Synthetic Indonesian IT Distributor B", "supplier_country": "Indonesia", "procurement_channel": "local_distributor", "default_currency": "IDR", "supplier_type": "hardware_distributor"},
        {"supplier_id": "SUP-0006", "supplier_name": "Synthetic Global Software License Partner", "supplier_country": "United States", "procurement_channel": "digital_vendor", "default_currency": "USD", "supplier_type": "software_vendor"},
        {"supplier_id": "SUP-0007", "supplier_name": "Synthetic Indonesian Software License Partner", "supplier_country": "Indonesia", "procurement_channel": "digital_vendor", "default_currency": "IDR", "supplier_type": "software_partner"},
        {"supplier_id": "SUP-0008", "supplier_name": "Synthetic Global Cloud Partner", "supplier_country": "United States", "procurement_channel": "digital_vendor", "default_currency": "USD", "supplier_type": "subscription_partner"},
        {"supplier_id": "SUP-0009", "supplier_name": "Synthetic Indonesian Cloud Partner", "supplier_country": "Indonesia", "procurement_channel": "digital_vendor", "default_currency": "IDR", "supplier_type": "subscription_partner"},
    ])

    # Explicit SKU-level supplier compatibility. All assignments are synthetic
    # portfolio assumptions and do not imply real-world authorization or agreements.
    supplier_product_rows: list[dict[str, str]] = []
    supplier_by_id = suppliers.set_index("supplier_id").to_dict("index")

    hardware_supplier_ids = ["SUP-0001", "SUP-0002", "SUP-0003", "SUP-0004", "SUP-0005"]
    software_supplier_ids = ["SUP-0006", "SUP-0007"]
    subscription_supplier_ids = ["SUP-0008", "SUP-0009"]

    for product in products.itertuples(index=False):
        if product.product_type == "hardware":
            candidate_ids = hardware_supplier_ids
        elif product.product_type == "software_license":
            candidate_ids = software_supplier_ids
        else:
            candidate_ids = subscription_supplier_ids

        # One primary and one alternate supplier per SKU, rotated across each pool.
        sku_number = int(str(product.sku).split("-")[-1])
        primary_id = candidate_ids[(sku_number - 1) % len(candidate_ids)]
        alternate_id = candidate_ids[(candidate_ids.index(primary_id) + 1) % len(candidate_ids)]

        for supplier_id, sourcing_role in [
            (primary_id, "primary"),
            (alternate_id, "alternate"),
        ]:
            supplier = supplier_by_id[supplier_id]
            supplier_product_rows.append({
                "supplier_id": supplier_id,
                "sku": str(product.sku),
                "brand": str(product.brand),
                "product_category": str(product.product_category),
                "product_type": str(product.product_type),
                "sourcing_role": sourcing_role,
                "procurement_channel": str(supplier["procurement_channel"]),
                "default_currency": str(supplier["default_currency"]),
            })

    supplier_product_compatibility = pd.DataFrame(supplier_product_rows)

    hardware_skus = products.loc[products["product_type"].eq("hardware"), "sku"].to_numpy()
    replacement_rows = []
    pair_count = min(max(1, n_products // 50), len(hardware_skus) // 2)
    if pair_count:
        selected = rng.choice(hardware_skus, size=pair_count * 2, replace=False)
        for i in range(pair_count):
            old_sku, new_sku = str(selected[2 * i]), str(selected[2 * i + 1])
            effective = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 500)))
            replacement_rows.append({
                "replacement_id": f"REP-{i + 1:05d}",
                "old_sku": old_sku,
                "new_sku": new_sku,
                "relationship_type": str(rng.choice(["equivalent", "successor"])),
                "effective_date": effective.date().isoformat(),
                "reason": str(rng.choice(["vendor_refresh", "technology_upgrade", "specification_change"])),
                "specification_change": str(rng.choice(["minor_performance_change", "capacity_increase", "interface_update", "feature_refresh"])),
                "old_sku_sellable_until": (effective + pd.Timedelta(days=int(rng.integers(90, 365)))).date().isoformat(),
            })
    replacements = pd.DataFrame(replacement_rows, columns=[
        "replacement_id", "old_sku", "new_sku", "relationship_type", "effective_date",
        "reason", "specification_change", "old_sku_sellable_until",
    ])
    return products, warehouses, customers, suppliers, replacements, supplier_product_compatibility


def build_fx_table(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Create reproducible synthetic monthly IDR-to-USD reference rates."""
    dates = pd.date_range(start=start.to_period("M").start_time, end=end, freq="MS")
    rng = np.random.default_rng(2026)
    idr_per_usd = FX_IDR_PER_USD_START
    rows = []
    for date in dates:
        idr_per_usd = float(np.clip(idr_per_usd + rng.normal(0, 100), 14500, 17500))
        rows.append({
            "exchange_rate_date": date.date().isoformat(),
            "from_currency": "IDR",
            "to_currency": "USD",
            "exchange_rate_to_usd": round(1 / idr_per_usd, 10),
            "idr_per_usd_reference": round(idr_per_usd, 2),
            "rate_type": "synthetic_monthly_reference",
        })
    table = pd.DataFrame(rows)
    FX_RATE_LOOKUP.clear()
    for row in table.itertuples(index=False):
        FX_RATE_LOOKUP[str(row.exchange_rate_date)[:7]] = float(row.exchange_rate_to_usd)
    return table


def get_fx_rate(date: pd.Timestamp, fx_table: pd.DataFrame) -> float:
    """Return USD value for one IDR using the latest prior monthly reference rate."""
    month_key = date.strftime("%Y-%m")
    if month_key in FX_RATE_LOOKUP:
        return FX_RATE_LOOKUP[month_key]
    # Fallback for dates before the first configured month.
    first_key = min(FX_RATE_LOOKUP)
    return FX_RATE_LOOKUP[first_key]


def make_business_dates(start: pd.Timestamp, end: pd.Timestamp, closure_dates: set[pd.Timestamp]) -> pd.DatetimeIndex:
    """Return weekdays that are not in configured closure periods."""
    dates = pd.date_range(start, end, freq="B")
    blocked = {pd.Timestamp(date).normalize() for date in closure_dates}
    return pd.DatetimeIndex([date for date in dates if date.normalize() not in blocked])


def quarter_factor(date: pd.Timestamp) -> float:
    """Apply Q3 > Q4 > Q1 > Q2 demand seasonality."""
    return {1: 1.05, 2: 0.82, 3: 1.25, 4: 1.14}[int(date.quarter)]


def generate_sales(
    rng: np.random.Generator, products: pd.DataFrame, warehouses: pd.DataFrame,
    customers: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    target_rows: int, fx_table: pd.DataFrame, closure_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    """Generate exactly target_rows distinct invoice lines while preserving SKU-week demand totals."""
    business_dates = make_business_dates(start, end, closure_dates)
    if business_dates.empty:
        raise ValueError("The selected date range has no valid business days.")

    customer_ids = customers["customer_id"].to_numpy()
    warehouse_ids = warehouses["warehouse_id"].to_numpy()
    currency_map = customers.set_index("customer_id")["customer_currency_preference"].to_dict()
    weeks = pd.date_range(start=start, end=end, freq="W-SUN")
    if len(weeks) == 0:
        weeks = pd.DatetimeIndex([end.normalize()])
    if weeks[-1].normalize() < end.normalize():
        weeks = weeks.append(pd.DatetimeIndex([weeks[-1] + pd.Timedelta(days=7)]))

    business_dates_by_week: dict[pd.Timestamp, pd.DatetimeIndex] = {}
    for week_end in weeks:
        week_start = max(start.normalize(), (week_end - pd.Timedelta(days=6)).normalize())
        bounded_week_end = min(end.normalize(), week_end.normalize())
        business_dates_by_week[week_end] = business_dates[
            (business_dates >= week_start) & (business_dates <= bounded_week_end)
        ]

    # First generate weekly demand independently of invoice-line count.
    # Invoice lines are allocated afterward so splitting demand into more lines
    # never changes the SKU-week quantity total.
    weekly_groups: list[dict[str, Any]] = []
    for product in products.itertuples(index=False):
        base = float(rng.uniform(4, 22))
        pattern = product.demand_type_seed
        for week_index, week_end in enumerate(weeks):
            week_business_dates = business_dates_by_week[week_end]
            if week_business_dates.empty:
                continue
            anchor = min(max(week_end, start), end)
            expected = base * quarter_factor(anchor)
            elapsed_years = week_index / 52.0
            if pattern == "smooth":
                expected *= 1 + 0.04 * elapsed_years
            elif pattern == "intermittent":
                if rng.random() > 0.22:
                    continue
                expected *= rng.uniform(1.5, 4.0)
            elif pattern == "erratic":
                expected *= float(rng.lognormal(mean=0.0, sigma=0.9))
            elif pattern == "lumpy":
                if rng.random() > 0.10:
                    continue
                expected *= float(rng.lognormal(mean=0.8, sigma=1.0))

            weekly_quantity = int(rng.poisson(max(expected, 0.2)))
            if weekly_quantity <= 0:
                continue
            # The per-group capacity is bounded to avoid an unrealistic number
            # of invoice lines for a single SKU-week. Every line must have qty >= 1.
            line_capacity = min(12, weekly_quantity)
            weekly_groups.append({
                "product": product,
                "week_end": week_end,
                "week_business_dates": week_business_dates,
                "weekly_quantity": weekly_quantity,
                "line_capacity": line_capacity,
            })

    if not weekly_groups:
        raise ValueError("No positive SKU-week demand was generated for the selected date range.")

    minimum_rows = len(weekly_groups)
    total_capacity = sum(group["line_capacity"] for group in weekly_groups)
    if target_rows < minimum_rows:
        raise ValueError(
            f"sales-target-rows ({target_rows}) is below the {minimum_rows} positive SKU-week groups; "
            "cannot preserve positive weekly demand while representing each group with at least one line."
        )
    if target_rows > total_capacity:
        raise ValueError(
            f"sales-target-rows ({target_rows}) exceeds the {total_capacity} available invoice-line capacity "
            "under the 12-lines-per-SKU-week limit. Increase the limit or review the target."
        )

    # Allocate exactly target_rows across positive SKU-week groups. Start with
    # one line per group, then distribute remaining lines proportionally to each
    # group's remaining capacity. This changes invoice granularity, not demand.
    capacities = np.array([g["line_capacity"] for g in weekly_groups], dtype=np.int64)
    line_counts = np.ones(len(weekly_groups), dtype=np.int64)
    remaining = target_rows - minimum_rows
    extra_capacity = capacities - 1
    total_extra_capacity = int(extra_capacity.sum())
    if remaining > 0:
        quotas = remaining * extra_capacity / total_extra_capacity
        allocated = np.floor(quotas).astype(np.int64)
        line_counts += allocated
        leftover = int(remaining - allocated.sum())
        if leftover:
            eligible = np.flatnonzero(line_counts < capacities)
            fractions = quotas - allocated
            order = eligible[np.argsort(fractions[eligible])[::-1]]
            line_counts[order[:leftover]] += 1

    rows: list[dict[str, Any]] = []
    line_id = 1
    for group, line_count in zip(weekly_groups, line_counts):
        product = group["product"]
        weekly_quantity = int(group["weekly_quantity"])
        line_count = int(line_count)
        # Positive integer partition: exact line count, no zero-quantity lines,
        # and the sum remains exactly equal to the generated weekly demand.
        quantities = np.full(line_count, weekly_quantity // line_count, dtype=np.int64)
        quantities[: weekly_quantity % line_count] += 1
        if int(quantities.sum()) != weekly_quantity or (quantities <= 0).any():
            raise RuntimeError("Internal sales allocation error: weekly quantity was not preserved.")

        for quantity in quantities:
            tx_date = pd.Timestamp(rng.choice(group["week_business_dates"]))
            customer_id = str(rng.choice(customer_ids))
            warehouse_id = str(rng.choice(warehouse_ids))
            currency = str(currency_map[customer_id])
            fx_rate = get_fx_rate(tx_date, fx_table) if currency == "IDR" else 1.0
            unit_price_usd = float(product.unit_price) * float(rng.uniform(0.96, 1.04))
            unit_price_original = unit_price_usd / fx_rate if currency == "IDR" else unit_price_usd
            unit_price_original = round(unit_price_original, 0 if currency == "IDR" else 2)
            line_amount_original = round(unit_price_original * int(quantity), 2)
            line_amount_usd = round(line_amount_original * fx_rate, 2)
            rows.append({
                "invoice_line_id": f"SIL-{line_id:09d}",
                "invoice_id": f"INV-{line_id:08d}",
                "transaction_date": tx_date,
                "sku": product.sku,
                "warehouse_id": warehouse_id,
                "customer_id": customer_id,
                "quantity": int(quantity),
                "unit_price": unit_price_original,
                "line_amount": line_amount_original,
                "transaction_currency": currency,
                "exchange_rate_date": tx_date.to_period("M").start_time.date().isoformat(),
                "exchange_rate_to_usd": fx_rate,
                "unit_price_usd": round(unit_price_original * fx_rate, 2),
                "line_amount_usd": line_amount_usd,
                "transaction_status": "posted",
            })
            line_id += 1

    sales = pd.DataFrame(rows)
    if len(sales) != target_rows:
        raise RuntimeError(f"Expected exactly {target_rows} sales rows before anomalies, generated {len(sales)}.")
    sales = sales.sort_values("transaction_date").reset_index(drop=True)
    # Reassign stable sequential IDs after sorting; no duplicate business records are cloned.
    sales["invoice_line_id"] = [f"SIL-{i:09d}" for i in range(1, len(sales) + 1)]
    sales["invoice_id"] = [f"INV-{i:08d}" for i in range(1, len(sales) + 1)]
    return sales
def procurement_profile(
    product: pd.Series,
    compatible_supplier_records: list[dict[str, str]],
    suppliers: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[str, str, str, int, int]:
    """Select a compatible supplier, preferring the mapped primary source."""
    if not compatible_supplier_records:
        raise ValueError(f"No compatible supplier configured for SKU {product['sku']}")

    supplier_lookup = suppliers.set_index("supplier_id").to_dict("index")
    roles = [record["sourcing_role"] for record in compatible_supplier_records]
    weights = np.array([0.75 if role == "primary" else 0.25 for role in roles], dtype=float)
    weights = weights / weights.sum()
    selected_index = int(rng.choice(len(compatible_supplier_records), p=weights))
    supplier_id = str(compatible_supplier_records[selected_index]["supplier_id"])
    supplier = supplier_lookup[supplier_id]
    channel = str(supplier["procurement_channel"])
    currency = str(supplier["default_currency"])
    product_type = str(product["product_type"])
    category = str(product["product_category"])

    if product_type == "hardware":
        if channel == "direct_import":
            if supplier["supplier_country"] == "China":
                lead_min, lead_max = (56, 84) if category == "Cabling, Power & Accessories" else (14, 28)
            else:  # Regional direct-import sourcing from Singapore
                lead_min, lead_max = 7, 21
        elif channel == "local_importer":
            lead_min, lead_max = 3, 14
        else:  # Local distributor
            lead_min, lead_max = 2, 10
    elif product_type == "software_license":
        # Synthetic digital provisioning / administrative lead time, not shipping.
        lead_min, lead_max = (1, 5) if currency == "USD" else (0, 3)
    else:
        # Synthetic subscription activation / renewal administration lead time.
        lead_min, lead_max = (1, 5) if currency == "USD" else (0, 3)

    return supplier_id, channel, currency, lead_min, lead_max


def generate_inventory(
    rng: np.random.Generator, products: pd.DataFrame, warehouses: pd.DataFrame, end: pd.Timestamp
) -> pd.DataFrame:
    """Generate monthly inventory snapshots for physical hardware only."""
    snapshot_dates = pd.date_range(start=end - pd.DateOffset(months=11), end=end, freq="MS")
    snapshot_dates = snapshot_dates.union(pd.DatetimeIndex([end])).sort_values()
    hardware_skus = products.loc[products["physical_inventory_flag"], "sku"]
    rows = []
    for sku in hardware_skus:
        for warehouse_id in warehouses["warehouse_id"]:
            starting = int(rng.integers(25, 500))
            for snapshot_date in snapshot_dates:
                on_hand = max(0, int(starting + rng.normal(0, max(5, starting * 0.12))))
                reserved = int(rng.integers(0, min(on_hand, 30) + 1)) if on_hand else 0
                rows.append({
                    "snapshot_date": snapshot_date,
                    "sku": sku,
                    "warehouse_id": warehouse_id,
                    "on_hand_quantity": on_hand,
                    "reserved_quantity": reserved,
                    "available_quantity": max(0, on_hand - reserved),
                })
                starting = max(0, on_hand + int(rng.integers(-35, 36)))
    return pd.DataFrame(rows)


def generate_purchase_orders(
    rng: np.random.Generator, products: pd.DataFrame, warehouses: pd.DataFrame,
    suppliers: pd.DataFrame,
    supplier_product_compatibility: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp, fx_table: pd.DataFrame,
) -> pd.DataFrame:
    """Generate PO lines using explicit SKU-level supplier compatibility."""
    n = max(300, min(6000, len(products) // 2))
    product_lookup = products.set_index("sku")
    supplier_country = suppliers.set_index("supplier_id")["supplier_country"].to_dict()
    compatibility_lookup = {
        str(sku): group.to_dict("records")
        for sku, group in supplier_product_compatibility.groupby("sku", sort=False)
    }
    rows = []
    date_span = max(1, (end - start).days - 90)
    statuses = ["open", "partial", "received", "cancelled"]
    for i in range(1, n + 1):
        sku = str(rng.choice(products["sku"].to_numpy()))
        product = product_lookup.loc[sku]
        compatible_supplier_records = compatibility_lookup.get(sku, [])
        supplier_id, channel, currency, lead_min, lead_max = procurement_profile(
            product, compatible_supplier_records, suppliers, rng
        )
        order_date = start + pd.Timedelta(days=int(rng.integers(0, date_span)))
        status = str(rng.choice(statuses, p=[0.25, 0.15, 0.50, 0.10]))
        lead_days = int(rng.integers(lead_min, lead_max + 1))
        expected_date = order_date + pd.Timedelta(days=lead_days)
        quantity = int(rng.integers(5, 251)) if product["product_type"] == "hardware" else int(rng.integers(1, 101))
        received_qty = quantity if status == "received" else (
            int(rng.integers(1, quantity + 1)) if status == "partial" else 0
        )
        if status == "received":
            # Early/late delivery is measured against the expected date, but an
            # actual receipt cannot precede the PO order date.
            delivery_variance_days = int(rng.integers(-7, 22))
            actual_date = max(
                order_date,
                expected_date + pd.Timedelta(days=delivery_variance_days),
            )
        else:
            actual_date = pd.NaT
        fx_rate = get_fx_rate(order_date, fx_table) if currency == "IDR" else 1.0
        unit_cost_usd = float(product["unit_cost"]) * float(rng.uniform(0.95, 1.08))
        unit_cost_original = unit_cost_usd / fx_rate if currency == "IDR" else unit_cost_usd
        rows.append({
            "po_line_id": f"POL-{i:07d}",
            "po_id": f"PO-{i:06d}",
            "supplier_id": supplier_id,
            "supplier_country": supplier_country[supplier_id],
            "procurement_channel": channel,
            "sku": sku,
            "warehouse_id": str(rng.choice(warehouses["warehouse_id"].to_numpy())),
            "order_date": order_date,
            "transaction_currency": currency,
            "exchange_rate_date": order_date.to_period("M").start_time.date().isoformat(),
            "exchange_rate_to_usd": fx_rate,
            "ordered_quantity": quantity,
            "received_quantity": min(received_qty, quantity),
            "unit_cost_original": round(unit_cost_original, 2),
            "line_amount_original": round(unit_cost_original * quantity, 2),
            "line_amount_usd": round(unit_cost_usd * quantity, 2),
            "lead_time_days": lead_days,
            "expected_receipt_date": expected_date,
            "actual_receipt_date": actual_date,
            "status": status,
        })
    if rows:
        rows[0]["status"] = "open"
        rows[0]["expected_receipt_date"] = end - pd.Timedelta(days=10)
        rows[0]["actual_receipt_date"] = pd.NaT
        rows[0]["received_quantity"] = 0
    return pd.DataFrame(rows)


def generate_quotations(
    rng: np.random.Generator, products: pd.DataFrame, customers: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp, fx_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate domestic quotation headers and lines."""
    n_quotes = max(500, min(10000, len(customers) // 5))
    customer_ids = customers["customer_id"].to_numpy()
    product_records = list(products.itertuples(index=False))
    quote_rows, line_rows = [], []
    line_counter = 1
    for i in range(1, n_quotes + 1):
        quote_date = start + pd.Timedelta(days=int(rng.integers(0, (end - start).days + 1)))
        status = str(rng.choice(["won", "lost", "open"], p=[0.42, 0.33, 0.25]))
        submitted_at = quote_date + pd.Timedelta(hours=int(rng.integers(0, 9)))
        approval_at = submitted_at + pd.Timedelta(hours=int(max(1, rng.lognormal(2.3, 0.7))))
        close_date = quote_date + pd.Timedelta(days=int(rng.integers(1, 45))) if status != "open" else pd.NaT
        currency = str(rng.choice(TRANSACTION_CURRENCIES, p=TRANSACTION_CURRENCY_PROBS))
        fx_rate = get_fx_rate(quote_date, fx_table) if currency == "IDR" else 1.0
        quotation_id = f"QT-{i:07d}"
        quote_rows.append({
            "quotation_id": quotation_id,
            "customer_id": str(rng.choice(customer_ids)),
            "quotation_date": quote_date,
            "submitted_at": submitted_at,
            "approval_at": approval_at,
            "status": status,
            "close_date": close_date,
            "transaction_currency": currency,
            "exchange_rate_date": quote_date.to_period("M").start_time.date().isoformat(),
            "exchange_rate_to_usd": fx_rate,
        })
        n_lines = int(rng.integers(1, 5))
        chosen = rng.choice(len(product_records), size=min(n_lines, len(product_records)), replace=False)
        for idx in chosen:
            product = product_records[int(idx)]
            quantity = int(rng.integers(1, 31))
            price_usd = float(product.unit_price) * float(rng.uniform(0.90, 1.10))
            price_original = price_usd / fx_rate if currency == "IDR" else price_usd
            amount_original = round(quantity * price_original, 2)
            line_rows.append({
                "quotation_line_id": f"QTL-{line_counter:08d}",
                "quotation_id": quotation_id,
                "sku": product.sku,
                "quantity": quantity,
                "unit_price": round(price_original, 2),
                "line_amount": amount_original,
                "transaction_currency": currency,
                "line_amount_usd": round(amount_original * fx_rate, 2),
            })
            line_counter += 1
    return pd.DataFrame(quote_rows), pd.DataFrame(line_rows)


def generate_subscriptions(
    rng: np.random.Generator, products: pd.DataFrame, customers: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp, fx_table: pd.DataFrame,
) -> pd.DataFrame:
    """Generate digital subscription contracts; do not assign them physical stock."""
    sub_skus = products.loc[products["product_type"].eq("subscription"), "sku"].to_numpy()
    n_subscriptions = max(300, min(5000, len(customers) // 2))
    rows = []
    customer_ids = customers["customer_id"].to_numpy()
    for i in range(1, n_subscriptions + 1):
        latest_start = max(1, (end - start).days - 365)
        contract_start = start + pd.Timedelta(days=int(rng.integers(0, latest_start)))
        term_months = int(rng.choice([12, 24, 36], p=[0.70, 0.20, 0.10]))
        renewal_date = contract_start + pd.DateOffset(months=term_months)
        currency = str(rng.choice(TRANSACTION_CURRENCIES, p=TRANSACTION_CURRENCY_PROBS))
        fx_rate = get_fx_rate(contract_start, fx_table) if currency == "IDR" else 1.0
        acv_usd = float(rng.uniform(500, 25000))
        acv_original = acv_usd / fx_rate if currency == "IDR" else acv_usd
        rows.append({
            "subscription_id": f"SUB-{i:07d}",
            "customer_id": str(rng.choice(customer_ids)),
            "sku": str(rng.choice(sub_skus)),
            "contract_start_date": contract_start,
            "renewal_date": renewal_date,
            "contract_term_months": term_months,
            "annual_contract_value": round(acv_original, 2),
            "transaction_currency": currency,
            "exchange_rate_date": contract_start.to_period("M").start_time.date().isoformat(),
            "exchange_rate_to_usd": fx_rate,
            "annual_contract_value_usd": round(acv_original * fx_rate, 2),
            "status": str(rng.choice(["active", "renewed", "cancelled", "expired"], p=[0.52, 0.20, 0.12, 0.16])),
            "fulfillment_type": "digital",
        })
    return pd.DataFrame(rows)


def inject_controlled_anomalies(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inject controlled anomalies after valid base data has been generated."""
    log: list[dict[str, Any]] = []

    def record(dataset: str, record_id: str, field: str, anomaly_type: str, description: str) -> None:
        log.append({
            "anomaly_id": f"ANOM-{len(log) + 1:04d}", "dataset": dataset,
            "record_id": record_id, "field": field, "anomaly_type": anomaly_type,
            "description": description,
        })

    sales = datasets["sales_transactions"].copy()
    if len(sales) > 10:
        source = sales.iloc[5].copy()
        duplicate = source.copy()
        duplicate["invoice_line_id"] = f"SIL-DUP-{source['invoice_line_id']}"
        sales = pd.concat([sales, pd.DataFrame([duplicate])], ignore_index=True)
        record("sales_transactions", duplicate["invoice_line_id"], "invoice_id,sku,warehouse_id,transaction_date,quantity",
               "duplicate_candidate", "Copied business fields with a new line ID; review before removal.")
    if len(sales) > 20:
        sales.loc[20, "customer_id"] = pd.NA
        record("sales_transactions", sales.loc[20, "invoice_line_id"], "customer_id",
               "missing_value", "Customer reference intentionally missing.")
    if len(sales) > 30:
        sales.loc[30, "quantity"] = -abs(int(sales.loc[30, "quantity"]))
        sales.loc[30, "line_amount"] = -abs(float(sales.loc[30, "line_amount"]))
        sales.loc[30, "line_amount_usd"] = -abs(float(sales.loc[30, "line_amount_usd"]))
        record("sales_transactions", sales.loc[30, "invoice_line_id"], "quantity",
               "invalid_value", "Negative quantity and corresponding amounts intentionally injected.")
    datasets["sales_transactions"] = sales

    quotations = datasets["sales_quotations"].copy()
    if len(quotations) > 5:
        quotations.loc[4, "approval_at"] = pd.NaT
        record("sales_quotations", quotations.loc[4, "quotation_id"], "approval_at",
               "missing_value", "Approval timestamp intentionally missing.")
    datasets["sales_quotations"] = quotations

    inventory = datasets["inventory_snapshots"].copy()
    if len(inventory) > 5:
        inventory.loc[3, "reserved_quantity"] = int(inventory.loc[3, "on_hand_quantity"]) + 25
        inventory.loc[3, "available_quantity"] = 0
        record("inventory_snapshots", f"{inventory.loc[3, 'snapshot_date']}|{inventory.loc[3, 'sku']}|{inventory.loc[3, 'warehouse_id']}",
               "reserved_quantity", "inconsistent_value", "Reserved quantity intentionally exceeds on-hand quantity.")
    datasets["inventory_snapshots"] = inventory

    subscriptions = datasets["subscription_records"].copy()
    if len(subscriptions) > 5:
        subscriptions.loc[2, "renewal_date"] = pd.NaT
        record("subscription_records", subscriptions.loc[2, "subscription_id"], "renewal_date",
               "missing_value", "Renewal date intentionally missing.")
    datasets["subscription_records"] = subscriptions

    return pd.DataFrame(log, columns=[
        "anomaly_id", "dataset", "record_id", "field", "anomaly_type", "description",
    ])


def validate_base_relations(datasets: dict[str, pd.DataFrame]) -> list[str]:
    """Validate primary keys, foreign keys, and domestic-only customer scope."""
    errors: list[str] = []
    products = set(datasets["products"]["sku"])
    warehouses = set(datasets["warehouses"]["warehouse_id"])
    customers = set(datasets["customers"]["customer_id"])
    suppliers = set(datasets["suppliers"]["supplier_id"])
    quotations = set(datasets["sales_quotations"]["quotation_id"])
    compatible_supplier_sku_pairs = set(zip(
        datasets["supplier_product_compatibility"]["supplier_id"].astype(str),
        datasets["supplier_product_compatibility"]["sku"].astype(str),
    ))
    invalid_po_rows = datasets["purchase_orders"].loc[
        ~datasets["purchase_orders"].apply(
            lambda row: (str(row["supplier_id"]), str(row["sku"])) in compatible_supplier_sku_pairs,
            axis=1,
        )
    ]
    if not invalid_po_rows.empty:
        errors.append(
            f"purchase_orders: {len(invalid_po_rows)} row(s) use a supplier not mapped to the SKU"
        )

    checks = [
        ("sales_transactions.sku", datasets["sales_transactions"]["sku"].dropna(), products),
        ("sales_transactions.warehouse_id", datasets["sales_transactions"]["warehouse_id"].dropna(), warehouses),
        ("sales_transactions.customer_id", datasets["sales_transactions"]["customer_id"].dropna(), customers),
        ("inventory_snapshots.sku", datasets["inventory_snapshots"]["sku"].dropna(), products),
        ("inventory_snapshots.warehouse_id", datasets["inventory_snapshots"]["warehouse_id"].dropna(), warehouses),
        ("purchase_orders.sku", datasets["purchase_orders"]["sku"].dropna(), products),
        ("purchase_orders.warehouse_id", datasets["purchase_orders"]["warehouse_id"].dropna(), warehouses),
        ("purchase_orders.supplier_id", datasets["purchase_orders"]["supplier_id"].dropna(), suppliers),
        ("supplier_product_compatibility.supplier_id", datasets["supplier_product_compatibility"]["supplier_id"].dropna(), suppliers),
        ("supplier_product_compatibility.sku", datasets["supplier_product_compatibility"]["sku"].dropna(), products),
        ("sales_quotations.customer_id", datasets["sales_quotations"]["customer_id"].dropna(), customers),
        ("quotation_lines.sku", datasets["quotation_lines"]["sku"].dropna(), products),
        ("quotation_lines.quotation_id", datasets["quotation_lines"]["quotation_id"].dropna(), quotations),
        ("subscription_records.sku", datasets["subscription_records"]["sku"].dropna(), products),
        ("subscription_records.customer_id", datasets["subscription_records"]["customer_id"].dropna(), customers),
        ("product_replacements.old_sku", datasets["product_replacements"]["old_sku"].dropna(), products),
        ("product_replacements.new_sku", datasets["product_replacements"]["new_sku"].dropna(), products),
    ]
    for name, values, valid_values in checks:
        unknown = set(values) - valid_values
        if unknown:
            errors.append(f"{name}: {len(unknown)} unknown foreign key value(s)")

    unique_keys = {
        "products.sku": (datasets["products"], "sku"),
        "warehouses.warehouse_id": (datasets["warehouses"], "warehouse_id"),
        "customers.customer_id": (datasets["customers"], "customer_id"),
        "suppliers.supplier_id": (datasets["suppliers"], "supplier_id"),
        "sales_transactions.invoice_line_id": (datasets["sales_transactions"], "invoice_line_id"),
        "purchase_orders.po_line_id": (datasets["purchase_orders"], "po_line_id"),
        "supplier_product_compatibility": (
            datasets["supplier_product_compatibility"].assign(
                _key=lambda frame: frame["supplier_id"].astype(str) + "|" + frame["sku"].astype(str)
            ),
            "_key",
        ),
        "sales_quotations.quotation_id": (datasets["sales_quotations"], "quotation_id"),
        "quotation_lines.quotation_line_id": (datasets["quotation_lines"], "quotation_line_id"),
        "subscription_records.subscription_id": (datasets["subscription_records"], "subscription_id"),
        "product_replacements.replacement_id": (datasets["product_replacements"], "replacement_id"),
    }
    for name, (frame, column) in unique_keys.items():
        if frame[column].duplicated().any():
            errors.append(f"{name}: duplicate primary key(s) found")

    for column in ["country", "ship_to_country"]:
        if not datasets["customers"][column].eq(COMPANY_COUNTRY).all():
            errors.append(f"customers.{column}: all values must be Indonesia")
    if not datasets["warehouses"]["country"].eq(COMPANY_COUNTRY).all():
        errors.append("warehouses.country: all warehouses must be in Indonesia")
    return errors


def json_safe(value: Any) -> Any:
    """Convert common Pandas and NumPy values into JSON-compatible values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def main() -> None:
    """Run generation, validation, and output writing."""
    args = parse_args()
    start, end = as_date(args.start_date), as_date(args.end_date)
    if end < start:
        raise ValueError("end-date must be on or after start-date")
    if args.sales_target_rows < 1:
        raise ValueError("sales-target-rows must be >= 1")

    rng = np.random.default_rng(args.seed)
    calendar, closure_dates = make_calendar(start, end)
    products, warehouses, customers, suppliers, replacements, supplier_product_compatibility = make_master_data(
        rng, args.products, args.warehouses, args.customers
    )
    fx_table = build_fx_table(start, end)

    datasets: dict[str, pd.DataFrame] = {
        "products": products,
        "warehouses": warehouses,
        "customers": customers,
        "suppliers": suppliers,
        "supplier_product_compatibility": supplier_product_compatibility,
        "product_replacements": replacements,
        "calendar": calendar,
        "fx_rates": fx_table,
    }
    datasets["sales_transactions"] = generate_sales(
        rng, products, warehouses, customers, start, end,
        args.sales_target_rows, fx_table, closure_dates,
    )
    datasets["inventory_snapshots"] = generate_inventory(rng, products, warehouses, end)
    datasets["purchase_orders"] = generate_purchase_orders(
        rng, products, warehouses, suppliers, supplier_product_compatibility, start, end, fx_table,
    )
    quotations, quotation_lines = generate_quotations(
        rng, products, customers, start, end, fx_table,
    )
    datasets["sales_quotations"] = quotations
    datasets["quotation_lines"] = quotation_lines
    datasets["subscription_records"] = generate_subscriptions(
        rng, products, customers, start, end, fx_table,
    )

    anomaly_log = inject_controlled_anomalies(datasets)
    relation_errors = validate_base_relations(datasets)
    if relation_errors:
        raise RuntimeError("Referential integrity validation failed:\n- " + "\n- ".join(relation_errors))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for name, frame in datasets.items():
        filename = f"{name}.csv"
        frame.to_csv(output_dir / filename, index=False, date_format="%Y-%m-%d %H:%M:%S")
        row_counts[filename] = int(len(frame))
    anomaly_log.to_csv(output_dir / "anomaly_log.csv", index=False)
    row_counts["anomaly_log.csv"] = int(len(anomaly_log))

    sales = datasets["sales_transactions"]
    report = {
        "project": "Demand Intelligence & Inventory Risk Engine",
        "synthetic_data": True,
        "company_country": COMPANY_COUNTRY,
        "sales_market": "Indonesia domestic market only",
        "reporting_currency": REPORTING_CURRENCY,
        "seed": args.seed,
        "configured_start_date": start.date().isoformat(),
        "configured_end_date": end.date().isoformat(),
        "product_count": int(len(products)),
        "warehouse_count": int(len(warehouses)),
        "customer_count": int(len(customers)),
        "sales_target_rows_before_controlled_anomalies": int(args.sales_target_rows),
        "sales_rows_before_controlled_anomalies": int(len(sales) - (1 if len(sales) > 10 else 0)),
        "sales_rows_including_controlled_anomalies": int(len(sales)),
        "sales_target_reached_before_anomalies": bool((len(sales) - (1 if len(sales) > 10 else 0)) >= args.sales_target_rows),
        "sales_date_min": sales["transaction_date"].min().date().isoformat() if len(sales) else None,
        "sales_date_max": sales["transaction_date"].max().date().isoformat() if len(sales) else None,
        "sales_unique_skus": int(sales["sku"].nunique()),
        "sales_currency_distribution": sales["transaction_currency"].value_counts().to_dict() if len(sales) else {},
        "supplier_country_distribution": datasets["purchase_orders"]["supplier_country"].value_counts().to_dict(),
        "procurement_channel_distribution": datasets["purchase_orders"]["procurement_channel"].value_counts().to_dict(),
        "product_category_distribution": products["product_category"].value_counts().to_dict(),
        "product_type_distribution": products["product_type"].value_counts().to_dict(),
        "controlled_anomaly_count": int(len(anomaly_log)),
        "referential_integrity_validation": "PASS",
        "business_assumptions": [
            "All records are synthetic; brand names do not imply vendor authorization.",
            "Company, customers, ship-to destinations, and warehouses are in Indonesia.",
            "Reporting currency is USD; source transaction currency is preserved.",
            "FX rates are synthetic monthly reference assumptions, not historical official rates.",
            "Hardware is inventory-managed; software licenses and subscriptions are digitally fulfilled by default.",
            "Closure periods are synthetic business assumptions, not official holiday declarations.",
            "Sales lines are distinct generated records; no row duplication is used to meet the target.",
            "Purchase order quantities are not treated as on-hand inventory.",
            "Supplier-to-SKU compatibility is explicitly mapped using synthetic primary and alternate sourcing assignments.",
            "Won quotations are not treated as realized revenue because sales-order linkage is not modeled.",
            "Product replacement relationships do not automatically merge SKU histories for forecasting.",
            "Controlled anomalies are documented in anomaly_log.csv and should be reviewed before cleaning.",
        ],
        "row_counts": row_counts,
    }
    with open(output_dir / "generation_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=json_safe)

    print("\n=== SYNTHETIC DATA GENERATION REPORT ===")
    print(f"Output directory       : {output_dir}")
    print(f"Random seed            : {args.seed}")
    print(f"Configured date range  : {start.date()} to {end.date()}")
    print(f"Products               : {len(products):,}")
    print(f"Warehouses              : {len(warehouses):,}")
    print(f"Customers               : {len(customers):,}")
    print(f"Sales target rows       : {args.sales_target_rows:,}")
    print(f"Sales rows generated    : {len(sales):,}")
    print(f"Sales date coverage     : {report['sales_date_min']} to {report['sales_date_max']}")
    print(f"Controlled anomalies    : {len(anomaly_log):,}")
    print("Referential integrity   : PASS")
    print("\nRows written:")
    for filename, count in row_counts.items():
        print(f"  {filename:<32} {count:>10,}")
    print("  generation_report.json          written")
    print("\nGeneration completed. Review anomaly_log.csv before cleaning.")


if __name__ == "__main__":
    main()
