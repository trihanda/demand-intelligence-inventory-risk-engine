"""
Generate reproducible synthetic datasets for the Demand Intelligence & Inventory Risk Engine.

Purpose:
    Generate relational synthetic data for a fictional Indonesia-based B2B technology
    provider. The data supports demand forecasting and pre-PR inventory decisions, while retaining
    quotation and subscription datasets as adjacent business context. Controlled inventory
    scenarios are validated at SKU/warehouse grain, with explicit warehouse IDs in the log.

Input:
    No external files required.

Outputs:
    CSV files under data/synthetic/, separate data-quality and business-scenario logs,
    and generation_report.json.

Usage:
    python src/generate_synthetic_data.py
    python src/generate_synthetic_data.py --seed 42
    python src/generate_synthetic_data.py --products 100 --customers 250 --sales-target-rows 30000
    python src/generate_synthetic_data.py --output-dir data/synthetic
    python src/generate_synthetic_data.py --collective-leave-policy workday

Notes:
    Calendar uses official Indonesia SKB public holidays/collective leave (2023-2026); the
    collective-leave treatment is configurable because private-sector implementation varies.
    All records are synthetic. Lead-time and MOQ values are planning assumptions, not
    supplier commitments. Inventory is a single as-of snapshot; historical stock movements
    are not reconstructed in this MVP. Price-shock monitoring is outside core scope.
    business_scenario_log.csv stores test expectations; anomaly_log.csv stores data issues.
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
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2026-09-30"
DEFAULT_PRODUCTS = 1000
DEFAULT_WAREHOUSES = 3
DEFAULT_CUSTOMERS = 5000
DEFAULT_SALES_TARGET_ROWS = 625_000

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
WAREHOUSE_CITY = "Jakarta"  # Version 1 treats warehouses as separate buildings in one city.

# Indonesia national holidays and collective leave based on official government SKB
# calendars for 2023-2026. Dates are stored in ISO format and clipped to the
# configured dataset range when the calendar is generated.
# Private-sector employers may decide whether to observe collective leave; the
# generator defaults to observing it, but exposes a CLI policy override.
DEFAULT_COLLECTIVE_LEAVE_POLICY = "observe"  # "observe" or "workday"

PUBLIC_HOLIDAYS: dict[int, list[tuple[str, str]]] = {
    2023: [
        ("2023-01-01", "Tahun Baru Masehi"),
        ("2023-01-22", "Tahun Baru Imlek 2574 Kongzili"),
        ("2023-02-18", "Isra Mikraj Nabi Muhammad SAW"),
        ("2023-03-22", "Hari Suci Nyepi Tahun Baru Saka 1945"),
        ("2023-04-07", "Wafat Isa Almasih"),
        ("2023-04-22", "Hari Raya Idul Fitri 1444 Hijriah"),
        ("2023-04-23", "Hari Raya Idul Fitri 1444 Hijriah"),
        ("2023-05-01", "Hari Buruh Internasional"),
        ("2023-05-18", "Kenaikan Isa Almasih"),
        ("2023-06-01", "Hari Lahir Pancasila"),
        ("2023-06-04", "Hari Raya Waisak 2567 BE"),
        ("2023-06-29", "Hari Raya Idul Adha 1444 Hijriah"),
        ("2023-07-19", "Tahun Baru Islam 1445 Hijriah"),
        ("2023-08-17", "Hari Kemerdekaan Republik Indonesia"),
        ("2023-09-28", "Maulid Nabi Muhammad SAW"),
        ("2023-12-25", "Hari Raya Natal"),
    ],
    2024: [
        ("2024-01-01", "Tahun Baru Masehi"),
        ("2024-02-08", "Isra Mikraj Nabi Muhammad SAW"),
        ("2024-02-10", "Tahun Baru Imlek 2575 Kongzili"),
        ("2024-03-11", "Hari Suci Nyepi Tahun Baru Saka 1946"),
        ("2024-03-29", "Wafat Yesus Kristus"),
        ("2024-03-31", "Hari Kebangkitan Yesus Kristus (Paskah)"),
        ("2024-04-10", "Hari Raya Idul Fitri 1445 Hijriah"),
        ("2024-04-11", "Hari Raya Idul Fitri 1445 Hijriah"),
        ("2024-05-01", "Hari Buruh Internasional"),
        ("2024-05-09", "Kenaikan Yesus Kristus"),
        ("2024-05-23", "Hari Raya Waisak 2568 BE"),
        ("2024-06-01", "Hari Lahir Pancasila"),
        ("2024-06-17", "Hari Raya Idul Adha 1445 Hijriah"),
        ("2024-07-07", "Tahun Baru Islam 1446 Hijriah"),
        ("2024-08-17", "Hari Kemerdekaan Republik Indonesia"),
        ("2024-09-16", "Maulid Nabi Muhammad SAW"),
        ("2024-12-25", "Hari Raya Natal"),
    ],
    2025: [
        ("2025-01-01", "Tahun Baru Masehi"),
        ("2025-01-27", "Isra Mikraj Nabi Muhammad SAW"),
        ("2025-01-29", "Tahun Baru Imlek 2576 Kongzili"),
        ("2025-03-29", "Hari Suci Nyepi Tahun Baru Saka 1947"),
        ("2025-03-31", "Hari Raya Idul Fitri 1446 Hijriah"),
        ("2025-04-01", "Hari Raya Idul Fitri 1446 Hijriah"),
        ("2025-04-18", "Wafat Yesus Kristus"),
        ("2025-04-20", "Hari Kebangkitan Yesus Kristus (Paskah)"),
        ("2025-05-01", "Hari Buruh Internasional"),
        ("2025-05-12", "Hari Raya Waisak 2569 BE"),
        ("2025-05-29", "Kenaikan Yesus Kristus"),
        ("2025-06-01", "Hari Lahir Pancasila"),
        ("2025-06-06", "Hari Raya Idul Adha 1446 Hijriah"),
        ("2025-06-27", "1 Muharam Tahun Baru Islam 1447 Hijriah"),
        ("2025-08-17", "Proklamasi Kemerdekaan Republik Indonesia"),
        ("2025-09-05", "Maulid Nabi Muhammad SAW"),
        ("2025-12-25", "Kelahiran Yesus Kristus"),
    ],
    2026: [
        ("2026-01-01", "Tahun Baru Masehi"),
        ("2026-01-16", "Isra Mikraj Nabi Muhammad SAW"),
        ("2026-02-17", "Tahun Baru Imlek 2577 Kongzili"),
        ("2026-03-19", "Hari Suci Nyepi Tahun Baru Saka 1948"),
        ("2026-03-21", "Hari Raya Idul Fitri 1447 Hijriah"),
        ("2026-03-22", "Hari Raya Idul Fitri 1447 Hijriah"),
        ("2026-04-03", "Wafat Yesus Kristus"),
        ("2026-04-05", "Hari Kebangkitan Yesus Kristus (Paskah)"),
        ("2026-05-01", "Hari Buruh Internasional"),
        ("2026-05-14", "Kenaikan Yesus Kristus"),
        ("2026-05-27", "Hari Raya Idul Adha 1447 Hijriah"),
        ("2026-05-31", "Hari Raya Waisak 2570 BE"),
        ("2026-06-01", "Hari Lahir Pancasila"),
        ("2026-06-16", "1 Muharam Tahun Baru Islam 1448 Hijriah"),
        ("2026-08-17", "Proklamasi Kemerdekaan Republik Indonesia"),
        ("2026-08-25", "Maulid Nabi Muhammad SAW"),
        ("2026-12-25", "Kelahiran Yesus Kristus"),
    ],
}

COLLECTIVE_LEAVE: dict[int, list[tuple[str, str]]] = {
    2023: [
        ("2023-01-23", "Cuti Bersama Tahun Baru Imlek"),
        ("2023-03-23", "Cuti Bersama Hari Suci Nyepi"),
        ("2023-04-19", "Cuti Bersama Idul Fitri 1444 Hijriah"),
        ("2023-04-20", "Cuti Bersama Idul Fitri 1444 Hijriah"),
        ("2023-04-21", "Cuti Bersama Idul Fitri 1444 Hijriah"),
        ("2023-04-24", "Cuti Bersama Idul Fitri 1444 Hijriah"),
        ("2023-04-25", "Cuti Bersama Idul Fitri 1444 Hijriah"),
        ("2023-06-02", "Cuti Bersama Hari Raya Waisak"),
        ("2023-06-28", "Cuti Bersama Idul Adha 1444 Hijriah"),
        ("2023-06-30", "Cuti Bersama Idul Adha 1444 Hijriah"),
        ("2023-12-26", "Cuti Bersama Hari Raya Natal"),
    ],
    2024: [
        ("2024-02-09", "Cuti Bersama Tahun Baru Imlek"),
        ("2024-03-12", "Cuti Bersama Hari Suci Nyepi"),
        ("2024-04-08", "Cuti Bersama Idul Fitri 1445 Hijriah"),
        ("2024-04-09", "Cuti Bersama Idul Fitri 1445 Hijriah"),
        ("2024-04-12", "Cuti Bersama Idul Fitri 1445 Hijriah"),
        ("2024-04-15", "Cuti Bersama Idul Fitri 1445 Hijriah"),
        ("2024-05-10", "Cuti Bersama Kenaikan Yesus Kristus"),
        ("2024-05-24", "Cuti Bersama Hari Raya Waisak"),
        ("2024-06-18", "Cuti Bersama Idul Adha 1445 Hijriah"),
        ("2024-12-26", "Cuti Bersama Hari Raya Natal"),
    ],
    2025: [
        ("2025-01-28", "Cuti Bersama Tahun Baru Imlek"),
        ("2025-03-28", "Cuti Bersama Hari Suci Nyepi"),
        ("2025-04-02", "Cuti Bersama Idul Fitri 1446 Hijriah"),
        ("2025-04-03", "Cuti Bersama Idul Fitri 1446 Hijriah"),
        ("2025-04-04", "Cuti Bersama Idul Fitri 1446 Hijriah"),
        ("2025-04-07", "Cuti Bersama Idul Fitri 1446 Hijriah"),
        ("2025-05-13", "Cuti Bersama Hari Raya Waisak"),
        ("2025-05-30", "Cuti Bersama Kenaikan Yesus Kristus"),
        ("2025-06-09", "Cuti Bersama Idul Adha 1446 Hijriah"),
        ("2025-08-18", "Cuti Bersama HUT ke-80 Kemerdekaan RI"),
        ("2025-12-26", "Cuti Bersama Hari Raya Natal"),
    ],
    2026: [
        ("2026-02-16", "Cuti Bersama Tahun Baru Imlek"),
        ("2026-03-18", "Cuti Bersama Hari Suci Nyepi"),
        ("2026-03-20", "Cuti Bersama Idul Fitri 1447 Hijriah"),
        ("2026-03-23", "Cuti Bersama Idul Fitri 1447 Hijriah"),
        ("2026-03-24", "Cuti Bersama Idul Fitri 1447 Hijriah"),
        ("2026-05-15", "Cuti Bersama Kenaikan Yesus Kristus"),
        ("2026-05-28", "Cuti Bersama Idul Adha 1447 Hijriah"),
        ("2026-12-24", "Cuti Bersama Kelahiran Yesus Kristus"),
    ],
}

# No extra company-wide closure dates are assumed in this version. Add dates here
# only when a specific fictional-company policy is intentionally part of a scenario.
COMPANY_CLOSURE_RANGES: list[tuple[str, str, str]] = []

HOLIDAY_CALENDAR_SOURCES = {
    "2023": "SKB 3 Menteri 2023, including amendments in March and June 2023",
    "2024": "SKB 3 Menteri 2024",
    "2025": "SKB 3 Menteri 2025, including 18 August collective leave amendment",
    "2026": "SKB 3 Menteri 2026",
}
HOLIDAY_CALENDAR_SOURCE_URLS = {
    "2023": [
        "https://setkab.go.id/pemerintah-terbitkan-skb-perubahan-libur-nasional-dan-cuti-bersama-2023/",
        "https://setkab.go.id/pemerintah-terbitkan-skb-perubahan-libur-nasional-dan-cuti-bersama-2023-2/",
    ],
    "2024": ["https://setkab.go.id/inilah-skb-3-menteri-libur-nasional-dan-cuti-bersama-2024/"],
    "2025": [
        "https://setkab.go.id/pemerintah-tetapkan-hari-libur-nasional-dan-cuti-bersama-tahun-2025/",
        "https://www.setneg.go.id/baca/index/sambut_hut_ke_80_kemerdekaan_ri_pemerintah_tetapkan_18_agustus_2025_sebagai_cuti_bersama",
    ],
    "2026": ["https://www.setneg.go.id/baca/index/inilah_skb_3_menteri_libur_nasional_dan_cuti_bersama_2026"],
}


def summarize_hardware_demand_patterns(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    closure_dates: set[pd.Timestamp],
) -> dict[str, dict[str, Any]]:
    """Summarize empirical weekly demand by synthetic seed label for QA reporting.

    Uses hardware SKUs only, excludes the final incomplete week, and excludes weeks
    with no valid business days from occurrence metrics. This is diagnostic metadata,
    not a feature to pass into a forecasting model.
    """
    hardware = products.loc[products["physical_inventory_flag"].astype(bool), ["sku", "demand_type_seed"]].copy()
    if hardware.empty:
        return {}

    business_dates = make_business_dates(start, end, closure_dates)
    week_ends = pd.date_range(start=start, end=end, freq="W-SUN")
    complete_week_ends = [
        week_end for week_end in week_ends
        if week_end.normalize() <= end.normalize()
        and week_end.normalize() >= start.normalize() + pd.Timedelta(days=6)
    ]
    valid_week_ends: list[pd.Timestamp] = []
    for week_end in complete_week_ends:
        week_start = week_end - pd.Timedelta(days=6)
        if len(business_dates[(business_dates >= week_start) & (business_dates <= week_end)]) > 0:
            valid_week_ends.append(week_end)
    if not valid_week_ends:
        return {}

    tx = sales.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    tx = tx.loc[
        tx["sku"].isin(hardware["sku"])
        & tx["transaction_date"].notna()
        & tx["quantity"].gt(0)
    ].copy()
    tx["week_end"] = tx["transaction_date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    tx = tx.loc[tx["week_end"].isin(valid_week_ends)]
    weekly = tx.groupby(["sku", "week_end"], as_index=False)["quantity"].sum()
    full_index = pd.MultiIndex.from_product(
        [hardware["sku"].tolist(), valid_week_ends], names=["sku", "week_end"]
    )
    weekly = weekly.set_index(["sku", "week_end"])["quantity"].reindex(full_index, fill_value=0).rename("quantity").reset_index()

    sku_rows: list[dict[str, Any]] = []
    for sku, group in weekly.groupby("sku", sort=False):
        values = group["quantity"].to_numpy(dtype=float)
        positive = values[values > 0]
        if positive.size == 0:
            adi = None
            cv2 = None
        else:
            adi = float(len(values) / len(positive))
            mean_positive = float(positive.mean())
            cv2 = float((positive.std(ddof=0) / mean_positive) ** 2) if mean_positive > 0 else None
        sku_rows.append({"sku": sku, "adi": adi, "cv2_positive_demand": cv2})
    sku_metrics = pd.DataFrame(sku_rows).merge(hardware, on="sku", how="left", validate="one_to_one")

    summary: dict[str, dict[str, Any]] = {}
    for label, group in sku_metrics.groupby("demand_type_seed", sort=True):
        adi = group["adi"].dropna()
        cv2 = group["cv2_positive_demand"].dropna()
        summary[str(label)] = {
            "sku_count": int(group["sku"].nunique()),
            "median_adi": round(float(adi.median()), 4) if len(adi) else None,
            "median_cv2_positive_demand": round(float(cv2.median()), 4) if len(cv2) else None,
            "median_positive_week_share": round(float((1.0 / adi).median()), 4) if len(adi) else None,
            "weeks_in_metrics": int(len(valid_week_ends)),
            "final_incomplete_week_excluded": True,
        }
    return summary


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
    parser.add_argument("--collective-leave-policy", choices=["observe", "workday"], default=DEFAULT_COLLECTIVE_LEAVE_POLICY,
                        help="Whether the fictional private company treats government collective leave as non-business days.")
    return parser.parse_args()


def as_date(value: str) -> pd.Timestamp:
    """Parse a date and normalize it to midnight."""
    return pd.Timestamp(value).normalize()


def make_calendar(
    start: pd.Timestamp,
    end: pd.Timestamp,
    collective_leave_policy: str = DEFAULT_COLLECTIVE_LEAVE_POLICY,
) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """Build an Indonesia calendar with separate holiday, collective-leave and company flags.

    Public holidays are always non-business weekdays. Collective leave is recorded
    independently and is non-business only when policy='observe'. Weekends are
    non-business regardless of holiday flags. Company-specific closures are empty
    by default and must be explicitly configured in COMPANY_CLOSURE_RANGES.
    """
    if collective_leave_policy not in {"observe", "workday"}:
        raise ValueError("collective_leave_policy must be 'observe' or 'workday'.")

    public_map: dict[pd.Timestamp, str] = {}
    collective_map: dict[pd.Timestamp, str] = {}
    company_map: dict[pd.Timestamp, str] = {}

    for year_events in PUBLIC_HOLIDAYS.values():
        for date_text, name in year_events:
            date = pd.Timestamp(date_text).normalize()
            if start <= date <= end:
                public_map[date] = name
    for year_events in COLLECTIVE_LEAVE.values():
        for date_text, name in year_events:
            date = pd.Timestamp(date_text).normalize()
            if start <= date <= end:
                collective_map[date] = name
    for range_start, range_end, reason in COMPANY_CLOSURE_RANGES:
        for date in pd.date_range(range_start, range_end, freq="D"):
            if start <= date <= end:
                company_map[date.normalize()] = reason

    calendar = pd.DataFrame({"calendar_date": pd.date_range(start, end, freq="D")})
    calendar["day_of_week"] = calendar["calendar_date"].dt.day_name()
    calendar["is_weekday"] = calendar["calendar_date"].dt.dayofweek < 5
    calendar["is_weekend"] = ~calendar["is_weekday"]
    calendar["is_public_holiday"] = calendar["calendar_date"].isin(public_map)
    calendar["public_holiday_name"] = calendar["calendar_date"].map(public_map).fillna("")
    calendar["is_collective_leave"] = calendar["calendar_date"].isin(collective_map)
    calendar["collective_leave_name"] = calendar["calendar_date"].map(collective_map).fillna("")
    calendar["is_company_closure"] = calendar["calendar_date"].isin(company_map)
    calendar["company_closure_name"] = calendar["calendar_date"].map(company_map).fillna("")
    observes_collective_leave = collective_leave_policy == "observe"
    calendar["collective_leave_is_non_business"] = bool(observes_collective_leave)
    calendar["is_business_day"] = (
        calendar["is_weekday"]
        & ~calendar["is_public_holiday"]
        & ~(calendar["is_collective_leave"] & observes_collective_leave)
        & ~calendar["is_company_closure"]
    )

    def make_closure_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if bool(row["is_public_holiday"]):
            reasons.append(f"public_holiday:{row['public_holiday_name']}")
        if bool(row["is_collective_leave"]) and observes_collective_leave:
            reasons.append(f"collective_leave:{row['collective_leave_name']}")
        if bool(row["is_company_closure"]):
            reasons.append(f"company_closure:{row['company_closure_name']}")
        return ";".join(reasons)

    calendar["closure_reason"] = calendar.apply(make_closure_reason, axis=1)

    def classify_day(row: pd.Series) -> str:
        if bool(row["is_weekend"]):
            return "weekend"
        if bool(row["is_public_holiday"]):
            return "public_holiday"
        if bool(row["is_company_closure"]):
            return "company_closure"
        if bool(row["is_collective_leave"]):
            return "collective_leave" if observes_collective_leave else "collective_leave_business_day"
        return "business_day"

    calendar["business_day_type"] = calendar.apply(classify_day, axis=1)
    calendar["holiday_calendar_policy"] = "Indonesia_SKB_2023_2026"
    calendar["collective_leave_policy"] = collective_leave_policy
    calendar["quarter"] = "Q" + calendar["calendar_date"].dt.quarter.astype(str)
    calendar["year"] = calendar["calendar_date"].dt.year
    calendar["month"] = calendar["calendar_date"].dt.month

    non_business_dates = set(public_map)
    if observes_collective_leave:
        non_business_dates.update(collective_map)
    non_business_dates.update(company_map)
    return calendar, non_business_dates


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
                "product_family_name": family,
                "base_uom": ("license" if product_type == "software_license" else
                             "subscription" if product_type == "subscription" else
                             "reel" if family in {"Fiber Cable", "Copper Cable"} else "unit"),
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
        "city": WAREHOUSE_CITY,
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
        {"supplier_id": "SUP-0001", "supplier_name": "Synthetic Asia Hardware Principal", "supplier_country": "China", "procurement_channel": "direct_import", "default_currency": "USD", "supplier_type": "hardware_vendor"},
        {"supplier_id": "SUP-0002", "supplier_name": "Synthetic Regional Technology Vendor", "supplier_country": "Singapore", "procurement_channel": "direct_import", "default_currency": "USD", "supplier_type": "hardware_vendor"},
        {"supplier_id": "SUP-0003", "supplier_name": "Synthetic Indonesian Distributor", "supplier_country": "Indonesia", "procurement_channel": "local_distributor", "default_currency": "IDR", "supplier_type": "hardware_distributor"},
        {"supplier_id": "SUP-0004", "supplier_name": "Synthetic Indonesian IT Importer", "supplier_country": "Indonesia", "procurement_channel": "local_importer", "default_currency": "IDR", "supplier_type": "hardware_distributor"},
        {"supplier_id": "SUP-0005", "supplier_name": "Synthetic Global Software Vendor", "supplier_country": "United States", "procurement_channel": "digital_vendor", "default_currency": "USD", "supplier_type": "software_vendor"},
        {"supplier_id": "SUP-0006", "supplier_name": "Synthetic Indonesian Software Partner", "supplier_country": "Indonesia", "procurement_channel": "digital_vendor", "default_currency": "IDR", "supplier_type": "software_partner"},
    ])

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
    return products, warehouses, customers, suppliers, replacements


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
    """Generate distinct sales lines efficiently while preserving weekly SKU demand totals."""
    business_dates = make_business_dates(start, end, closure_dates)
    if business_dates.empty:
        raise ValueError("The selected date range has no valid business days.")

    weeks = pd.date_range(start=start, end=end, freq="W-SUN")
    if len(weeks) == 0:
        weeks = pd.DatetimeIndex([end.normalize()])
    if weeks[-1].normalize() < end.normalize():
        weeks = weeks.append(pd.DatetimeIndex([weeks[-1] + pd.Timedelta(days=7)]))

    # Precompute business dates, quarter factors, and exposure per week.
    # A full business week has five business days. Weeks shortened by configured
    # closures or by the configured data boundary receive proportionally less demand,
    # preventing a full weekly quantity from being concentrated into only a few days.
    week_dates: list[np.ndarray] = []
    quarter_factors = np.empty(len(weeks), dtype=float)
    week_exposure = np.zeros(len(weeks), dtype=float)
    for week_idx, week_end in enumerate(weeks):
        week_start = max(start.normalize(), (week_end - pd.Timedelta(days=6)).normalize())
        bounded_week_end = min(end.normalize(), week_end.normalize())
        date_slice = business_dates[
            (business_dates >= week_start) & (business_dates <= bounded_week_end)
        ]
        week_dates.append(date_slice.to_numpy(dtype="datetime64[ns]"))
        week_exposure[week_idx] = min(1.0, len(date_slice) / 5.0)
        anchor = min(max(week_end, start), end)
        quarter_factors[week_idx] = quarter_factor(anchor)
    elapsed_years = np.arange(len(weeks), dtype=float) / 52.0
    week_indices_all = np.arange(len(weeks), dtype=int)
    week_has_business_dates = np.array([len(dates) > 0 for dates in week_dates], dtype=bool)

    # Create one aggregated quantity per active SKU-week first. This is much faster
    # than generating hundreds of thousands of invoice lines inside nested Python loops.
    active_sku_indices: list[np.ndarray] = []
    active_week_indices: list[np.ndarray] = []
    active_week_quantities: list[np.ndarray] = []
    product_rows = list(products.itertuples(index=False))
    for product_idx, product in enumerate(product_rows):
        base = float(rng.uniform(4, 22))
        pattern = str(product.demand_type_seed)
        expected = base * quarter_factors.copy()
        if pattern == "smooth":
            expected *= 1 + 0.04 * elapsed_years
        elif pattern == "intermittent":
            # Inactive weeks must remain exactly zero. On active weeks, keep demand
            # size relatively stable so this class differs from lumpy demand by timing.
            active_mask = rng.random(len(weeks)) <= 0.22
            expected[active_mask] *= rng.uniform(1.8, 2.2, size=int(active_mask.sum()))
            expected[~active_mask] = 0.0
        elif pattern == "erratic":
            # A mean-one lognormal multiplier preserves approximate average demand
            # while creating substantial week-to-week size variability. Sigma=1.0
            # is deliberately stronger than a narrow uniform multiplier so the class
            # is distinguishable using empirical CV^2.
            sigma = 1.0
            multiplier = rng.lognormal(mean=-(sigma ** 2) / 2.0, sigma=sigma, size=len(weeks))
            expected *= multiplier
        elif pattern == "lumpy":
            # Lumpy demand combines infrequent occurrence with highly variable event
            # sizes. Event weeks receive a mean-one lognormal multiplier; inactive
            # weeks remain zero.
            active_mask = rng.random(len(weeks)) <= 0.10
            sigma = 0.95
            multiplier = rng.lognormal(
                mean=-(sigma ** 2) / 2.0, sigma=sigma, size=int(active_mask.sum())
            )
            expected[active_mask] *= multiplier
            expected[~active_mask] = 0.0

        # Scale demand to the number of available business days. This handles partial
        # weeks and holiday/closure weeks without packing a full week's demand into
        # a shortened date window.
        expected *= week_exposure
        # Do not floor expected demand to a positive value: Poisson(0) is exactly 0,
        # which preserves intended inactive weeks for intermittent and lumpy SKUs.
        quantities = rng.poisson(np.maximum(expected, 0.0)).astype(np.int32)
        active_mask = (quantities > 0) & week_has_business_dates
        if active_mask.any():
            selected_weeks = week_indices_all[active_mask]
            n_active = len(selected_weeks)
            active_sku_indices.append(np.full(n_active, product_idx, dtype=np.int32))
            active_week_indices.append(selected_weeks.astype(np.int32))
            active_week_quantities.append(quantities[active_mask])

    if not active_week_quantities:
        return pd.DataFrame(columns=[
            "invoice_line_id", "invoice_id", "transaction_date", "sku", "warehouse_id",
            "customer_id", "quantity", "unit_price", "line_amount", "transaction_currency",
            "exchange_rate_date", "exchange_rate_to_usd", "unit_price_usd", "line_amount_usd",
            "transaction_status",
        ])

    sku_week_product_idx = np.concatenate(active_sku_indices)
    sku_week_idx = np.concatenate(active_week_indices)
    sku_week_qty = np.concatenate(active_week_quantities)

    # Use the requested row count to control invoice-line granularity after observing
    # the actual number of active SKU-weeks. This avoids a rough active-share estimate
    # inflating the dataset. A SKU-week with positive demand always keeps at least one
    # line; extra lines can be created only when its weekly quantity supports them.
    max_line_counts = np.minimum(sku_week_qty, 12).astype(np.int32)
    line_counts = np.ones(len(sku_week_qty), dtype=np.int32)
    remaining_rows = max(0, int(target_rows) - int(len(line_counts)))
    while remaining_rows > 0:
        eligible = np.flatnonzero(line_counts < max_line_counts)
        if eligible.size == 0:
            break
        additions = min(remaining_rows, int(eligible.size))
        chosen = rng.choice(eligible, size=additions, replace=False)
        line_counts[chosen] += 1
        remaining_rows -= additions

    # Split each SKU-week quantity into positive invoice-line quantities. Grouping by
    # line count allows vectorized multinomial-equivalent binomial draws while preserving
    # the weekly quantity total exactly.
    row_sku_idx_parts: list[np.ndarray] = []
    row_week_idx_parts: list[np.ndarray] = []
    row_qty_parts: list[np.ndarray] = []
    for line_count in range(1, int(line_counts.max()) + 1):
        group_positions = np.flatnonzero(line_counts == line_count)
        if group_positions.size == 0:
            continue
        q = sku_week_qty[group_positions].astype(np.int64)
        if line_count == 1:
            split = q.reshape(-1, 1)
        else:
            split = np.ones((len(group_positions), line_count), dtype=np.int64)
            remaining = q - line_count
            for col in range(line_count - 1):
                allocated = rng.binomial(remaining, 1.0 / (line_count - col))
                split[:, col] += allocated
                remaining -= allocated
            split[:, -1] += remaining
        row_sku_idx_parts.append(np.repeat(sku_week_product_idx[group_positions], line_count))
        row_week_idx_parts.append(np.repeat(sku_week_idx[group_positions], line_count))
        row_qty_parts.append(split.reshape(-1))

    row_sku_idx = np.concatenate(row_sku_idx_parts).astype(np.int32)
    row_week_idx = np.concatenate(row_week_idx_parts).astype(np.int32)
    row_quantity = np.concatenate(row_qty_parts).astype(np.int32)

    # Draw a valid business date for each invoice line, grouped by week so no row-level
    # pandas Timestamp conversion or repeated search is required.
    order = np.argsort(row_week_idx, kind="stable")
    sorted_week_idx = row_week_idx[order]
    transaction_dates = np.empty(len(row_week_idx), dtype="datetime64[ns]")
    boundaries = np.searchsorted(sorted_week_idx, np.arange(len(weeks) + 1), side="left")
    for week_idx in range(len(weeks)):
        lo, hi = int(boundaries[week_idx]), int(boundaries[week_idx + 1])
        if hi <= lo:
            continue
        candidates = week_dates[week_idx]
        if len(candidates) == 0:
            raise RuntimeError(f"No business dates available for week index {week_idx}.")
        chosen = rng.choice(candidates, size=hi - lo, replace=True)
        transaction_dates[order[lo:hi]] = chosen

    customer_ids = customers["customer_id"].to_numpy(dtype=object)
    customer_currency = customers["customer_currency_preference"].to_numpy(dtype=object)
    warehouse_ids = warehouses["warehouse_id"].to_numpy(dtype=object)
    customer_selection = rng.integers(0, len(customer_ids), size=len(row_sku_idx))
    warehouse_selection = rng.integers(0, len(warehouse_ids), size=len(row_sku_idx))
    transaction_customer = customer_ids[customer_selection]
    transaction_warehouse = warehouse_ids[warehouse_selection]
    currency = customer_currency[customer_selection]

    # Resolve monthly FX rates in vectorized form.
    transaction_months = transaction_dates.astype("datetime64[M]")
    unique_months, month_indices = np.unique(transaction_months, return_inverse=True)
    fx_lookup = dict(FX_RATE_LOOKUP)
    default_fx = float(next(iter(fx_lookup.values())))
    rates_by_month = np.array([fx_lookup.get(str(month), default_fx) for month in unique_months], dtype=float)
    fx_rate = rates_by_month[month_indices]
    fx_rate = np.where(currency == "IDR", fx_rate, 1.0)

    sku_values = products["sku"].to_numpy(dtype=object)
    product_prices = products["unit_price"].to_numpy(dtype=float)
    line_unit_price_usd = product_prices[row_sku_idx] * rng.uniform(0.96, 1.04, size=len(row_sku_idx))
    unit_price_original = np.where(
        currency == "IDR",
        np.round(line_unit_price_usd / fx_rate),
        np.round(line_unit_price_usd, 2),
    )
    line_amount_original = np.round(unit_price_original * row_quantity, 2)
    unit_price_usd = np.round(unit_price_original * fx_rate, 2)
    line_amount_usd = np.round(line_amount_original * fx_rate, 2)
    month_starts = transaction_months.astype("datetime64[D]").astype(str)

    # The row target controls invoice-line granularity, not a hard cap. If natural
    # positive line counts differ slightly from the target, retain all rows so weekly
    # demand is not distorted by downsampling transactions after demand generation.
    n_rows = len(row_sku_idx)
    sales = pd.DataFrame({
        "invoice_line_id": [f"SIL-{i:09d}" for i in range(1, n_rows + 1)],
        "invoice_id": [f"INV-{i:08d}" for i in range(1, n_rows + 1)],
        "transaction_date": pd.to_datetime(transaction_dates),
        "sku": sku_values[row_sku_idx],
        "warehouse_id": transaction_warehouse,
        "customer_id": transaction_customer,
        "quantity": row_quantity,
        "unit_price": unit_price_original,
        "line_amount": line_amount_original,
        "transaction_currency": currency,
        "exchange_rate_date": month_starts,
        "exchange_rate_to_usd": fx_rate,
        "unit_price_usd": unit_price_usd,
        "line_amount_usd": line_amount_usd,
        "transaction_status": "posted",
    })
    sales = sales.sort_values("transaction_date", kind="stable").reset_index(drop=True)
    sales["invoice_line_id"] = [f"SIL-{i:09d}" for i in range(1, len(sales) + 1)]
    sales["invoice_id"] = [f"INV-{i:08d}" for i in range(1, len(sales) + 1)]
    return sales


def generate_sku_sourcing_policy(
    rng: np.random.Generator,
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Generate SKU-route sourcing policies for physical inventory-managed hardware.

    Values are explicitly labeled synthetic planning assumptions. They are starting
    points for testing, not universal procurement standards or measured supplier facts.
    """
    supplier_meta = suppliers.set_index("supplier_id").to_dict("index")
    route_templates = {
        # Typical lead-time components and P90 transit estimates are deliberately separate.
        "SUP-0001": {
            "transport_mode": "sea", "sourcing_lane": "China-to-Jakarta-sea",
            "internal_approval_days": 10, "principal_approval_days": 10,
            "indent_days": 56, "transit_days": 35, "transit_days_p90": 56,
            "receiving_days": 3, "buffer_days": 7,
            "transit_min_days": 28, "cost_factor": 1.00, "currency": "USD",
        },
        "SUP-0002": {
            "transport_mode": "air", "sourcing_lane": "Singapore-to-Jakarta-air",
            "internal_approval_days": 10, "principal_approval_days": 5,
            "indent_days": 14, "transit_days": 7, "transit_days_p90": 14,
            "receiving_days": 3, "buffer_days": 5,
            "transit_min_days": 3, "cost_factor": 1.08, "currency": "USD",
        },
        "SUP-0003": {
            "transport_mode": "road", "sourcing_lane": "Indonesia-local-distributor",
            "internal_approval_days": 7, "principal_approval_days": 0,
            "indent_days": 1, "transit_days": 4, "transit_days_p90": 7,
            "receiving_days": 2, "buffer_days": 3,
            "transit_min_days": 2, "cost_factor": 1.12, "currency": "IDR",
        },
        "SUP-0004": {
            "transport_mode": "road", "sourcing_lane": "Indonesia-local-importer",
            "internal_approval_days": 7, "principal_approval_days": 0,
            "indent_days": 7, "transit_days": 7, "transit_days_p90": 14,
            "receiving_days": 2, "buffer_days": 4,
            "transit_min_days": 3, "cost_factor": 1.09, "currency": "IDR",
        },
    }
    hardware = products.loc[products["physical_inventory_flag"]].copy()
    rows: list[dict[str, Any]] = []
    policy_counter = 1
    all_routes = list(route_templates)

    for product in hardware.itertuples(index=False):
        category = str(product.product_category)
        family = str(getattr(product, "product_family_name", ""))
        # Route mix varies by product family rather than forcing a single sourcing path.
        if category == "Cabling, Power & Accessories":
            route_probs = np.array([0.32, 0.08, 0.38, 0.22])
        elif category in {"Servers & Storage", "Cybersecurity Products", "Networking Equipment"}:
            route_probs = np.array([0.30, 0.20, 0.30, 0.20])
        else:
            route_probs = np.array([0.20, 0.12, 0.43, 0.25])
        primary = str(rng.choice(all_routes, p=route_probs))
        selected_routes = [primary]
        if rng.random() < 0.25:
            alternative_choices = [r for r in all_routes if r != primary]
            # Alternative route is a documented possibility, not guaranteed capacity.
            selected_routes.append(str(rng.choice(alternative_choices)))

        for priority, supplier_id in enumerate(selected_routes, start=1):
            t = route_templates[supplier_id]
            supplier = supplier_meta[supplier_id]
            is_cable = family in {"Fiber Cable", "Copper Cable"}
            is_accessory = family in {"Transceiver", "Rack Accessory", "Server Memory", "Network Interface"}
            if supplier_id == "SUP-0001":
                moq_range = (5, 12) if is_cable else ((3, 8) if is_accessory else (1, 3))
                multiples = [1, 5, 10] if is_cable else [1, 2, 5]
            elif supplier_id == "SUP-0002":
                moq_range = (2, 6) if (is_cable or is_accessory) else (1, 2)
                multiples = [1, 2, 5]
            elif supplier_id == "SUP-0003":
                moq_range = (1, 4) if not is_cable else (1, 5)
                multiples = [1, 1, 2, 5]
            else:
                moq_range = (2, 6) if (is_cable or is_accessory) else (1, 3)
                multiples = [1, 2, 5]
            moq = int(rng.integers(moq_range[0], moq_range[1] + 1))
            order_multiple = int(rng.choice(multiples))
            planned_typical = int(
                t["internal_approval_days"] + t["principal_approval_days"] + 1
                + t["indent_days"] + t["transit_days"] + t["receiving_days"]
            )
            conservative = int(
                t["internal_approval_days"] + t["principal_approval_days"] + 1
                + t["indent_days"] + t["transit_days_p90"] + t["receiving_days"] + t["buffer_days"]
            )
            rows.append({
                "policy_id": f"POLICY-{policy_counter:07d}",
                "sku": str(product.sku),
                "supplier_id": supplier_id,
                "supplier_country": supplier["supplier_country"],
                "procurement_channel": supplier["procurement_channel"],
                "transport_mode": t["transport_mode"],
                "sourcing_lane": t["sourcing_lane"],
                "currency": t["currency"],
                "route_priority": priority,
                "is_preferred_route": priority == 1,
                "internal_approval_days": int(t["internal_approval_days"]),
                "principal_approval_days": int(t["principal_approval_days"]),
                "indent_days": int(t["indent_days"]),
                "transit_days": int(t["transit_days"]),
                "transit_days_p90": int(t["transit_days_p90"]),
                "receiving_days": int(t["receiving_days"]),
                "buffer_days": int(t["buffer_days"]),
                "planned_total_lead_time_days": planned_typical,
                "conservative_total_lead_time_days": conservative,
                "moq_quantity": moq,
                "order_multiple": order_multiple,
                "moq_unit": str(product.base_uom),
                "effective_from": start.date().isoformat(),
                "effective_to": "",
                "lead_time_basis": "synthetic_planning_assumption",
            })
            policy_counter += 1

    policy = pd.DataFrame(rows)

    # Case-test MOQ targets are selected in main() after recent sales demand is available.
    moq_skus: list[str] = []
    supplier_delay_candidates = policy.loc[
        policy["is_preferred_route"] & policy["supplier_id"].eq("SUP-0001"), "sku"
    ].drop_duplicates().tolist()
    policy.attrs["supplier_delay_candidate_skus"] = supplier_delay_candidates
    policy.attrs["moq_surplus_skus"] = moq_skus
    policy.attrs["route_templates"] = route_templates
    return policy

def completed_weekly_demand_by_sku_warehouse(
    sales: pd.DataFrame,
    as_of_date: pd.Timestamp,
    weeks: int = 52,
) -> dict[tuple[str, str], float]:
    """Return mean positive posted demand per complete calendar week by SKU/warehouse.

    The last complete Sunday is used as the cutoff. A partial week after that cutoff
    is excluded so inventory coverage and scenario assertions use the same denominator.
    Weeks with no sales remain implicit zeros because totals are divided by all `weeks`.
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    days_since_sunday = (as_of.weekday() + 1) % 7
    last_complete_week_end = as_of - pd.Timedelta(days=days_since_sunday)
    first_week_start = last_complete_week_end - pd.Timedelta(days=weeks * 7 - 1)

    tx = sales.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce").dt.normalize()
    tx = tx.loc[
        tx["transaction_date"].between(first_week_start, last_complete_week_end)
        & tx["transaction_status"].eq("posted")
        & tx["quantity"].gt(0)
    ]
    if tx.empty:
        return {}
    grouped = tx.groupby(["sku", "warehouse_id"], as_index=False)["quantity"].sum()
    return {
        (str(row.sku), str(row.warehouse_id)): float(row.quantity) / float(weeks)
        for row in grouped.itertuples(index=False)
    }


def select_scenario_warehouse_targets(
    scenario_targets: dict[str, list[str]],
    demand_by_sku_warehouse: dict[tuple[str, str], float],
    warehouse_ids: list[str],
) -> dict[str, dict[str, str]]:
    """Select an explicit target warehouse for every controlled scenario SKU.

    All scenarios target a warehouse with meaningful observed demand. MOQ tests use the
    strongest-demand warehouse, and their synthetic MOQ is calibrated to create a visible
    but bounded excess-coverage case rather than exploiting an almost-zero-demand site.
    """
    result: dict[str, dict[str, str]] = {}
    for group_name, skus in scenario_targets.items():
        sku_to_wh: dict[str, str] = {}
        for sku_value in skus:
            sku = str(sku_value)
            scored = [
                (wh, float(demand_by_sku_warehouse.get((sku, wh), 0.0)))
                for wh in warehouse_ids
            ]
            positive = [(wh, demand) for wh, demand in scored if demand > 0]
            if not positive:
                # Keep the scenario reference explicit, but validation will fail if the
                # intended demand-driven scenario has no demand at any warehouse.
                sku_to_wh[sku] = warehouse_ids[0]
            else:
                sku_to_wh[sku] = max(positive, key=lambda item: item[1])[0]
        result[group_name] = sku_to_wh
    return result


def generate_inventory(
    rng: np.random.Generator,
    products: pd.DataFrame,
    warehouses: pd.DataFrame,
    sales: pd.DataFrame,
    as_of_date: pd.Timestamp,
    scenario_targets: dict[str, list[str]],
    scenario_warehouse_targets: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Generate an as-of stock snapshot at SKU/warehouse grain.

    Stock is scaled to trailing demand for the same SKU and warehouse. Controlled
    stockout, overstock, supplier-delay, and MOQ cases are applied only at their
    designated warehouse, not against SKU-level aggregated stock.
    """
    hardware = products.loc[products["physical_inventory_flag"]].copy()
    weekly_demand_map = completed_weekly_demand_by_sku_warehouse(sales, as_of_date, weeks=52)
    target_sets = {key: set(str(v) for v in values) for key, values in scenario_targets.items()}

    rows: list[dict[str, Any]] = []
    warehouse_ids = warehouses["warehouse_id"].astype(str).tolist()
    for product in hardware.itertuples(index=False):
        sku = str(product.sku)
        for warehouse_id in warehouse_ids:
            demand = float(weekly_demand_map.get((sku, warehouse_id), 0.0))
            scenario_group = None
            if sku in target_sets.get("stockout_skus", set()) and scenario_warehouse_targets.get("stockout_skus", {}).get(sku) == warehouse_id:
                scenario_group = "stockout_skus"
            elif sku in target_sets.get("supplier_delay_skus", set()) and scenario_warehouse_targets.get("supplier_delay_skus", {}).get(sku) == warehouse_id:
                scenario_group = "supplier_delay_skus"
            elif sku in target_sets.get("overstock_skus", set()) and scenario_warehouse_targets.get("overstock_skus", {}).get(sku) == warehouse_id:
                scenario_group = "overstock_skus"
            elif sku in target_sets.get("moq_surplus_skus", set()) and scenario_warehouse_targets.get("moq_surplus_skus", {}).get(sku) == warehouse_id:
                scenario_group = "moq_surplus_skus"

            if demand <= 0:
                # No observed trailing demand at this warehouse; don't invent a cover
                # ratio from a global minimum weekly-demand floor.
                on_hand = 0
            else:
                if scenario_group == "stockout_skus":
                    cover_weeks = float(rng.uniform(0.15, 0.45))
                elif scenario_group == "supplier_delay_skus":
                    cover_weeks = float(rng.uniform(1.0, 1.6))
                elif scenario_group == "overstock_skus":
                    cover_weeks = float(rng.uniform(30.0, 38.0))
                elif scenario_group == "moq_surplus_skus":
                    cover_weeks = float(rng.uniform(0.7, 1.2))
                else:
                    cover_weeks = float(rng.uniform(5.0, 12.0))
                on_hand = int(round(demand * cover_weeks))
                if scenario_group == "moq_surplus_skus":
                    on_hand = max(1, on_hand)
                elif scenario_group == "overstock_skus":
                    on_hand = max(1, on_hand)

            reserved = int(rng.integers(0, int(on_hand * 0.12) + 1)) if on_hand > 0 else 0
            blocked = int(rng.integers(0, int(on_hand * 0.04) + 1)) if on_hand > 0 else 0
            blocked = min(blocked, max(0, on_hand - reserved))
            available = max(0, on_hand - reserved - blocked)
            rows.append({
                "snapshot_date": as_of_date,
                "sku": sku,
                "warehouse_id": warehouse_id,
                "on_hand_quantity": on_hand,
                "reserved_quantity": reserved,
                "blocked_quantity": blocked,
                "available_quantity": available,
            })
    return pd.DataFrame(rows)

def generate_purchase_orders(
    rng: np.random.Generator,
    products: pd.DataFrame,
    warehouses: pd.DataFrame,
    suppliers: pd.DataFrame,
    sourcing_policy: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fx_table: pd.DataFrame,
    scenario_targets: dict[str, list[str]],
    scenario_warehouse_targets: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Generate line-level PO history and an as-of open-PO pipeline.

    Historical received/cancelled PO lines support lead-time analysis. A small, explicit
    set of current open or partially received lines supports incoming-supply risk tests.
    """
    hardware = products.loc[products["physical_inventory_flag"]].copy()
    product_lookup = products.set_index("sku")
    supplier_meta = suppliers.set_index("supplier_id").to_dict("index")
    policy = sourcing_policy.copy()
    policy_groups = {str(sku): group.reset_index(drop=True) for sku, group in policy.groupby("sku")}
    route_templates = sourcing_policy.attrs.get("route_templates", {})
    if not route_templates:
        raise ValueError("Sourcing policy route templates are missing; regenerate policy within main().")

    # Scale historical PO count to inventory-managed hardware, with a floor for small test runs.
    historical_count = max(600, min(8000, len(hardware) * 3))
    max_lead_time = int(policy["conservative_total_lead_time_days"].max()) if len(policy) else 120
    latest_historical_req = end - pd.Timedelta(days=max_lead_time + 20)
    earliest_req = start
    if latest_historical_req < earliest_req:
        latest_historical_req = earliest_req

    columns = [
        "po_line_id", "po_id", "policy_id", "supplier_id", "supplier_country",
        "procurement_channel", "transport_mode", "sourcing_lane", "sku", "warehouse_id",
        "requisition_date", "internal_approval_date", "principal_approval_date", "po_date",
        "expected_dispatch_date", "actual_dispatch_date",
        "original_expected_receipt_date", "expected_receipt_date",
        "original_expected_available_for_sale_date", "expected_available_for_sale_date",
        "actual_receipt_date", "actual_available_for_sale_date", "supplier_eta_update_date",
        "transaction_currency", "exchange_rate_date", "exchange_rate_to_usd",
        "ordered_quantity", "received_quantity", "cancelled_quantity", "remaining_quantity",
        "unit_cost_original", "line_amount_original", "unit_cost_usd", "line_amount_usd",
        "internal_approval_days_actual", "principal_approval_days_actual", "indent_days_actual",
        "transit_days_actual", "receiving_days_actual", "expected_po_to_receipt_days",
        "actual_po_to_receipt_days", "conservative_policy_lead_time_days",
        "planned_total_lead_time_days", "actual_total_lead_time_days",
        "delay_days", "status", "procurement_stage", "cancelled_date",
    ]
    rows: list[dict[str, Any]] = []
    next_id = 1

    def sample_stage(policy_row: dict[str, Any]) -> dict[str, int]:
        def around(value: int, pct: float = 0.18, minimum: int = 0) -> int:
            if value <= 0:
                return 0
            low = max(minimum, int(round(value * (1 - pct))))
            high = max(low, int(round(value * (1 + pct))))
            return int(rng.integers(low, high + 1))
        supplier_id = str(policy_row["supplier_id"])
        t = route_templates[supplier_id]
        return {
            "internal": around(int(policy_row["internal_approval_days"]), 0.25, 1),
            "principal": around(int(policy_row["principal_approval_days"]), 0.30, 0),
            "indent": around(int(policy_row["indent_days"]), 0.25, 0),
            "transit": int(rng.integers(int(t["transit_min_days"]), int(policy_row["transit_days_p90"]) + 1)),
            "receiving": around(int(policy_row["receiving_days"]), 0.25, 1),
        }

    def build_line(
        sku: str,
        policy_row: dict[str, Any],
        requisition_date: pd.Timestamp,
        line_status: str,
        *,
        delay_days: int = 0,
        partial_received_ratio: float | None = None,
        receipt_variance_days: int = 0,
        stage_override: dict[str, int] | None = None,
        warehouse_id_override: str | None = None,
    ) -> dict[str, Any]:
        nonlocal next_id
        product = product_lookup.loc[sku]
        supplier_id = str(policy_row["supplier_id"])
        t = route_templates[supplier_id]
        stages = stage_override if stage_override is not None else sample_stage(policy_row)
        req = requisition_date.normalize()
        internal_date = req + pd.Timedelta(days=stages["internal"])
        principal_date = internal_date + pd.Timedelta(days=stages["principal"])
        po_date = principal_date + pd.Timedelta(days=1)
        expected_dispatch = po_date + pd.Timedelta(days=stages["indent"])
        original_receipt = expected_dispatch + pd.Timedelta(days=stages["transit"])
        original_available = original_receipt + pd.Timedelta(days=stages["receiving"])
        current_receipt = original_receipt + pd.Timedelta(days=delay_days)
        current_available = original_available + pd.Timedelta(days=delay_days)

        # Do not pretend open orders have completed transit/receiving durations.
        actual_dispatch = expected_dispatch if expected_dispatch <= end else pd.NaT
        actual_receipt = pd.NaT
        actual_available = pd.NaT
        actual_indenting = stages["indent"] if pd.notna(actual_dispatch) else pd.NA
        actual_transit = pd.NA
        actual_receiving = pd.NA
        actual_total = pd.NA
        cancelled_date = pd.NaT

        if line_status == "received":
            actual_dispatch_date = max(
                po_date,
                expected_dispatch + pd.Timedelta(days=int(rng.integers(-2, 3))),
            )
            actual_receipt = max(
                actual_dispatch_date + pd.Timedelta(days=1),
                original_receipt + pd.Timedelta(days=receipt_variance_days),
            )
            actual_available = actual_receipt + pd.Timedelta(days=stages["receiving"])
            if actual_available > end:
                # This line is not complete at the dataset cutoff; caller should not use received status.
                raise ValueError("Received PO line has actual availability date after the as-of date.")
            actual_dispatch = actual_dispatch_date
            actual_indenting = int((actual_dispatch - po_date).days)
            actual_transit = int((actual_receipt - actual_dispatch).days)
            actual_receiving = int((actual_available - actual_receipt).days)
            actual_total = int((actual_available - req).days)
            received_qty = 0  # set to ordered quantity after ordered quantity is drawn below
        elif line_status == "partially_received":
            # The first partial receipt is already represented in the current stock snapshot.
            actual_receipt = min(end - pd.Timedelta(days=3), original_receipt)
            actual_available = actual_receipt + pd.Timedelta(days=stages["receiving"])
            if actual_available > end:
                actual_available = end - pd.Timedelta(days=1)
                actual_receipt = actual_available - pd.Timedelta(days=stages["receiving"])
            actual_dispatch = expected_dispatch if expected_dispatch <= actual_receipt else pd.NaT
            actual_indenting = int((actual_dispatch - po_date).days) if pd.notna(actual_dispatch) else pd.NA
            actual_transit = int((actual_receipt - actual_dispatch).days) if pd.notna(actual_dispatch) else pd.NA
            actual_receiving = int((actual_available - actual_receipt).days)
            received_qty = -1  # placeholder until ordered quantity is drawn
        elif line_status == "cancelled":
            cancelled_date = po_date + pd.Timedelta(days=int(rng.integers(1, max(2, stages["indent"] + 1))))
            cancelled_date = min(end, cancelled_date)
            actual_dispatch = pd.NaT
            actual_indenting = pd.NA
            received_qty = 0
            cancelled_qty = -1  # placeholder
        elif line_status == "open":
            received_qty = 0
        else:
            raise ValueError(f"Unsupported PO line status: {line_status}")

        moq = int(policy_row["moq_quantity"])
        multiple = max(1, int(policy_row["order_multiple"]))
        # A controlled partial receipt must contain at least two units so both the
        # received and remaining balances can be positive.
        qty_low = max(moq, 2 if line_status == "partially_received" else 1)
        raw_qty = int(rng.integers(qty_low, qty_low + max(20, int(qty_low * 4)) + 1))
        ordered_qty = int(np.ceil(raw_qty / multiple) * multiple)
        if line_status == "received":
            received_qty = ordered_qty
            cancelled_qty = 0
            remaining_qty = 0
        elif line_status == "partially_received":
            ratio = partial_received_ratio if partial_received_ratio is not None else float(rng.uniform(0.15, 0.70))
            received_qty = min(ordered_qty - 1, max(1, int(round(ordered_qty * ratio))))
            cancelled_qty = 0
            remaining_qty = ordered_qty - received_qty
        elif line_status == "cancelled":
            received_qty = 0
            cancelled_qty = ordered_qty
            remaining_qty = 0
        else:
            received_qty = 0
            cancelled_qty = 0
            remaining_qty = ordered_qty

        # Current ETA for open supply is not earlier than the as-of date. For delayed POs,
        # the original promise is preserved separately so supplier delay can be measured.
        if line_status == "open":
            if current_available <= end:
                current_available = end + pd.Timedelta(days=max(7, delay_days or 14))
                current_receipt = current_available - pd.Timedelta(days=stages["receiving"])
            actual_dispatch = expected_dispatch if expected_dispatch <= end else pd.NaT
        elif line_status == "partially_received":
            # Outstanding quantity is expected after the snapshot date.
            if current_available <= end:
                current_available = end + pd.Timedelta(days=max(14, delay_days or 21))
                current_receipt = current_available - pd.Timedelta(days=stages["receiving"])

        if line_status in {"received", "cancelled"}:
            status_stage = line_status
        elif line_status == "partially_received":
            status_stage = "partially_received"
        elif pd.isna(actual_dispatch):
            status_stage = "awaiting_dispatch"
        elif pd.isna(actual_receipt):
            status_stage = "in_transit"
        else:
            status_stage = "open"

        currency = str(policy_row["currency"])
        supplier_country = supplier_meta[supplier_id]["supplier_country"]
        fx = get_fx_rate(po_date, fx_table) if currency == "IDR" else 1.0
        cost_factor = float(t["cost_factor"])
        cost_usd = float(product["unit_cost"]) * cost_factor * float(rng.uniform(0.97, 1.03))
        cost_original = cost_usd / fx if currency == "IDR" else cost_usd
        original_iso = original_available.date().isoformat()
        current_iso = current_available.date().isoformat()
        original_receipt_iso = original_receipt.date().isoformat()
        current_receipt_iso = current_receipt.date().isoformat()
        eta_update = end if delay_days > 0 else pd.NaT
        line = {
            "po_line_id": f"POL-{next_id:08d}",
            "po_id": f"PO-{next_id:07d}",
            "policy_id": str(policy_row["policy_id"]),
            "supplier_id": supplier_id,
            "supplier_country": supplier_country,
            "procurement_channel": str(policy_row["procurement_channel"]),
            "transport_mode": str(policy_row["transport_mode"]),
            "sourcing_lane": str(policy_row["sourcing_lane"]),
            "sku": sku,
            "warehouse_id": str(
                warehouse_id_override
                if warehouse_id_override is not None
                else rng.choice(warehouses["warehouse_id"].to_numpy())
            ),
            "requisition_date": req,
            "internal_approval_date": internal_date,
            "principal_approval_date": principal_date if stages["principal"] > 0 else pd.NaT,
            "po_date": po_date,
            "expected_dispatch_date": expected_dispatch,
            "actual_dispatch_date": actual_dispatch,
            "original_expected_receipt_date": original_receipt_iso,
            "expected_receipt_date": current_receipt_iso,
            "original_expected_available_for_sale_date": original_iso,
            "expected_available_for_sale_date": current_iso,
            "actual_receipt_date": actual_receipt,
            "actual_available_for_sale_date": actual_available,
            "supplier_eta_update_date": eta_update,
            "transaction_currency": currency,
            "exchange_rate_date": po_date.to_period("M").start_time.date().isoformat(),
            "exchange_rate_to_usd": fx,
            "ordered_quantity": ordered_qty,
            "received_quantity": received_qty,
            "cancelled_quantity": cancelled_qty,
            "remaining_quantity": remaining_qty,
            "unit_cost_original": round(cost_original, 2),
            "line_amount_original": round(cost_original * ordered_qty, 2),
            "unit_cost_usd": round(cost_usd, 2),
            "line_amount_usd": round(cost_usd * ordered_qty, 2),
            "internal_approval_days_actual": stages["internal"],
            "principal_approval_days_actual": stages["principal"] if stages["principal"] > 0 else pd.NA,
            "indent_days_actual": actual_indenting,
            "transit_days_actual": actual_transit,
            "receiving_days_actual": actual_receiving,
            "expected_po_to_receipt_days": int((current_receipt - po_date).days) if line_status in {"open", "partially_received"} else int((original_receipt - po_date).days),
            "actual_po_to_receipt_days": int((actual_receipt - po_date).days) if line_status == "received" else pd.NA,
            "conservative_policy_lead_time_days": int(policy_row["conservative_total_lead_time_days"]),
            "planned_total_lead_time_days": int(policy_row["planned_total_lead_time_days"]),
            "actual_total_lead_time_days": actual_total,
            "delay_days": max(0, int((current_available - original_available).days)),
            "status": line_status,
            "procurement_stage": status_stage,
            "cancelled_date": cancelled_date,
        }
        next_id += 1
        return line

    # Historical procurement data: completed/cancelled orders, with enough cutoff buffer
    # so records marked received really were available before the as-of date.
    if latest_historical_req < earliest_req:
        historical_dates = [start]
    else:
        span = max(1, (latest_historical_req - earliest_req).days)
        historical_dates = None
    hardware_skus = hardware["sku"].to_numpy()
    for _ in range(historical_count):
        sku = str(rng.choice(hardware_skus))
        group = policy_groups[sku]
        if len(group) > 1 and rng.random() < 0.22:
            row_policy = group.iloc[1].to_dict()
        else:
            row_policy = group.iloc[0].to_dict()
        if historical_dates is None:
            req = start + pd.Timedelta(days=int(rng.integers(0, span + 1)))
        else:
            req = start
        stages = sample_stage(row_policy)
        estimated_available = req + pd.Timedelta(days=(stages["internal"] + stages["principal"] + 1 + stages["indent"] + stages["transit"] + stages["receiving"]))
        if estimated_available > end - pd.Timedelta(days=7):
            # Move the requisition earlier so a historical completed record can be closed honestly.
            req -= estimated_available - (end - pd.Timedelta(days=7))
            req = max(req, start)
        # Use a stable seed-independent status mix for normal historical records.
        chosen_status = "cancelled" if rng.random() < 0.09 else "received"
        line = build_line(
            str(sku), row_policy, req, chosen_status,
            receipt_variance_days=int(rng.integers(-3, 6)), stage_override=stages,
        )
        rows.append(line)

    # Explicit current incoming supply for ordinary cases (not stockout or overstock targets).
    normal_inbound_skus = list(scenario_targets.get("normal_inbound_skus", []))
    for sku in normal_inbound_skus:
        group = policy_groups[str(sku)]
        row_policy = group.iloc[0].to_dict()
        stages = sample_stage(row_policy)
        total_days = stages["internal"] + stages["principal"] + 1 + stages["indent"] + stages["transit"] + stages["receiving"]
        post_approval_days = stages["indent"] + stages["transit"] + stages["receiving"]
        # ETA is 7-30 days after the snapshot, but the requisition and approvals remain historical.
        eta_offset = min(int(rng.integers(7, 31)), max(1, post_approval_days - 1))
        req = end + pd.Timedelta(days=eta_offset - total_days)
        line = build_line(
            str(sku), row_policy, req, "open", stage_override=stages,
            warehouse_id_override=scenario_warehouse_targets.get("normal_inbound_skus", {}).get(str(sku)),
        )
        rows.append(line)

    # Controlled supplier-delay cases: original promise has passed, supplier has revised
    # ETA forward, and stock is deliberately low enough to risk depletion before arrival.
    delay_skus = list(scenario_targets.get("supplier_delay_skus", []))
    for sku in delay_skus:
        group = policy_groups[str(sku)]
        route_row = group.loc[group["supplier_id"].eq("SUP-0001")]
        if route_row.empty:
            route_row = group.iloc[[0]]
        row_policy = route_row.iloc[0].to_dict()
        stages = sample_stage(row_policy)
        normal_total = stages["internal"] + stages["principal"] + 1 + stages["indent"] + stages["transit"] + stages["receiving"]
        req = end - pd.Timedelta(days=normal_total + 7)
        line = build_line(
            str(sku), row_policy, req, "open", delay_days=28, stage_override=stages,
            warehouse_id_override=scenario_warehouse_targets.get("supplier_delay_skus", {}).get(str(sku)),
        )
        rows.append(line)

    # Controlled partial receipts: received quantity is already represented in on-hand stock;
    # only the remaining quantity should count as incoming supply.
    partial_targets = list(scenario_targets.get("partial_po_skus", []))
    for sku in partial_targets:
        group = policy_groups[str(sku)]
        row_policy = group.iloc[0].to_dict()
        stages = sample_stage(row_policy)
        normal_total = stages["internal"] + stages["principal"] + 1 + stages["indent"] + stages["transit"] + stages["receiving"]
        req = end - pd.Timedelta(days=normal_total + 7)
        line = build_line(
            str(sku), row_policy, req, "partially_received", delay_days=21,
            partial_received_ratio=float(rng.uniform(0.25, 0.65)), stage_override=stages,
            warehouse_id_override=scenario_warehouse_targets.get("partial_po_skus", {}).get(str(sku)),
        )
        rows.append(line)

    po = pd.DataFrame(rows, columns=columns)
    po["status"] = po["status"].replace({"partially_received": "partial"})
    return po

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
    """Inject a small number of data-quality anomalies, separate from business scenarios."""
    log: list[dict[str, Any]] = []

    def record(dataset: str, record_id: str, field: str, anomaly_type: str, description: str) -> None:
        log.append({
            "anomaly_id": f"ANOM-{len(log) + 1:05d}", "dataset": dataset,
            "record_id": str(record_id), "field": field, "anomaly_type": anomaly_type,
            "description": description,
        })

    sales = datasets["sales_transactions"].copy()
    if len(sales) > 10:
        source = sales.iloc[5].copy()
        duplicate = source.copy()
        duplicate["invoice_line_id"] = f"SIL-DUP-{source['invoice_line_id']}"
        sales = pd.concat([sales, pd.DataFrame([duplicate])], ignore_index=True)
        record("sales_transactions", duplicate["invoice_line_id"], "invoice_id,sku,warehouse_id,transaction_date,quantity",
               "duplicate_candidate", "Copied transaction business fields with a new line ID for data-quality testing.")
    if len(sales) > 20:
        sales.loc[20, "customer_id"] = pd.NA
        record("sales_transactions", sales.loc[20, "invoice_line_id"], "customer_id",
               "missing_value", "Customer reference intentionally missing for data-quality testing.")
    if len(sales) > 30:
        sales.loc[30, "quantity"] = -abs(int(sales.loc[30, "quantity"]))
        sales.loc[30, "line_amount"] = -abs(float(sales.loc[30, "line_amount"]))
        sales.loc[30, "line_amount_usd"] = -abs(float(sales.loc[30, "line_amount_usd"]))
        record("sales_transactions", sales.loc[30, "invoice_line_id"], "quantity",
               "invalid_value", "Negative quantity and corresponding amounts intentionally injected for data-quality testing.")
    datasets["sales_transactions"] = sales

    quotations = datasets["sales_quotations"].copy()
    if len(quotations) > 5:
        quotations.loc[4, "approval_at"] = pd.NaT
        record("sales_quotations", quotations.loc[4, "quotation_id"], "approval_at",
               "missing_value", "Approval timestamp intentionally missing for data-quality testing.")
    datasets["sales_quotations"] = quotations

    subscriptions = datasets["subscription_records"].copy()
    if len(subscriptions) > 5:
        subscriptions.loc[2, "renewal_date"] = pd.NaT
        record("subscription_records", subscriptions.loc[2, "subscription_id"], "renewal_date",
               "missing_value", "Renewal date intentionally missing for data-quality testing.")
    datasets["subscription_records"] = subscriptions

    return pd.DataFrame(log, columns=[
        "anomaly_id", "dataset", "record_id", "field", "anomaly_type", "description",
    ])

def validate_business_scenario_design(
    scenario_targets: dict[str, list[str]],
    datasets: dict[str, pd.DataFrame],
    weekly_demand_by_sku_warehouse: dict[tuple[str, str], float],
    scenario_warehouse_targets: dict[str, dict[str, str]],
    as_of_date: pd.Timestamp,
) -> list[str]:
    """Validate controlled scenarios at the same SKU/warehouse grain as operations."""
    errors: list[str] = []
    inv = datasets["inventory_snapshots"]
    po = datasets["purchase_orders"]
    policy = datasets["sku_sourcing_policy"]
    available_map = {
        (str(row.sku), str(row.warehouse_id)): int(row.available_quantity)
        for row in inv.itertuples(index=False)
    }
    active = po.loc[
        po["status"].isin(["open", "partial"]) & po["remaining_quantity"].gt(0)
    ].copy()

    def scenario_warehouse(group_name: str, sku: str) -> str | None:
        return scenario_warehouse_targets.get(group_name, {}).get(str(sku))

    def demand_for(sku: str, warehouse_id: str) -> float:
        return float(weekly_demand_by_sku_warehouse.get((str(sku), str(warehouse_id)), 0.0))

    for sku_value in scenario_targets.get("stockout_skus", []):
        sku = str(sku_value)
        wh = scenario_warehouse("stockout_skus", sku)
        if wh is None:
            errors.append(f"Stockout scenario {sku}: target warehouse is missing.")
            continue
        demand = demand_for(sku, wh)
        if demand <= 0:
            errors.append(f"Stockout scenario {sku}/{wh}: no positive trailing demand at the target warehouse.")
            continue
        cover = available_map.get((sku, wh), 0) / demand
        if cover >= 0.5:
            errors.append(f"Stockout scenario {sku}/{wh}: sellable cover {cover:.2f} weeks is not low enough (<0.5 required).")
        if active["sku"].eq(sku).any():
            errors.append(f"Stockout scenario {sku}: active inbound exists; expected no controlled incoming PO for this SKU.")

    for sku_value in scenario_targets.get("supplier_delay_skus", []):
        sku = str(sku_value)
        wh = scenario_warehouse("supplier_delay_skus", sku)
        if wh is None:
            errors.append(f"Supplier-delay scenario {sku}: target warehouse is missing.")
            continue
        matches = active.loc[
            active["sku"].eq(sku) & active["warehouse_id"].astype(str).eq(wh)
            & active["delay_days"].fillna(0).ge(20)
        ]
        if matches.empty:
            errors.append(f"Supplier-delay scenario {sku}/{wh}: no active PO at target warehouse with at least 20 revised delay days.")
            continue
        row = matches.iloc[-1]
        original = pd.to_datetime(row["original_expected_available_for_sale_date"])
        current = pd.to_datetime(row["expected_available_for_sale_date"])
        if not (original < as_of_date < current):
            errors.append(f"Supplier-delay scenario {sku}/{wh}: original ETA, as-of date, and revised ETA are not in the expected order.")
        demand = demand_for(sku, wh)
        if demand <= 0:
            errors.append(f"Supplier-delay scenario {sku}/{wh}: no positive trailing demand at the target warehouse.")
            continue
        cover = available_map.get((sku, wh), 0) / demand
        time_to_eta = max(0.0, (current - as_of_date).days / 7.0)
        if cover >= time_to_eta:
            errors.append(f"Supplier-delay scenario {sku}/{wh}: sellable cover ({cover:.2f} weeks) does not fall before revised ETA ({time_to_eta:.2f} weeks).")

    for sku_value in scenario_targets.get("overstock_skus", []):
        sku = str(sku_value)
        wh = scenario_warehouse("overstock_skus", sku)
        if wh is None:
            errors.append(f"Overstock scenario {sku}: target warehouse is missing.")
            continue
        demand = demand_for(sku, wh)
        if demand <= 0:
            errors.append(f"Overstock scenario {sku}/{wh}: no positive trailing demand at the target warehouse.")
            continue
        cover = available_map.get((sku, wh), 0) / demand
        if cover < 25:
            errors.append(f"Overstock scenario {sku}/{wh}: sellable cover {cover:.2f} weeks is below the 25-week test threshold.")

    for sku_value in scenario_targets.get("moq_surplus_skus", []):
        sku = str(sku_value)
        wh = scenario_warehouse("moq_surplus_skus", sku)
        if wh is None:
            errors.append(f"MOQ scenario {sku}: target warehouse is missing.")
            continue
        hit = policy.loc[policy["sku"].eq(sku) & policy["is_preferred_route"]]
        if hit.empty:
            errors.append(f"MOQ scenario {sku}: no preferred sourcing route.")
            continue
        row = hit.iloc[0]
        if row["supplier_id"] not in {"SUP-0003", "SUP-0004"} or int(row["moq_quantity"]) < 10:
            errors.append(f"MOQ scenario {sku}: expected a locally available route with a deliberately binding MOQ >= 10.")
        demand = demand_for(sku, wh)
        stock = available_map.get((sku, wh), 0)
        if demand <= 0:
            errors.append(f"MOQ scenario {sku}/{wh}: no positive trailing demand at the target warehouse.")
            continue
        if stock < 1:
            errors.append(f"MOQ scenario {sku}/{wh}: at least one sellable unit must remain at the target warehouse.")
        # Test a simple eight-week order-up-to policy. The scenario is meaningful only
        # when MOQ/order multiple drives the order above modeled net need.
        target_stock = 8.0 * demand
        net_need = max(0, int(np.ceil(target_stock - stock)))
        moq = int(row["moq_quantity"])
        multiple = max(1, int(row["order_multiple"]))
        unconstrained_need = max(0, net_need)
        desired_qty = max(moq, unconstrained_need)
        suggested_qty = int(np.ceil(desired_qty / multiple) * multiple) if desired_qty > 0 else 0
        post_order_cover = (stock + suggested_qty) / demand
        if suggested_qty <= unconstrained_need:
            errors.append(f"MOQ scenario {sku}/{wh}: MOQ is not binding relative to the eight-week target (need={unconstrained_need}, suggestion={suggested_qty}).")
        if post_order_cover <= 10.0:
            errors.append(f"MOQ scenario {sku}/{wh}: MOQ does not create visible surplus; post-order cover is {post_order_cover:.2f} weeks.")

    for sku_value in scenario_targets.get("partial_po_skus", []):
        sku = str(sku_value)
        wh = scenario_warehouse("partial_po_skus", sku)
        if wh is None:
            errors.append(f"Partial-PO scenario {sku}: target warehouse is missing.")
            continue
        matches = active.loc[
            active["sku"].eq(sku) & active["warehouse_id"].astype(str).eq(wh)
            & active["status"].eq("partial")
        ]
        if matches.empty:
            errors.append(f"Partial-PO scenario {sku}/{wh}: no partially received open line with remaining quantity at target warehouse.")
            continue
        row = matches.iloc[-1]
        eta = pd.to_datetime(row["expected_available_for_sale_date"])
        if int(row["received_quantity"]) <= 0 or int(row["remaining_quantity"]) <= 0 or eta <= as_of_date:
            errors.append(
                f"Partial-PO scenario {sku}/{wh}: receipt/remaining/ETA inconsistent "
                f"(received={row['received_quantity']}, remaining={row['remaining_quantity']}, "
                f"ETA={eta}, as_of={as_of_date})."
            )

    for sku_value in scenario_targets.get("normal_inbound_skus", []):
        sku = str(sku_value)
        wh = scenario_warehouse("normal_inbound_skus", sku)
        if wh is None:
            errors.append(f"Normal-inbound scenario {sku}: target warehouse is missing.")
            continue
        matches = active.loc[
            active["sku"].eq(sku) & active["warehouse_id"].astype(str).eq(wh)
            & active["status"].eq("open") & active["delay_days"].eq(0)
        ]
        if matches.empty:
            errors.append(f"Normal-inbound scenario {sku}/{wh}: no undelayed open PO at target warehouse.")
            continue
        if pd.to_datetime(matches.iloc[-1]["expected_available_for_sale_date"]) <= as_of_date:
            errors.append(f"Normal-inbound scenario {sku}/{wh}: ETA must be after the inventory snapshot date.")

    return errors


def build_business_scenario_log(
    scenario_targets: dict[str, list[str]],
    purchase_orders: pd.DataFrame,
    scenario_warehouse_targets: dict[str, dict[str, str]],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Create scenario expectations with explicit target warehouse and PO linkage."""
    descriptions = {
        "stockout_skus": (
            "low_stock_no_confirmed_inbound",
            "stockout_risk_alert",
            "Sellable stock at the named warehouse is low relative to its recent weekly demand; no active inbound PO is assigned to the SKU.",
        ),
        "supplier_delay_skus": (
            "supplier_eta_revised",
            "supplier_delay_and_stockout_risk_alert",
            "An open PO at the named warehouse has a later current ETA than its original promise; local sellable stock may deplete before arrival.",
        ),
        "overstock_skus": (
            "excess_stock_cover",
            "overstock_review_alert",
            "Sellable stock at the named warehouse is deliberately high relative to that warehouse's average weekly demand.",
        ),
        "moq_surplus_skus": (
            "minimum_order_constraint",
            "moq_surplus_warning",
            "The named warehouse has low demand and limited sellable stock; the preferred route's MOQ/order multiple is expected to exceed net replenishment need.",
        ),
        "normal_inbound_skus": (
            "normal_open_purchase_order",
            "count_incoming_supply_once",
            "An open PO for the named warehouse with a future ETA should count as incoming supply but not as current sellable stock.",
        ),
        "partial_po_skus": (
            "partial_receipt_outstanding_balance",
            "count_only_remaining_quantity",
            "The PO at the named warehouse records a partial receipt, so only the unreceived and uncancelled balance counts as incoming supply. The inventory snapshot is an aggregate as-of position, not a receipt-level stock-movement ledger in this MVP.",
        ),
    }
    rows: list[dict[str, Any]] = []
    counter = 1
    for group_name, skus in scenario_targets.items():
        if group_name not in descriptions:
            continue
        scenario_type, expected_detection, description = descriptions[group_name]
        for sku_value in skus:
            sku = str(sku_value)
            target_wh = scenario_warehouse_targets.get(group_name, {}).get(sku, "")
            related = purchase_orders.iloc[0:0].copy()
            if group_name == "supplier_delay_skus":
                related = purchase_orders.loc[
                    purchase_orders["sku"].eq(sku)
                    & purchase_orders["warehouse_id"].astype(str).eq(target_wh)
                    & purchase_orders["delay_days"].fillna(0).gt(0)
                    & purchase_orders["status"].isin(["open", "partial"])
                    & purchase_orders["remaining_quantity"].gt(0)
                ]
            elif group_name == "normal_inbound_skus":
                related = purchase_orders.loc[
                    purchase_orders["sku"].eq(sku)
                    & purchase_orders["warehouse_id"].astype(str).eq(target_wh)
                    & purchase_orders["status"].eq("open")
                    & purchase_orders["delay_days"].eq(0)
                    & purchase_orders["remaining_quantity"].gt(0)
                ]
            elif group_name == "partial_po_skus":
                related = purchase_orders.loc[
                    purchase_orders["sku"].eq(sku)
                    & purchase_orders["warehouse_id"].astype(str).eq(target_wh)
                    & purchase_orders["status"].eq("partial")
                    & purchase_orders["remaining_quantity"].gt(0)
                ]
            related_po_id = str(related.iloc[-1]["po_line_id"]) if not related.empty else ""
            if not related.empty:
                target_wh = str(related.iloc[-1]["warehouse_id"])
            rows.append({
                "scenario_id": f"SCN-{counter:04d}",
                "scenario_type": scenario_type,
                "sku": sku,
                "warehouse_id": target_wh,
                "related_po_line_id": related_po_id,
                "as_of_date": as_of_date.date().isoformat(),
                "expected_detection": expected_detection,
                "description": description,
            })
            counter += 1
    return pd.DataFrame(rows, columns=[
        "scenario_id", "scenario_type", "sku", "warehouse_id", "related_po_line_id",
        "as_of_date", "expected_detection", "description",
    ])

def validate_base_relations(datasets: dict[str, pd.DataFrame]) -> list[str]:
    """Validate relationships and key stock/PO invariants while allowing logged sales anomalies."""
    errors: list[str] = []
    products_df = datasets["products"]
    products = set(products_df["sku"])
    hardware_products = set(products_df.loc[products_df["physical_inventory_flag"], "sku"])
    warehouses_df = datasets["warehouses"]
    warehouses = set(warehouses_df["warehouse_id"])
    calendar_df = datasets["calendar"].copy()
    if calendar_df["calendar_date"].isna().any() or calendar_df["calendar_date"].duplicated().any():
        errors.append("calendar: calendar_date must be valid and unique")
    if not (calendar_df["is_weekend"] == ~calendar_df["is_weekday"]).all():
        errors.append("calendar: weekend and weekday flags are inconsistent")
    if (calendar_df["is_business_day"] & ~calendar_df["is_weekday"]).any():
        errors.append("calendar: a weekend is marked as a business day")
    if (calendar_df["is_business_day"] & calendar_df["is_public_holiday"]).any():
        errors.append("calendar: a public holiday is marked as a business day")
    if calendar_df["collective_leave_policy"].nunique() != 1:
        errors.append("calendar: collective leave policy must be consistent")
    elif str(calendar_df["collective_leave_policy"].iloc[0]) == "observe":
        if (calendar_df["is_business_day"] & calendar_df["is_collective_leave"]).any():
            errors.append("calendar: observed collective leave is incorrectly marked as a business day")
    sales_dates = pd.to_datetime(datasets["sales_transactions"]["transaction_date"], errors="coerce").dt.normalize()
    calendar_business_dates = set(calendar_df.loc[calendar_df["is_business_day"], "calendar_date"].dt.normalize())
    if sales_dates.isna().any() or not sales_dates.isin(calendar_business_dates).all():
        errors.append("sales_transactions: transaction dates must fall on valid business days in calendar.csv")
    customers = set(datasets["customers"]["customer_id"])
    suppliers = set(datasets["suppliers"]["supplier_id"])
    quotations = set(datasets["sales_quotations"]["quotation_id"])

    checks = [
        ("sales_transactions.sku", datasets["sales_transactions"]["sku"].dropna(), products),
        ("sales_transactions.warehouse_id", datasets["sales_transactions"]["warehouse_id"].dropna(), warehouses),
        ("sales_transactions.customer_id", datasets["sales_transactions"]["customer_id"].dropna(), customers),
        ("inventory_snapshots.sku", datasets["inventory_snapshots"]["sku"].dropna(), hardware_products),
        ("inventory_snapshots.warehouse_id", datasets["inventory_snapshots"]["warehouse_id"].dropna(), warehouses),
        ("purchase_orders.sku", datasets["purchase_orders"]["sku"].dropna(), hardware_products),
        ("purchase_orders.warehouse_id", datasets["purchase_orders"]["warehouse_id"].dropna(), warehouses),
        ("purchase_orders.supplier_id", datasets["purchase_orders"]["supplier_id"].dropna(), suppliers),
        ("sku_sourcing_policy.sku", datasets["sku_sourcing_policy"]["sku"].dropna(), hardware_products),
        ("sku_sourcing_policy.supplier_id", datasets["sku_sourcing_policy"]["supplier_id"].dropna(), suppliers),
        ("sales_quotations.customer_id", datasets["sales_quotations"]["customer_id"].dropna(), customers),
        ("quotation_lines.sku", datasets["quotation_lines"]["sku"].dropna(), products),
        ("quotation_lines.quotation_id", datasets["quotation_lines"]["quotation_id"].dropna(), quotations),
        ("subscription_records.sku", datasets["subscription_records"]["sku"].dropna(), products),
        ("subscription_records.customer_id", datasets["subscription_records"]["customer_id"].dropna(), customers),
        ("product_replacements.old_sku", datasets["product_replacements"]["old_sku"].dropna(), hardware_products),
        ("product_replacements.new_sku", datasets["product_replacements"]["new_sku"].dropna(), hardware_products),
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
        "sku_sourcing_policy.policy_id": (datasets["sku_sourcing_policy"], "policy_id"),
        "sales_quotations.quotation_id": (datasets["sales_quotations"], "quotation_id"),
        "quotation_lines.quotation_line_id": (datasets["quotation_lines"], "quotation_line_id"),
        "subscription_records.subscription_id": (datasets["subscription_records"], "subscription_id"),
        "product_replacements.replacement_id": (datasets["product_replacements"], "replacement_id"),
    }
    for name, (frame, column) in unique_keys.items():
        if frame[column].duplicated().any():
            errors.append(f"{name}: duplicate primary key(s) found")

    policy = datasets["sku_sourcing_policy"]
    if policy.duplicated(["sku", "supplier_id"]).any():
        errors.append("sku_sourcing_policy: duplicate SKU/supplier route rows")
    preferred_counts = policy.loc[policy["is_preferred_route"]].groupby("sku").size()
    if not preferred_counts.eq(1).all() or set(preferred_counts.index) != hardware_products:
        errors.append("sku_sourcing_policy: each hardware SKU must have exactly one preferred route")
    base_uom_map = products_df.set_index("sku")["base_uom"].to_dict()
    if not policy.apply(lambda row: base_uom_map.get(row["sku"]) == row["moq_unit"], axis=1).all():
        errors.append("sku_sourcing_policy: MOQ unit must match the SKU base unit")
    valid_policy_pairs = set(zip(policy["policy_id"], policy["sku"], policy["supplier_id"]))
    for row in datasets["purchase_orders"].itertuples(index=False):
        if (row.policy_id, row.sku, row.supplier_id) not in valid_policy_pairs:
            errors.append(f"purchase_orders policy/SKU/supplier mismatch at {row.po_line_id}")
            break

    inv = datasets["inventory_snapshots"].copy()
    if inv.duplicated(["snapshot_date", "sku", "warehouse_id"]).any():
        errors.append("inventory_snapshots: duplicate SKU/warehouse/as-of keys")
    if inv["snapshot_date"].nunique() != 1:
        errors.append("inventory_snapshots: this MVP expects one consistent as-of snapshot date")
    if set(inv["sku"]) - hardware_products:
        errors.append("inventory_snapshots: non-hardware SKU included in physical inventory")
    expected_inventory_pairs = {(sku, warehouse_id) for sku in hardware_products for warehouse_id in set(warehouses_df["warehouse_id"])}
    actual_inventory_pairs = set(zip(inv["sku"], inv["warehouse_id"]))
    if actual_inventory_pairs != expected_inventory_pairs:
        errors.append("inventory_snapshots: each hardware SKU must have one row for every warehouse at the as-of date")
    if (inv[["on_hand_quantity", "reserved_quantity", "blocked_quantity", "available_quantity"]] < 0).any().any():
        errors.append("inventory_snapshots: negative inventory quantity found")
    if (inv["reserved_quantity"] + inv["blocked_quantity"] > inv["on_hand_quantity"]).any():
        errors.append("inventory_snapshots: reserved + blocked exceeds on-hand")
    if not (inv["available_quantity"] == inv["on_hand_quantity"] - inv["reserved_quantity"] - inv["blocked_quantity"]).all():
        errors.append("inventory_snapshots: available quantity is inconsistent with on-hand/reserved/blocked")

    po = datasets["purchase_orders"].copy()
    quantity_balance = po["received_quantity"] + po["cancelled_quantity"] + po["remaining_quantity"]
    if not quantity_balance.eq(po["ordered_quantity"]).all():
        errors.append("purchase_orders: ordered quantity does not reconcile to received + cancelled + remaining")
    if (po[["ordered_quantity", "received_quantity", "cancelled_quantity", "remaining_quantity"]] < 0).any().any():
        errors.append("purchase_orders: negative quantity found")
    if po.loc[po["status"].eq("received"), "remaining_quantity"].ne(0).any():
        errors.append("purchase_orders: received lines must have zero remaining quantity")
    received_rows = po["status"].eq("received")
    if (po.loc[received_rows, "received_quantity"] != po.loc[received_rows, "ordered_quantity"]).any():
        errors.append("purchase_orders: received lines must have received quantity equal to ordered quantity")
    if po.loc[received_rows, "actual_available_for_sale_date"].isna().any():
        errors.append("purchase_orders: received lines must have an actual available-for-sale date")
    if po.loc[po["status"].eq("cancelled"), "remaining_quantity"].ne(0).any():
        errors.append("purchase_orders: cancelled lines must have zero remaining quantity")
    cancelled_rows = po["status"].eq("cancelled")
    if (po.loc[cancelled_rows, "cancelled_quantity"] != po.loc[cancelled_rows, "ordered_quantity"]).any():
        errors.append("purchase_orders: fully cancelled lines must have cancelled quantity equal to ordered quantity")
    if po.loc[po["status"].isin(["open", "partial"]), "remaining_quantity"].le(0).any():
        errors.append("purchase_orders: open/partial lines must have positive remaining quantity")
    allowed_statuses = {"received", "cancelled", "open", "partial"}
    if not set(po["status"].dropna().unique()).issubset(allowed_statuses):
        errors.append("purchase_orders: unsupported PO status value found")
    active = po.loc[po["status"].isin(["open", "partial"]) & po["remaining_quantity"].gt(0)]
    if active["expected_available_for_sale_date"].isna().any():
        errors.append("purchase_orders: open/partial lines must have current expected availability date")
    as_of_date = pd.to_datetime(inv["snapshot_date"]).max()
    active_eta = pd.to_datetime(active["expected_available_for_sale_date"], errors="coerce")
    if active_eta.le(as_of_date).any():
        errors.append("purchase_orders: active incoming supply ETA must be later than the inventory snapshot date")
    if active["actual_available_for_sale_date"].notna().any() & active["status"].eq("open").any():
        # Open lines may not be fully received; partial lines can have a first actual receipt.
        open_rows = active["status"].eq("open")
        if active.loc[open_rows, "actual_available_for_sale_date"].notna().any():
            errors.append("purchase_orders: open lines must not have an actual available-for-sale date")
    if (po["po_date"] < po["requisition_date"]).any():
        errors.append("purchase_orders: PO date cannot precede requisition date")
    received_dates = po.loc[received_rows]
    if (pd.to_datetime(received_dates["actual_receipt_date"]) > pd.to_datetime(received_dates["actual_available_for_sale_date"])).any():
        errors.append("purchase_orders: actual receipt must not occur after available-for-sale date")

    for column in ["country", "ship_to_country"]:
        if not datasets["customers"][column].eq(COMPANY_COUNTRY).all():
            errors.append(f"customers.{column}: all values must be Indonesia")
    if not warehouses_df["country"].eq(COMPANY_COUNTRY).all():
        errors.append("warehouses.country: all warehouses must be in Indonesia")
    if not warehouses_df["city"].eq(WAREHOUSE_CITY).all():
        errors.append(f"warehouses.city: all warehouses must be {WAREHOUSE_CITY} for version 1")
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
    """Generate core data, apply controlled cases, validate, then write outputs."""
    args = parse_args()
    start, end = as_date(args.start_date), as_date(args.end_date)
    if end < start:
        raise ValueError("end-date must be on or after start-date")
    if args.sales_target_rows < 1:
        raise ValueError("sales-target-rows must be >= 1")

    rng = np.random.default_rng(args.seed)
    calendar, closure_dates = make_calendar(start, end, args.collective_leave_policy)
    products, warehouses, customers, suppliers, replacements = make_master_data(
        rng, args.products, args.warehouses, args.customers
    )
    fx_table = build_fx_table(start, end)
    datasets: dict[str, pd.DataFrame] = {
        "products": products,
        "warehouses": warehouses,
        "customers": customers,
        "suppliers": suppliers,
        "product_replacements": replacements,
        "calendar": calendar,
        "fx_rates": fx_table,
    }
    datasets["sales_transactions"] = generate_sales(
        rng, products, warehouses, customers, start, end,
        args.sales_target_rows, fx_table, closure_dates,
    )
    demand_pattern_quality = summarize_hardware_demand_patterns(
        datasets["sales_transactions"], products, start, end, closure_dates
    )
    policy = generate_sku_sourcing_policy(rng, products, suppliers, start, end)
    datasets["sku_sourcing_policy"] = policy

    # Choose scenario SKUs using demand profile and supplier-route eligibility.
    hardware = products.loc[products["physical_inventory_flag"]].copy()
    demand_by_sku_warehouse = completed_weekly_demand_by_sku_warehouse(
        datasets["sales_transactions"], end, weeks=52
    )
    demand_52w: dict[str, float] = {}
    for (sku_key, _warehouse_key), weekly_qty in demand_by_sku_warehouse.items():
        demand_52w[sku_key] = demand_52w.get(sku_key, 0.0) + float(weekly_qty)
    eligible_delay = list(policy.attrs.get("supplier_delay_candidate_skus", []))
    eligible_delay = sorted(eligible_delay, key=lambda s: demand_52w.get(s, 0.0), reverse=True)
    supplier_delay_skus = eligible_delay[:min(5, len(eligible_delay))]

    # Pick low-demand intermittent/lumpy SKUs with a locally available preferred route.
    # This ensures a high MOQ is a genuine constraint relative to a short route's need,
    # rather than applying an arbitrary high minimum to all products or all overseas lanes.
    local_preferred = set(policy.loc[
        policy["is_preferred_route"] & policy["supplier_id"].isin(["SUP-0003", "SUP-0004"]), "sku"
    ])
    product_type_by_sku = products.set_index("sku")["demand_type_seed"].to_dict()
    low_candidates = [
        str(sku) for sku in hardware["sku"]
        if str(sku) in local_preferred
        and product_type_by_sku.get(str(sku)) in {"intermittent", "lumpy"}
        and 0.4 <= demand_52w.get(str(sku), 0.0) <= 2.5
        and str(sku) not in supplier_delay_skus
    ]
    low_candidates = sorted(low_candidates, key=lambda sku: demand_52w.get(sku, 0.0))
    low_demand_moq = low_candidates[:min(5, len(low_candidates))]
    for sku in low_demand_moq:
        preferred_idx = policy.index[policy["sku"].eq(sku) & policy["is_preferred_route"]]
        if len(preferred_idx):
            idx = preferred_idx[0]
            product = hardware.loc[hardware["sku"].eq(sku)].iloc[0]
            family = str(product.get("product_family_name", ""))
            # Only the selected test SKUs get a deliberately binding MOQ.
            base_moq = 20 if family in {"Fiber Cable", "Copper Cable"} else 10
            wh_demands = [
                float(demand_by_sku_warehouse.get((str(sku), str(wh)), 0.0))
                for wh in warehouses["warehouse_id"].astype(str).tolist()
                if float(demand_by_sku_warehouse.get((str(sku), str(wh)), 0.0)) > 0
            ]
            # Configure a controlled MOQ challenge at the selected (highest-demand)
            # warehouse: minimum order covers ~11 weeks there, so an eight-week
            # order-up-to policy detects MOQ-driven excess without relying on a nearly
            # zero-demand warehouse. This remains a synthetic test assumption.
            target_warehouse_demand = max(wh_demands) if wh_demands else 0.0
            demand_based_moq = int(np.ceil((11.0 * target_warehouse_demand) / 5.0) * 5) if target_warehouse_demand > 0 else base_moq
            moq = max(base_moq, demand_based_moq)
            policy.loc[idx, "moq_quantity"] = moq
            policy.loc[idx, "order_multiple"] = 5
    policy.attrs["moq_surplus_skus"] = low_demand_moq

    excluded = set(supplier_delay_skus) | set(low_demand_moq)
    demand_ranked = sorted(
        [str(s) for s in hardware["sku"] if str(s) not in excluded],
        key=lambda s: demand_52w.get(s, 0.0), reverse=True,
    )
    stockout_skus = demand_ranked[:min(5, len(demand_ranked))]
    excluded |= set(stockout_skus)
    # Choose moderate-to-high demand SKUs for overstock, avoiding other test groups.
    overstock_candidates = [s for s in demand_ranked if s not in excluded]
    overstock_candidates = sorted(overstock_candidates, key=lambda s: demand_52w.get(s, 0.0), reverse=True)
    overstock_skus = overstock_candidates[:min(10, len(overstock_candidates))] if overstock_candidates else []
    excluded |= set(overstock_skus)
    # Separate controlled partial receipts and ordinary open PO examples.
    remaining_candidates = [str(s) for s in hardware["sku"] if str(s) not in excluded]
    partial_po_skus = remaining_candidates[:min(5, len(remaining_candidates))]
    excluded |= set(partial_po_skus)
    normal_candidates = [str(s) for s in hardware["sku"] if str(s) not in excluded]
    # Model a broader, but still manageable, as-of incoming pipeline across SKUs.
    # Random selection avoids making current inbound supply depend on SKU ID order.
    normal_inbound_count = min(60, len(normal_candidates))
    normal_inbound_skus = sorted(
        rng.choice(normal_candidates, size=normal_inbound_count, replace=False).tolist()
    ) if normal_inbound_count else []

    scenario_targets = {
        "supplier_delay_skus": supplier_delay_skus,
        "stockout_skus": stockout_skus,
        "overstock_skus": overstock_skus,
        "moq_surplus_skus": low_demand_moq,
        "partial_po_skus": partial_po_skus,
        "normal_inbound_skus": normal_inbound_skus,
    }
    scenario_warehouse_targets = select_scenario_warehouse_targets(
        scenario_targets,
        demand_by_sku_warehouse,
        warehouses["warehouse_id"].astype(str).tolist(),
    )
    datasets["inventory_snapshots"] = generate_inventory(
        rng, products, warehouses, datasets["sales_transactions"], end,
        scenario_targets, scenario_warehouse_targets,
    )
    datasets["purchase_orders"] = generate_purchase_orders(
        rng, products, warehouses, suppliers, policy, start, end, fx_table,
        scenario_targets, scenario_warehouse_targets,
    )
    quotations, quotation_lines = generate_quotations(rng, products, customers, start, end, fx_table)
    datasets["sales_quotations"] = quotations
    datasets["quotation_lines"] = quotation_lines
    datasets["subscription_records"] = generate_subscriptions(rng, products, customers, start, end, fx_table)

    quality_anomalies = inject_controlled_anomalies(datasets)
    scenario_log = build_business_scenario_log(
        scenario_targets, datasets["purchase_orders"], scenario_warehouse_targets, end
    )
    scenario_errors = validate_business_scenario_design(
        scenario_targets, datasets, demand_by_sku_warehouse, scenario_warehouse_targets, end
    )
    relation_errors = validate_base_relations(datasets)
    scenario_link_errors: list[str] = []
    required_po_scenarios = {"supplier_eta_revised", "normal_open_purchase_order", "partial_receipt_outstanding_balance"}
    valid_warehouse_ids = set(warehouses["warehouse_id"].astype(str))
    po_by_line = datasets["purchase_orders"].set_index("po_line_id", drop=False)
    for row in scenario_log.itertuples(index=False):
        has_po_link = bool(str(row.related_po_line_id).strip())
        if not str(row.warehouse_id).strip() or str(row.warehouse_id) not in valid_warehouse_ids:
            scenario_link_errors.append(f"{row.scenario_id}: scenario must reference a valid warehouse_id")
        if row.scenario_type in required_po_scenarios and not has_po_link:
            scenario_link_errors.append(f"{row.scenario_id}: scenario requires a related active PO line")
        if row.scenario_type in {"low_stock_no_confirmed_inbound", "excess_stock_cover", "minimum_order_constraint"} and has_po_link:
            scenario_link_errors.append(f"{row.scenario_id}: scenario should not reference an unrelated PO line")
        if has_po_link and row.related_po_line_id in po_by_line.index:
            po_row = po_by_line.loc[row.related_po_line_id]
            if str(po_row["sku"]) != str(row.sku):
                scenario_link_errors.append(f"{row.scenario_id}: related PO SKU does not match scenario SKU")
            if str(po_row["warehouse_id"]) != str(row.warehouse_id):
                scenario_link_errors.append(f"{row.scenario_id}: related PO warehouse does not match scenario warehouse")
    errors = scenario_errors + relation_errors + scenario_link_errors
    if errors:
        raise RuntimeError("Synthetic data validation failed:\n- " + "\n- ".join(errors))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove legacy outputs that are no longer part of the current scope.
    # Supplier/SKU compatibility is now represented by sku_sourcing_policy.csv,
    # and purchase-price history was removed from the core business use cases.
    obsolete_outputs = [
        "purchase_price_history.csv",
        "supplier_product_compability.csv",  # legacy filename spelling
        "supplier_product_compatibility.csv",  # standard spelling, if present
    ]
    for obsolete_name in obsolete_outputs:
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
    row_counts: dict[str, int] = {}
    for name, frame in datasets.items():
        filename = f"{name}.csv"
        frame.to_csv(output_dir / filename, index=False, date_format="%Y-%m-%d %H:%M:%S")
        row_counts[filename] = int(len(frame))
    quality_anomalies.to_csv(output_dir / "anomaly_log.csv", index=False)
    row_counts["anomaly_log.csv"] = int(len(quality_anomalies))
    scenario_log.to_csv(output_dir / "business_scenario_log.csv", index=False)
    row_counts["business_scenario_log.csv"] = int(len(scenario_log))

    po = datasets["purchase_orders"]
    inv = datasets["inventory_snapshots"]
    active_po = po.loc[po["status"].isin(["open", "partial"]) & po["remaining_quantity"].gt(0)]
    actual_lead = po.loc[po["status"].eq("received"), "actual_total_lead_time_days"].dropna()
    lead_time_by_route = {}
    received_po = po.loc[po["status"].eq("received")].copy()
    for lane, group in received_po.groupby("sourcing_lane"):
        values = group["actual_total_lead_time_days"].dropna()
        if len(values):
            lead_time_by_route[str(lane)] = {
                "received_po_count": int(len(values)),
                "p50_days": float(values.median()),
                "p75_days": float(values.quantile(0.75)),
                "p90_days": float(values.quantile(0.90)),
            }
    report = {
        "project": "Demand Intelligence & Inventory Risk Engine",
        "synthetic_data": True,
        "company_country": COMPANY_COUNTRY,
        "sales_market": "Indonesia domestic market only",
        "reporting_currency": REPORTING_CURRENCY,
        "seed": args.seed,
        "configured_start_date": start.date().isoformat(),
        "configured_end_date": end.date().isoformat(),
        "inventory_as_of_date": end.date().isoformat(),
        "product_count": int(len(products)),
        "hardware_sku_count": int(products["physical_inventory_flag"].sum()),
        "warehouse_count": int(len(warehouses)),
        "customer_count": int(len(customers)),
        "sales_target_rows": int(args.sales_target_rows),
        "sales_rows_before_quality_anomalies": int(max(0, len(datasets["sales_transactions"]) - 1)),
        "sales_target_reached_before_quality_anomalies": bool(max(0, len(datasets["sales_transactions"]) - 1) >= args.sales_target_rows),
        "sales_rows_including_quality_anomaly": int(len(datasets["sales_transactions"])),
        "sales_date_min": str(pd.to_datetime(datasets["sales_transactions"]["transaction_date"]).min().date()),
        "sales_date_max": str(pd.to_datetime(datasets["sales_transactions"]["transaction_date"]).max().date()),
        "sales_unique_skus": int(datasets["sales_transactions"]["sku"].nunique()),
        "hardware_demand_pattern_quality": demand_pattern_quality,
        "demand_pattern_quality_note": (
            "Diagnostic metrics use hardware SKUs, positive sales quantities, complete weeks only, "
            "and exclude weeks with no valid business days. Demand-type labels are generator QA metadata, "
            "not forecasting features. Metrics should be reviewed after every regeneration."
        ),
        "calendar_configuration": {
            "basis": "Indonesia national public holidays and collective leave from official SKB calendars for 2023-2026",
            "sources_by_year": HOLIDAY_CALENDAR_SOURCES,
            "source_urls_by_year": HOLIDAY_CALENDAR_SOURCE_URLS,
            "collective_leave_policy": args.collective_leave_policy,
            "collective_leave_policy_note": (
                "Collective leave is recorded separately. By default it is treated as non-business for this synthetic company; "
                "private-sector implementation may vary by employer policy. Use --collective-leave-policy workday to keep these dates operational."
            ),
            "public_holiday_dates_in_range": int(calendar["is_public_holiday"].sum()),
            "public_holiday_weekdays_in_range": int((calendar["is_public_holiday"] & calendar["is_weekday"]).sum()),
            "collective_leave_dates_in_range": int(calendar["is_collective_leave"].sum()),
            "collective_leave_weekdays_in_range": int((calendar["is_collective_leave"] & calendar["is_weekday"]).sum()),
            "collective_leave_weekdays_observed_in_range": int((calendar["is_collective_leave"] & calendar["is_weekday"] & (args.collective_leave_policy == "observe")).sum()),
            "weekend_days_in_range": int(calendar["is_weekend"].sum()),
            "business_days_in_range": int(calendar["is_business_day"].sum()),
            "company_closure_dates_in_range": int(calendar["is_company_closure"].sum()),
            "company_closure_ranges": COMPANY_CLOSURE_RANGES,
        },
        "sourcing_policy_rows": int(len(policy)),
        "purchase_order_line_count": int(len(po)),
        "active_open_or_partial_po_lines": int(len(active_po)),
        "active_incoming_remaining_quantity": int(active_po["remaining_quantity"].sum()),
        "received_po_line_count": int(po["status"].eq("received").sum()),
        "cancelled_po_line_count": int(po["status"].eq("cancelled").sum()),
        "partial_po_line_count": int(po["status"].eq("partial").sum()),
        "actual_lead_time_days_p50_received_pos": float(actual_lead.median()) if len(actual_lead) else None,
        "actual_lead_time_days_p90_received_pos": float(actual_lead.quantile(0.90)) if len(actual_lead) else None,
        "actual_lead_time_summary_by_sourcing_lane": lead_time_by_route,
        "inventory_snapshot_rows": int(len(inv)),
        "inventory_snapshot_dates": int(inv["snapshot_date"].nunique()),
        "data_quality_anomaly_count": int(len(quality_anomalies)),
        "business_scenario_count": int(len(scenario_log)),
        "scenario_skus": {key: value for key, value in scenario_targets.items()},
        "scenario_warehouse_targets": scenario_warehouse_targets,
        "scenario_warehouse_validation": "PASS",
        "referential_integrity_validation": "PASS",
        "business_scope": [
            "Core decisions: when to reorder, how much to reorder, and what inventory/supply risk requires review before PR.",
            "Forecasting population is physical inventory-managed hardware only.",
            "Daily calendar separates weekends, official Indonesian public holidays, collective leave, and optional company-specific closures.",
            "Inventory snapshot is a single as-of position; historical inventory movements are not simulated in this MVP.",
            "Incoming supply is derived from valid open/partial PO lines and remaining quantity; no On PO column is stored in inventory snapshots.",
            "Purchase price shock / purchase-price monitoring is excluded from core business scenarios.",
            "All lead-time, MOQ and demand values are synthetic assumptions; no values represent actual supplier commitments.",
            "Business scenario labels are stored in business_scenario_log.csv, not in operational feature columns.",
            "Data-quality issues are separately recorded in anomaly_log.csv.",
        ],
        "row_counts": row_counts,
    }
    with open(output_dir / "generation_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=json_safe)

    print("\n=== SYNTHETIC DATA GENERATION REPORT ===")
    print(f"Output directory          : {output_dir}")
    print(f"Random seed               : {args.seed}")
    print(f"Collective leave policy   : {args.collective_leave_policy}")
    print(f"Business days             : {int(calendar['is_business_day'].sum()):,}")
    print(f"Public holidays / cuti    : {int(calendar['is_public_holiday'].sum()):,} / {int(calendar['is_collective_leave'].sum()):,}")
    print(f"Configured date range     : {start.date()} to {end.date()}")
    print(f"Products / hardware SKUs  : {len(products):,} / {int(products['physical_inventory_flag'].sum()):,}")
    print(f"Warehouses / customers    : {len(warehouses):,} / {len(customers):,}")
    print(f"Sales rows                : {len(datasets['sales_transactions']):,}")
    print(f"Sourcing policy rows      : {len(policy):,}")
    print(f"PO lines / active PO lines: {len(po):,} / {len(active_po):,}")
    print(f"Inventory snapshot rows   : {len(inv):,} (as of {end.date()})")
    print(f"Data quality anomalies    : {len(quality_anomalies):,}")
    print(f"Business scenarios        : {len(scenario_log):,}")
    print("Core data validation      : PASS")
    print("\nRows written:")
    for filename, count in row_counts.items():
        print(f"  {filename:<40} {count:>10,}")
    print("\nGeneration completed. Review business_scenario_log.csv and generation_report.json.")



if __name__ == "__main__":
    main()
