# Data Preparation Summary

## Objective and scope

This report summarizes the data preparation pipeline for the **Demand Intelligence & Inventory Risk Engine**, a portfolio prototype for weekly demand forecasting and inventory-risk analysis at a fictional B2B technology solutions provider.

The data is synthetic. Findings reflect the generator and its assumptions, not verified market behavior or business impact.

| Setting | Definition |
|---|---|
| Forecasting population | Physical inventory-managed hardware |
| Demand grain | SKU-week, aggregated across warehouses |
| Week definition | Monday–Sunday, identified by `week_start` |
| Configured data/calendar window | 1 Oct 2023–30 Sep 2026; transactions run 2 Oct 2023–30 Sep 2026 |
| Forecast horizon | 13 weeks |
| Primary use case | Inventory planning and stock-risk assessment |

## Pipeline summary

| Stage | Main result | Detailed artifacts |
|---|---|---|
| Sales profiling | 500,001 source rows, 15 columns, 1,000 SKUs | `reports/sales_data_profiling/` |
| Data quality | 1 missing cell; 5 rule-level issue records across 4 source rows | `reports/data_quality_assessment/` |
| Standardization | 499,999 transactions retained; 2 excluded | `reports/data_standardization/`; processed CSVs in `data/processed/` |
| Weekly aggregation | Complete panel of 1,000 SKUs × 157 weeks = 157,000 rows | `reports/weekly_demand_aggregation/`; `data/processed/weekly_demand.csv` |
| Demand pattern analysis | 514 smooth, 187 intermittent, 175 erratic, 124 lumpy SKUs | `reports/demand_pattern_analysis/` |
| Trend and calendar analysis | 600 hardware SKUs, 94,200 SKU-weeks | `reports/seasonality_trend/` |
| Forecasting-ready preparation | 94,200 rows; 62,400 eligible rows after feature-history checks | `reports/forecasting_data_preparation/`; `data/processed/forecasting_ready_dataset.csv` |

## Key findings and decisions

### Data quality and standardization

The source has one missing `customer_id`, one negative-quantity row, one negative-line-amount row, and no line-amount mismatches. The five issue records are rule-level findings rather than five distinct transactions: four source rows are involved. All source rows have `posted` status.

Two rows were excluded during standardization: one with negative quantity and line amount, and one later occurrence of a duplicate business-key candidate. The three exclusion-reason flags therefore apply to two unique rows. A missing `customer_id` was retained when it did not prevent SKU-level demand calculation. Excluded records are preserved separately for traceability.

### Weekly demand and demand patterns

Weekly demand is aggregated across warehouses and zero-filled to create a complete SKU-week panel. Demand patterns use ADI = 1.32 and CV² = 0.49 thresholds; CV² is calculated from positive-demand weekly quantities only.

| Pattern | Meaning | SKUs | Share |
|---|---|---:|---:|
| Smooth | Frequent demand with relatively stable quantities | 514 | 51.4% |
| Intermittent | Infrequent demand, with relatively consistent quantities when it occurs | 187 | 18.7% |
| Erratic | Frequent demand with highly variable quantities | 175 | 17.5% |
| Lumpy | Infrequent demand with highly variable quantities | 124 | 12.4% |
| **Total** |  | **1,000** | **100%** |

For the 600 hardware SKUs, training-only segmentation is 308 smooth (51.3%), 120 intermittent (20.0%), 96 erratic (16.0%), and 76 lumpy (12.7%). This uses the 130-week training window, so its distribution is not directly comparable with the exploratory results above. These labels describe demand behavior; they are not model predictors.

### Hardware demand profile and calendar

For the 600 hardware SKUs, total demand was **1,213,876 units** over 157 weekly periods. Aggregate weekly demand averaged **7,731.69 units** (standard deviation **1,920.22**); the fitted linear slope was about **+5.45 units per week**. Average demand in the final 13 weeks was **28.1% higher** than in the first 13 weeks. These are descriptive comparisons, not evidence that growth will persist. The exploratory trend and quarterly summaries include the final partial week starting 28 Sep 2026 (calendar coverage ends 30 Sep); interpret the final 13-week average and quarter comparisons cautiously. The forecasting split excludes that partial week.

Quarter 3 had the highest average weekly demand (**9,379.35 units**) and Quarter 2 the lowest (**6,070.56**); Q3 was about **54.5% higher** than Q2. Annual weekly-profile correlations were **0.603** for 2024 vs. 2025 (52 comparable weeks), **0.627** for 2024 vs. 2026 (39 weeks), and **0.592** for 2025 vs. 2026 (39 weeks). These results are exploratory: quarterly demand factors are built into the synthetic generator, and the overlaps do not establish stable real-world seasonality.

The configured synthetic calendar spans **1 Oct 2023–30 Sep 2026** (1,096 dates). The demand series has 156 complete calendar weeks and one partial week, with no weeks lacking calendar coverage. Thirteen weeks overlap configured closure periods; five zero-demand weeks also overlap closures. This overlap does not establish causation. Closure dates are synthetic assumptions, not verified official Indonesian holidays.

## Forecasting-ready dataset

The hardware panel contains **600 SKUs × 157 weeks = 94,200 rows**. The target is `weekly_quantity`. Features include demand lags (1, 2, 4, 13, 26, 52 weeks), shifted rolling means (4, 13, 26 weeks), shifted 13-week rolling standard deviation, calendar fields, and cyclical week/month encodings. Lag and rolling features use prior weeks only; the generator field `demand_type_seed` is excluded to avoid leakage.

| Split | Weeks | Week range | Rows | Modeling-eligible rows |
|---|---:|---|---:|---:|
| Train | 130 | 2023-10-02–2026-03-23 | 78,000 | 46,800 |
| Validation | 13 | 2026-03-30–2026-06-22 | 7,800 | 7,800 |
| Test | 13 | 2026-06-29–2026-09-21 | 7,800 | 7,800 |
| Partial week excluded | 1 | Starts 2026-09-28 | 600 | 0 |
| **Total** | **157** |  | **94,200** | **62,400** |

The remaining **31,200 rows** lack sufficient feature history, primarily for the 52-week lag. Training-window segmentation for the hardware population is 308 smooth, 120 intermittent, 96 erratic, and 76 lumpy SKUs. The `intermittent_demand_flag_train` (ADI ≥ 1.32) marks 196 SKUs (32.7%), including both intermittent and lumpy patterns. Segments are metadata, not model predictors.

The preparation report records passing checks for unique SKU-week keys, expected grid size, target completeness and non-negativity, calendar integrity, partial-week exclusion, and leakage-safe lags. The chronological splits can support rolling one-step-ahead evaluation, but do not by themselves implement a fixed-origin 13-week forecast.

## Limitations and next steps

- Zero-filling assumes complete transaction capture in this synthetic dataset. In production, absent sales may reflect missing feeds, inactive products, or stockouts that censor demand.
- Sales demand alone does not describe inventory on hand, inbound purchase orders, lead times, reservations, or warehouse transfers.
- Quarterly profiles and annual correlations are exploratory; seasonality should be assessed through out-of-sample model comparison.
- Use the same forecast origins, 13-week horizon, eligible SKU population, and metrics for all model comparisons.

**Next:** implement a fixed-origin 13-week **Seasonal Naive** baseline, then evaluate **LightGBM** against it. Use **WAPE** as the primary metric, with **MAE** and forecast bias as supporting metrics. During a fixed-origin horizon, do not update lag or rolling features with actual demand observed inside that horizon. Chronos-2 can be considered as a later alternative. Do not claim model accuracy or business impact until evaluation has been run.
