# Sales Data Profiling Report

## 1. Objective

This analysis establishes a baseline understanding of the sales transaction dataset before data quality assessment, weekly demand aggregation, and demand forecasting.

The profiling focuses on transaction volume, historical coverage, SKU activity, and transaction quantity distribution.

## 2. Dataset Overview

| Metric | Result |
|---|---:|
| Dataset type | Synthetic sales transaction data |
| Transaction records | 32,134 |
| Number of columns | 10 |
| Unique SKUs | 100 |
| First transaction date | 2023-10-01 |
| Last transaction date | 2026-09-27 |
| Historical coverage | 1,093 calendar days |
| Missing or invalid dates | 0 |
| Missing SKU values | 0 |
| Missing or invalid quantities | 0 |

The dataset contains approximately three years of transaction history across 100 unique SKUs. All records have parseable transaction dates, non-missing SKU identifiers, and quantities that can be converted to numeric values.

## 3. Transaction Quantity Distribution

| Statistic | Result |
|---|---:|
| Mean quantity | 8.89 units |
| Median quantity | 7 units |
| Standard deviation | 7.45 units |
| 25th percentile | 4 units |
| 75th percentile | 11 units |
| 95th percentile | 24 units |
| 99th percentile | 38 units |
| Minimum quantity | -1 unit |
| Maximum quantity | 68 units |

The mean transaction quantity is higher than the median, indicating that larger transactions influence the average.

One transaction contains a negative quantity. Its business meaning has not yet been established. It may represent a return, reversal, or intentionally introduced data anomaly and requires further investigation.

## 4. SKU Transaction Coverage

All 100 SKUs appear at least once in the transaction dataset.

| Metric | Result |
|---|---:|
| Minimum transactions per SKU | 85 |
| Median transactions per SKU | 381.5 |
| Average transactions per SKU | 321.34 |
| Maximum transactions per SKU | 430 |

Transaction frequency varies across SKUs. This indicates differences in recorded transaction activity, but transaction counts alone do not establish demand intensity, demand intermittency, or forecasting suitability.

The number of transactions per SKU must also be distinguished from the number of weeks with demand. Weekly activity will be evaluated after the data is standardized and aggregated.

## 5. Key Findings

1. **Historical coverage:** The dataset spans approximately three years, providing an initial basis for analyzing demand trends and potential seasonality.
2. **SKU coverage:** All 100 SKUs have recorded transactions, although transaction frequency varies considerably.
3. **Quantity distribution:** The median transaction quantity is 7 units, compared with a mean of 8.89 units.
4. **Data quality investigation:** One negative quantity requires classification before the forecasting dataset is finalized.
5. **Weekly demand is not yet established:** Transaction-level statistics cannot directly describe weekly demand patterns.

## 6. Limitations and Assumptions

- The dataset is synthetic and does not represent verified business performance.
- The absence of a transaction does not automatically imply zero demand.
- Stock availability and stockout periods must be considered when interpreting observed sales.
- Negative quantities have not yet been classified as returns, reversals, or anomalies.
- No data cleaning, weekly aggregation, or forecasting was performed in this analysis.

## 7. Output Artifacts

The profiling script generates the following files:

- `sales_data_profile.json` — structured dataset and transaction coverage summary.
- `sku_transaction_coverage.csv` — transaction frequency and historical coverage by SKU.
- `transaction_quantity_distribution.csv` — descriptive statistics for transaction quantities.

## 8. Next Steps

The next analysis will assess missing values, duplicate transaction identifiers, invalid values, transaction statuses, and consistency between quantity, unit price, and line amount.

The findings will guide data cleaning and the preparation of a validated weekly demand dataset for forecasting.

---

*This report documents the initial profiling stage of the Demand Intelligence & Inventory Risk Engine portfolio project.*