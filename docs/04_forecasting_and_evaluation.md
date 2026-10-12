# Forecasting Target & Evaluation Metrics

## 1. Objective

Define the forecasting target, forecast horizon, evaluation methodology, and performance metrics for the Demand Intelligence & Inventory Risk Engine.

The forecasting component will estimate future product demand to support inventory planning and early detection of potential stock shortages.

## 2. Forecasting Target

**Target variable:** Weekly quantity sold per product.

| Parameter | Initial Design |
|---|---|
| Forecast target | Weekly quantity sold |
| Forecast granularity | Product-level, weekly |
| Historical period | 36 months |
| Forecast horizon | 13 weeks (approximately 90 days) |
| Primary use case | Inventory planning and stock risk assessment |
| Initial scope | Physical hardware products |
| Data source | `sales_transactions` |

Historical transactions will be aggregated into weekly product-level quantities. Missing weeks will be handled explicitly, distinguishing periods with no recorded sales from periods with missing data.

Warehouse-level forecasting may be introduced if sufficient demand history and reliable warehouse allocation are available.

Recorded sales will be used as a proxy for demand. Sales may understate actual demand during stockout periods, so this limitation will be considered during data generation and evaluation.

## 3. Forecasting Approach

The prototype will use a benchmark-driven machine learning approach.

### Baseline

Seasonal Naive will be evaluated where sufficient seasonal history is available. A historical-average baseline may also be used when appropriate.

### Primary Candidate: LightGBM

LightGBM will be evaluated as the primary global forecasting model across eligible products.

Potential features include:

- Lagged demand: previous weekly quantities.
- Rolling statistics: historical demand averages and variability.
- Calendar features: week of year and month.
- Product attributes: category and replenishment lead time, where relevant.

All features must be derived exclusively from information available at the forecast origin to prevent data leakage.

### Alternative Candidate: XGBoost

XGBoost may be evaluated if additional model comparison is justified by the results of the baseline and LightGBM experiments.

### Multi-Step Forecasting Strategy

The initial implementation will use recursive forecasting to generate predictions for the 13-week horizon. Each subsequent prediction will use previously predicted demand where future lagged observations are unavailable.

The evaluation will account for potential error accumulation across the forecast horizon. Alternative multi-step strategies may be considered if recursive forecasting performs poorly.

### Model Selection

The final model will be selected based on out-of-sample performance, forecast bias, computational practicality, and usefulness for inventory planning.

A more complex model will not automatically be considered better than a simpler baseline.

## 4. Evaluation Method

The dataset will be split chronologically to prevent future information from leaking into model training.

- **Training set:** Earlier observations used to fit the model.
- **Validation set:** Observations used for model comparison and parameter tuning.
- **Test set:** Later observations reserved for final evaluation.

A rolling-origin evaluation will be used where feasible. At each forecast origin, the model will use only information available at that time to predict the following 13 weeks.

Forecast performance will be evaluated across the full horizon and by forecast horizon where sufficient observations are available.

The final date ranges and split proportions will be determined after the synthetic dataset has been generated and validated.

## 5. Evaluation Metrics

Forecast performance will be assessed using complementary metrics.

| Metric | Purpose |
|---|---|
| WAPE (Weighted Absolute Percentage Error) | Primary aggregate measure of forecast error relative to total actual demand |
| MAE (Mean Absolute Error) | Measures average absolute forecast error in units sold |
| Forecast Bias | Identifies systematic overforecasting or underforecasting |
| MASE (Mean Absolute Scaled Error) | Optional measure comparing forecast error with a naive benchmark |

### WAPE

WAPE will be the primary aggregate metric:

WAPE = Sum of absolute forecast errors / Sum of actual demand × 100%

WAPE will be reported only when total actual demand is greater than zero. Results will also be reviewed across relevant product groups and forecast horizons to avoid aggregate performance hiding poor results for individual segments.

### MAE

MAE will measure the average absolute difference between predicted and actual weekly quantities, expressed in units.

### Forecast Bias

Forecast bias will be calculated as:

Forecast Bias = Sum of (Forecast − Actual) / Sum of Actual × 100%

Positive bias indicates overforecasting, while negative bias indicates underforecasting. The metric will be interpreted alongside WAPE and MAE because overforecasting and underforecasting can have different operational consequences.

### MASE

MASE may be included to compare forecast performance with a naive benchmark, particularly across products with different demand scales. Its scaling factor will be calculated using training data only.

MAPE will not be used as the primary metric because it is undefined when actual demand is zero and can be misleading for low-volume or intermittent-demand products.

## 6. Inventory Planning Considerations

Forecast accuracy alone does not establish whether inventory decisions will improve.

Forecasts will be assessed alongside:

- Current on-hand inventory.
- Reserved inventory.
- Incoming purchase orders and expected receipt dates.
- Product replenishment lead time.
- Expected demand before replenishment.
- Potential stock coverage during the replenishment period.

The risk assessment must consider the timing of expected demand and incoming supply, rather than relying only on total demand over the 13-week horizon.

Forecasts will support inventory risk assessment but will not automatically determine purchase quantities.

## 7. Success Criteria

The forecasting component will be considered useful for the prototype if:

- It produces forecasts for eligible products across the 13-week horizon.
- It is evaluated against the selected baseline using the same test periods.
- Forecast errors and bias are documented across relevant forecast horizons.
- The evaluation process is reproducible.
- Results can be interpreted in the context of inventory and replenishment decisions.

The preferred model should demonstrate competitive or improved out-of-sample performance against the baseline. If it does not, the baseline may remain the preferred method.

No numerical accuracy target or business improvement will be claimed before evaluation.

## 8. Limitations

- Synthetic data may not reflect real-world demand complexity.
- Performance on synthetic data does not establish performance on actual company data.
- Products with sparse or intermittent demand may require separate evaluation.
- Recorded sales may not represent unconstrained demand during stockout periods.
- Recursive forecasts may accumulate errors over longer horizons.
- Model selection and evaluation settings may be refined after data profiling.

## 9. Expected Output

- Validated weekly demand dataset.
- Baseline forecast results.
- LightGBM forecast results and optional XGBoost comparison.
- Out-of-sample evaluation across the 13-week horizon.
- WAPE, MAE, forecast bias, and optional MASE results.
- Forecast outputs suitable for inventory risk analysis.