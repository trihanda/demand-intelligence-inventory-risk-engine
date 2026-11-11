# Forecasting Target & Evaluation Metrics

## 1. Objective

The forecasting component estimates future product demand to support inventory planning and early detection of potential stock shortages. The prototype uses weekly SKU-level forecasts over a 13-week horizon and evaluates forecast quality in the context of inventory decisions.

**Current status:** The weekly demand panel, forecasting-ready dataset, and initial statistical/intermittent-demand benchmarks have been completed. LightGBM and Chronos-2 remain planned candidates and have not yet been evaluated. Results in this document are therefore interim benchmark findings, not a final model selection.

## 2. Forecasting Target and Dataset

**Target variable:** Weekly quantity sold per hardware SKU.

| Parameter | Implemented design |
|---|---|
| Forecast target | Weekly quantity sold |
| Forecast granularity | SKU-level, weekly (Monday–Sunday) |
| Product scope | Physical hardware products |
| Forecast horizon | 13 weeks (approximately 90 days) |
| Primary use case | Inventory planning and stock risk assessment |
| Data source | Synthetic sales transactions and supporting product/calendar data |
| Complete historical weeks | 195 weeks, 2 January 2023–27 September 2026 |
| SKU count | 600 hardware SKUs |
| Weekly panel size | 117,000 SKU-week rows (600 × 195) |

The weekly panel is a complete SKU-week grid. Weeks with no recorded sales are represented as zero demand. The panel contains 34,299 zero-demand SKU-weeks (29.3%). Two full-closure weeks are identified in the business calendar. The partial final week (28–30 September 2026) is excluded from the complete-week modeling panel; it contains 1,901 transaction rows and 4,828 units.

Recorded sales are used as a proxy for demand. Sales may understate unconstrained demand during stockout periods. The data are synthetic and do not establish performance on actual company data.

## 3. Forecasting-Ready Dataset and Time Split

The forecasting-ready dataset contains lagged demand, rolling statistics, product attributes, and calendar features. The current feature set contains 34 predictors. Demand-history features include lags 1, 2, 4, 13, 26, and 52 weeks, plus rolling means and variability measures. Lag and rolling features are shifted so that each forecast origin uses historical information only.

The chronological split is:

| Split | Weeks | Date range | Panel rows | Modeling-eligible rows |
|---|---:|---|---:|---:|
| Training | 169 | 2 Jan 2023–23 Mar 2026 | 101,400 | 70,200 |
| Validation | 13 | 30 Mar 2026–22 Jun 2026 | 7,800 | 7,800 |
| Test | 13 | 29 Jun 2026–21 Sep 2026 | 7,800 | 7,800 |
| **Total eligible** | **195** | | **117,000 panel rows** | **85,800** |

The first 52 weeks are not eligible for model rows that require the 52-week lag. This accounts for the difference between the full training panel (101,400 rows) and the training modeling-eligible rows (70,200).

Demand-pattern metadata is derived from training history and used for evaluation segmentation, not as a forecasting predictor. The resulting SKU counts are 308 smooth, 96 erratic, 119 intermittent, and 77 lumpy. The SKU classifications are consistent between the validation and test prediction records.

## 4. Forecasting Approaches

### 4.1 Completed benchmark candidates

The following methods have been evaluated using fixed-origin 13-week backtesting:

- **Closure-aware Seasonal Naive (52 weeks):** Uses the corresponding week from the prior year. When the source week is a full business closure, a business-day-matched fallback based on adjacent operational weeks is used. A target week that is itself a full closure is forecast as zero.
- **Croston:** Intermittent-demand benchmark. Full-closure historical weeks are excluded from occurrence-interval updates. Forecasts remain constant across open weeks in the fixed-origin horizon.
- **Croston-SBA:** Bias-adjusted Croston variant, evaluated with the same closure treatment.
- **TSB:** Intermittent-demand benchmark that updates demand occurrence probability. Future occurrence probability decays by the fixed factor (1 − beta) after each open target week; full-closure weeks do not cause decay.

Croston and TSB parameters were fixed at alpha = 0.10 and beta = 0.10. These values were not tuned on validation or test. Full-closure weeks are not treated as zero-demand opportunities in the Croston/TSB occurrence calculations.

### 4.2 Planned candidates

**LightGBM** remains the primary supervised global forecasting candidate. It will use the forecasting-ready features and will be evaluated against the same forecast origins, target weeks, horizon, and metrics as the existing benchmarks. Any hyperparameter selection must use training and validation data only.

**Chronos-2** remains an optional pretrained time-series foundation-model candidate. It has not yet been evaluated in this project. Any comparison must use compatible forecast origins, target weeks, and evaluation metrics.

### 4.3 Model selection principles

Model selection will consider WAPE, MAE, RMSE, forecast bias, stability across demand patterns and horizons, and operational usefulness for inventory planning. A lower WAPE alone is not sufficient if the model has material systematic underforecasting or overforecasting.

## 5. Evaluation Method

The completed statistical benchmark uses seven fixed-origin folds: five development folds, one validation fold, and one final test fold. Each fold predicts 13 weeks ahead for 600 SKUs. This produces 54,600 prediction rows per model (7 folds × 13 weeks × 600 SKUs).

At each origin, forecasts are generated using information available at that origin. Actual demand within the 13-week forecast horizon is not used to generate that fold's predictions. Full-closure calendar weeks are handled explicitly according to each method's policy.

The validation set is used to compare candidates and select the segmented-routing rule. The test set is held out for final comparison and must not be used to tune model parameters or revise the routing rule.

### Segmented-routing experiment

A routing candidate was constructed by selecting the model with the lowest validation WAPE within each training-derived demand pattern. The resulting rule is:

| Demand pattern | Model selected using validation |
|---|---|
| Smooth | Croston-SBA |
| Erratic | TSB |
| Intermittent | TSB |
| Lumpy | TSB |

The rule was then applied unchanged to test predictions. This is an experimental comparison; it is not yet the selected production strategy.

## 6. Evaluation Metrics

| Metric | Purpose |
|---|---|
| WAPE (Weighted Absolute Percentage Error) | Primary aggregate measure of absolute forecast error relative to total actual demand |
| MAE (Mean Absolute Error) | Average absolute error in units sold |
| RMSE (Root Mean Squared Error) | Gives greater weight to large forecast errors |
| Forecast bias | Identifies systematic overforecasting or underforecasting |

### WAPE

WAPE is calculated as:

`WAPE = sum(|Forecast − Actual|) / sum(Actual) × 100%`

WAPE is reported only when total actual demand is greater than zero. Results are also reviewed by demand pattern and forecast horizon so aggregate performance does not conceal weaker segments.

### MAE and RMSE

`MAE = mean(|Forecast − Actual|)`

`RMSE = sqrt(mean((Forecast − Actual)²))`

Both are expressed in the same unit as weekly quantity sold.

### Forecast Bias

`Forecast Bias = sum(Forecast − Actual) / sum(Actual) × 100%`

Positive bias indicates overforecasting; negative bias indicates underforecasting. Bias is interpreted alongside error metrics because underforecasting can increase stockout exposure, while overforecasting can increase excess-inventory risk.

MAPE is not used as the primary metric because it is undefined when actual demand is zero and can be misleading for low-volume or intermittent-demand products. MASE may be added in a later comparison if its scaling factor is calculated using training data only.

## 7. Interim Benchmark Results

The following results are from the completed seven-fold benchmark. Croston and TSB parameters were fixed at 0.10 and were not tuned on validation or test.

### 7.1 Overall global-model comparison

| Model | Validation WAPE | Validation bias | Test WAPE | Test MAE | Test RMSE | Test bias |
|---|---:|---:|---:|---:|---:|---:|
| Closure-aware Seasonal Naive | 62.87% | −7.44% | 58.61% | 7.54 | 16.11 | +0.63% |
| Croston | 64.64% | +28.82% | 56.43% | 7.26 | **12.20** | −33.92% |
| Croston-SBA | 61.28% | +22.38% | 58.12% | 7.47 | 12.42 | −37.23% |
| TSB | 60.91% | −26.53% | 72.74% | 9.35 | 14.72 | −62.73% |

On the test period, Croston has the lowest WAPE among the global candidates (56.43%) and the lowest RMSE (12.20). The closure-aware Seasonal Naive has a higher WAPE (58.61%) but nearly neutral aggregate bias (+0.63%). Croston's test bias (−33.92%) indicates substantial aggregate underforecasting. These results illustrate why model comparison must consider both forecast error and bias.

### 7.2 Segmented-routing results

| Demand pattern | Selected model | Validation WAPE | Validation bias | Test WAPE | Test bias |
|---|---|---:|---:|---:|---:|
| Smooth | Croston-SBA | 33.34% | +19.88% | 40.63% | −39.64% |
| Erratic | TSB | 80.89% | −31.33% | 76.94% | −61.82% |
| Intermittent | TSB | 153.09% | −12.14% | 122.26% | −59.33% |
| Lumpy | TSB | 195.89% | +9.59% | 136.06% | −56.41% |

The unchanged routing rule produces the following aggregate metrics:

| Metric | Validation | Test |
|---|---:|---:|
| WAPE | 54.94% | 56.39% |
| MAE | 4.20 | 7.25 |
| RMSE | 7.91 | 12.94 |
| Bias | +7.09% | −45.83% |

On test, segmented routing has WAPE of 56.39%, only 0.04 percentage point lower than global Croston (56.43%). Its RMSE is higher (12.94 versus 12.20) and its aggregate underforecasting is more severe (bias −45.83% versus −33.92%). Therefore, the current experiment does not demonstrate a material or operationally convincing advantage over the global Croston benchmark. Segmented routing remains exploratory and should not yet be treated as the final model strategy.

## 8. Inventory Planning Considerations

Forecast accuracy alone does not establish whether inventory decisions will improve. Forecasts should be assessed alongside:

- Current on-hand inventory.
- Reserved inventory.
- Incoming purchase orders and expected receipt dates.
- Product replenishment lead time.
- Expected demand before replenishment.
- Potential stock coverage during the replenishment period.

The risk assessment must consider the timing of expected demand and incoming supply, rather than relying only on total demand over the 13-week horizon. Forecasts support inventory risk assessment but do not automatically determine purchase quantities.

## 9. Limitations

- The data are synthetic and may not reflect real-world demand complexity; results do not establish performance on actual company data.
- Recorded sales may not represent unconstrained demand during stockout periods.
- Croston and TSB parameters were fixed at 0.10 and were not tuned; other settings may produce different results.
- Current intermittent-demand benchmarks show substantial segment-level errors and, in several cases, material underforecasting.
- The segmented-routing experiment selects models by validation WAPE; its test result shows significant negative bias and no material WAPE improvement over global Croston.
- LightGBM and Chronos-2 have not yet been evaluated.
- Forecasts are fractional in the current benchmark output. Any rounding or non-negative post-processing policy should be evaluated separately rather than assumed to improve accuracy.
- Forecasting accuracy is only one input to inventory decisions; service levels, lead-time demand, safety stock, and replenishment constraints require separate evaluation.

## 10. Next Steps

1. Implement and backtest LightGBM using the same fixed-origin folds and 13-week horizon.
2. Tune model parameters on training/validation data only, then evaluate the locked model on test.
3. Compare LightGBM against the existing global benchmarks using WAPE, MAE, RMSE, bias, demand-pattern segments, and forecast horizon.
4. Reassess whether segmented routing adds value after the LightGBM results are available.
5. Consider Chronos-2 as an optional comparison if a reproducible and computationally practical evaluation can be run.
6. Connect the selected forecast output to inventory risk calculations only after forecast bias and operational implications have been assessed.

## 11. Expected Outputs

- Validated weekly demand dataset and forecasting-ready feature dataset.
- Reproducible fixed-origin 13-week backtest predictions and reports.
- Baseline and candidate model comparisons across overall metrics, demand patterns, and forecast horizons.
- Documented forecast bias and inventory-planning implications.
- Forecast outputs suitable for the subsequent inventory risk assessment component.
