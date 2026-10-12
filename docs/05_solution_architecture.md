# Solution Architecture

## 1. Objective

Define the proposed architecture for the Demand Intelligence & Inventory Risk Engine, covering data processing, demand forecasting, inventory risk assessment, dashboard reporting, and automated notifications.

The architecture is designed for a portfolio prototype using synthetic data and does not assume access to live enterprise systems.

## 2. Architecture Overview

The proposed workflow consists of five layers.

### Layer 1: Data Sources

Synthetic business datasets will represent:

- Historical sales transactions.
- Inventory snapshots.
- Purchase orders.
- Sales quotations and quotation line items.
- Subscription records.
- Product, customer, and warehouse master data.

### Layer 2: Data Processing and Storage

Python and pandas will be used to:

- Generate, validate, and clean synthetic source data.
- Standardize dates and identifiers.
- Load source datasets into the database.
- Prepare data for analytical processing.

PostgreSQL and SQL will be used to:

- Store structured business datasets.
- Join related tables and apply business transformations.
- Aggregate historical demand into weekly observations.
- Prepare analysis-ready tables for forecasting and reporting.

### Layer 3: Analytics and Risk Engine

The analytics layer will include:

- Seasonal Naive baseline and LightGBM forecasting.
- Chronos-2 as an alternative forecasting model.
- Inventory risk calculations based on forecasts, available stock, and incoming purchase orders.
- Quotation aging and pipeline analysis.
- Subscription renewal monitoring.
- Rules-based alert triggering.

Python will handle forecasting, model evaluation, and risk calculations. SQL will prepare the structured data and analytical datasets used by these processes.

### Layer 4: Reporting and Notifications

**Power BI**

- Monitor demand forecasts and inventory risk.
- Analyze quotation pipeline and subscription renewals.
- Investigate detailed results and supporting KPIs.

**Microsoft Teams**

- Notify relevant users when predefined conditions are triggered.
- Provide alert priority, supporting evidence, and recommended actions.
- Include AI-generated summaries based on validated results.

Power Automate or another suitable integration method may be used to deliver notifications, depending on account access, licensing, and technical feasibility.

### Layer 5: Decision Support

Users review dashboard details and alerts, investigate the underlying data, and decide on appropriate operational actions.

The prototype will not automatically approve quotations, create purchase orders, or execute inventory decisions.

## 3. Proposed Technology Stack

| Component | Proposed Technology | Purpose |
|---|---|---|
| Data generation and processing | Python, pandas | Generate synthetic data, validate records, and prepare datasets |
| Data storage | PostgreSQL | Store structured business datasets |
| Data transformation | SQL | Join tables, aggregate demand, and prepare analytical datasets |
| Forecasting | Seasonal Naive, LightGBM, Chronos-2 | Generate and compare demand forecasts |
| Risk assessment | Python | Calculate inventory risk and trigger predefined alert conditions |
| Business intelligence | Power BI | Visualize KPIs, forecasts, and risk indicators |
| Workflow automation | Microsoft Power Automate or an equivalent integration approach | Deliver notifications where supported |
| Notifications | Microsoft Teams | Deliver proactive risk alerts |
| AI-generated summaries | An available LLM integration | Explain validated alert evidence and recommended actions |
| Version control | Git and GitHub | Maintain code, documentation, and version history |

Technology choices may be refined during implementation based on integration requirements, licensing, and available resources. CSV files may be used for initial data exchange and intermediate outputs where appropriate.

## 4. Data Flow

1. Generate synthetic business datasets using Python.
2. Validate source data and load structured datasets into PostgreSQL.
3. Use SQL to transform and join data into analysis-ready tables.
4. Build weekly demand datasets and generate forecasts using Python.
5. Calculate inventory, quotation, and subscription risk indicators.
6. Store analytical outputs in PostgreSQL and export datasets for Power BI where appropriate.
7. Evaluate predefined alert conditions.
8. Deliver relevant Microsoft Teams notifications with supporting information, where integration is available.
9. Generate AI summaries from validated alert results, where an LLM integration is available.
10. Allow users to review the evidence and determine the next action.

## 5. Refresh and Automation

The target design is a repeatable workflow with scheduled data processing and dashboard refresh.

For the initial prototype, data generation, SQL transformations, forecasting, and risk calculations may run locally or through a scheduled workflow.

Power BI refresh, Microsoft Teams notifications, and AI-generated summaries will be implemented only where technically feasible with the available accounts, licenses, and integration methods.

If a component is simulated rather than connected to a live service, this will be documented explicitly.

## 6. Architecture Constraints

- The project uses synthetic data rather than live company records.
- PostgreSQL is the proposed database for structured data storage and SQL-based transformations.
- CSV files may be used for initial data loading, data exchange, and intermediate outputs.
- Power BI refresh and Teams notification capabilities depend on the selected deployment, account access, licensing, and integration approach.
- AI summaries must use validated risk outputs and must not independently invent facts or determine risk classifications.
- The prototype does not guarantee production-level security, scalability, or reliability.

## 7. Expected Outcome

A documented and implementable architecture connecting synthetic business data, SQL-based data preparation, forecasting, risk assessment, Power BI reporting, and Microsoft Teams notifications into a decision-support workflow.