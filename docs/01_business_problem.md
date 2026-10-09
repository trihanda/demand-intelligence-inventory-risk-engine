# Business Problem Definition

## 1. Business Context

This project uses a fictional B2B technology solutions provider that sells IT hardware, software licenses, and subscription-based products to enterprise customers, system integrators, and resellers.

The company manages commercial and operational data, including sales quotations, historical sales, inventory, purchase orders, and subscription renewals.

## 2. Business Problem

The company lacks an integrated, forward-looking view of demand, inventory risks, sales opportunities, and subscription renewals. As a result, identifying potential issues and deciding which actions require attention may depend on manual reporting and follow-up.

The project addresses four key business problems:

1. **Demand Forecasting & Inventory Risk**  
   Limited visibility into future product demand makes it difficult to anticipate stock shortages and assess whether available inventory can cover expected demand until replenishment.

2. **Sales Quotation Intelligence**  
   Limited visibility into quotation conversion, aging, and follow-up priorities makes it harder to identify promising sales opportunities and quotations that may require attention.

3. **Subscription Renewal Intelligence**  
   Upcoming subscription expirations and renewal outcomes need to be monitored to identify renewals that require follow-up and potential recurring revenue exposure.

4. **Delayed Risk Detection & Follow-up**  
   Relying on manual report reviews can delay the identification of operational and commercial risks. Relevant users need timely alerts with clear explanations and recommended actions.

## 3. Project Objectives

The project aims to:

- Forecast product demand to support inventory planning.
- Identify potential stock shortages and replenishment risks.
- Monitor quotation conversion, aging, and follow-up priorities.
- Track upcoming subscription renewals and renewal outcomes.
- Deliver proactive Microsoft Teams alerts with AI-generated summaries explaining the trigger, supporting evidence, and recommended actions.

Power BI will provide an interactive dashboard for ongoing monitoring and investigation, while Microsoft Teams will deliver alerts when predefined risk or follow-up conditions are met.

## 4. Scope

**In scope**
- Historical sales analysis and demand forecasting.
- Inventory risk assessment across multiple warehouses.
- Sales quotation and pipeline analysis.
- Subscription renewal monitoring.
- Rules-based risk detection and Microsoft Teams alerts.
- AI-generated summaries based on validated data and predefined rules.
- Power BI dashboard for monitoring and decision support.

**Out of scope**
- Automatic purchase order creation or approval.
- Automatic quotation approval or sales decisions.
- Production deployment or live system integration unless explicitly implemented as part of the prototype.

## 5. Key Assumptions & Limitations

- This is a portfolio prototype based on a fictional business scenario.
- The initial design assumes 36 months of historical data, weekly demand analysis, multiple warehouses, and a 90-day forecasting horizon. These assumptions will be validated against available data.
- A won quotation represents a commercial outcome, not proof of a completed sale or realized revenue. Sales order or invoice data is required to verify actual sales.
- Purchase orders must be evaluated using their status and expected receipt dates; ordered quantities are not equivalent to available stock.
- Upcoming subscription renewals represent potential renewal opportunities, not guaranteed revenue.
- Synthetic data may be used where real data is unavailable and will be clearly labelled.

## 6. Success Criteria

The prototype will be evaluated based on:

- **Forecasting:** Performance against a baseline model using an appropriate error metric.
- **Inventory risk:** Ability to identify potential stock shortages and provide useful warning time.
- **Sales quotations:** Visibility into conversion, aging, and follow-up priorities.
- **Renewals:** Visibility into upcoming expirations and renewal follow-up requirements.
- **Alerts:** Correct triggering, relevant evidence, clear recommended actions, and successful notification delivery.
- **Data pipeline:** Reliable processing, data validation, and refresh performance.

Actual business improvements, such as reduced stockouts or increased revenue, will not be claimed unless supported by measured results.

## 7. Expected Outcome

The expected outcome is a prototype decision-support system that connects demand forecasting, inventory risk assessment, commercial pipeline monitoring, and subscription renewal tracking in one workflow.

The system will help users identify what requires attention, understand why it matters, and determine the next appropriate action through Power BI dashboards and proactive Microsoft Teams alerts.