# Target Users & Decision Use Cases

## 1. Target Users

The system is designed to support the following business roles in a B2B technology solutions provider.

| User | Primary Responsibility | Information Needed |
|---|---|---|
| Sales Operations | Monitor quotation activity, sales pipeline, and follow-up priorities | Quotation status, aging, conversion trends, and pending follow-ups |
| Inventory / Supply Chain | Maintain stock availability and coordinate replenishment | Demand forecasts, on-hand stock, incoming purchase orders, and stock risk |
| Subscription / Account Management | Monitor subscription expirations and coordinate renewals | Upcoming expiry dates, renewal status, and accounts requiring follow-up |
| Management | Monitor business performance and prioritize actions | Key KPIs, high-priority risks, and overall commercial and inventory trends |

These roles represent the intended users of the prototype. Actual responsibilities may vary by company.

## 2. Decision Use Cases

### UC-01: Identify Potential Stock Shortages

**User:** Inventory / Supply Chain

**Decision:** Which products and warehouses require attention to prevent potential stock shortages?

**Required information:**
- Historical product demand and forecast.
- Current on-hand inventory.
- Incoming purchase orders and expected receipt dates.
- Estimated stock coverage and replenishment lead time.

**Expected action:** Review at-risk products and prioritize replenishment or stock transfers where appropriate.

### UC-02: Prioritize Sales Quotation Follow-ups

**User:** Sales Operations

**Decision:** Which open quotations should be prioritized for follow-up?

**Required information:**
- Quotation status and age.
- Quotation value and customer.
- Quotation history and available conversion patterns.
- Last update or follow-up date, if available.
- Approval turnaround time, if relevant timestamps are available.

**Expected action:** Identify aging quotations and coordinate follow-ups with the relevant sales representatives.

Won quotations are not treated as realized sales or revenue unless supported by separate transaction data.

### UC-03: Monitor Upcoming Subscription Renewals

**User:** Subscription / Account Management

**Decision:** Which subscriptions require renewal follow-up, and when?

**Required information:**
- Subscription expiry date.
- Customer and product information.
- Renewal status and outcome, if available.
- Contract value, if available and validated.

**Expected action:** Prioritize upcoming renewals and follow up with customers before expiration.

### UC-04: Detect Risks and Notify Relevant Users

**User:** Relevant operational teams and management

**Decision:** Which emerging risks require immediate attention, and what should happen next?

**Required information:**
- Triggered risk condition.
- Supporting data and relevant thresholds.
- Risk priority.
- Recommended next action and responsible role.

**Expected action:** Send a Microsoft Teams alert with an AI-generated summary explaining the trigger, supporting evidence, and recommended action. Users can investigate further through the Power BI dashboard.

AI-generated summaries must be based on validated results and predefined risk rules. Actual notification integration depends on technical feasibility and available account access.

## 3. Decision Support Workflow

The prototype follows this workflow:

1. Collect and process available business data.
2. Calculate forecasts, KPIs, and risk indicators.
3. Display overall performance and detailed information in Power BI.
4. Trigger Microsoft Teams alerts when predefined conditions are met.
5. Provide an AI-generated explanation based on validated results.
6. Allow users to review the evidence and decide on the appropriate action.

The system supports decision-making; it does not automatically approve quotations, create purchase orders, or make final business decisions.

## 4. Assumptions & Limitations

- Target users are defined based on the fictional business scenario.
- Some required fields, such as quotation follow-up dates, subscription renewal outcomes, or purchase order expected receipt dates, may not be available in the actual dataset.
- Use cases will be refined after data profiling.
- Alert thresholds and priorities will be defined and tested during implementation.
- AI-generated summaries must rely on validated metrics and triggered rules rather than inventing facts or independently determining business risks.