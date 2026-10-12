# Data Sources & Data Dictionary

## 1. Objective

Define the data requirements and planned table structure for the Demand Intelligence & Inventory Risk Engine.

The data model will support demand forecasting, inventory risk assessment, sales quotation analysis, subscription renewal monitoring, and automated risk alerts.

**Data status:** Synthetic dataset planned. The schema below is a preliminary design and will be validated during data generation and profiling.

## 2. Data Sources

The prototype will use five categories of business data.

| Data Source | Purpose | Planned Table(s) |
|---|---|---|
| Historical Sales | Analyze historical demand and forecast future product demand | `sales_transactions` |
| Inventory | Monitor stock availability by product and warehouse | `inventory_snapshots` |
| Purchase Orders | Track incoming stock and replenishment timing | `purchase_orders` |
| Sales Quotations | Analyze quotation status, pipeline value, and follow-up priorities | `quotations`, `quotation_items` |
| Subscription Records | Monitor expiry dates and renewal status | `subscriptions` |

Three supporting master tables will provide consistent product, customer, and warehouse references: `products`, `customers`, and `warehouses`.

All datasets will initially be synthetic and generated for this portfolio project. They do not represent actual company records.

## 3. Planned Data Model

The prototype will use nine relational tables.

### 3.1 Master Tables

#### `products`

| Column | Data Type | Description |
|---|---|---|
| `product_id` | String | Unique product identifier (PK) |
| `product_name` | String | Product name |
| `product_type` | String | Hardware, software license, or subscription |
| `category` | String | Product category |
| `unit_cost` | Decimal | Estimated unit acquisition cost |
| `lead_time_days` | Integer | Expected replenishment lead time in days |
| `reorder_point` | Integer | Initial inventory threshold for replenishment review |

#### `customers`

| Column | Data Type | Description |
|---|---|---|
| `customer_id` | String | Unique customer identifier (PK) |
| `customer_name` | String | Synthetic customer name |
| `customer_segment` | String | Customer segment |
| `industry` | String | Customer industry |

#### `warehouses`

| Column | Data Type | Description |
|---|---|---|
| `warehouse_id` | String | Unique warehouse identifier (PK) |
| `warehouse_name` | String | Warehouse name |
| `location` | String | Warehouse location |

### 3.2 Sales and Inventory Tables

#### `sales_transactions`

**Grain:** One record per sales transaction line.

| Column | Data Type | Description |
|---|---|---|
| `transaction_id` | String | Unique transaction-line identifier (PK) |
| `transaction_date` | Date | Sales transaction date |
| `product_id` | String | Sold product (FK) |
| `warehouse_id` | String | Fulfillment warehouse (FK) |
| `customer_id` | String | Customer (FK) |
| `quantity_sold` | Integer | Quantity sold |
| `unit_price` | Decimal | Selling price per unit |

#### `inventory_snapshots`

**Grain:** One record per product, warehouse, and snapshot date.

| Column | Data Type | Description |
|---|---|---|
| `snapshot_id` | String | Unique snapshot-record identifier (PK) |
| `snapshot_date` | Date | Date of inventory snapshot |
| `product_id` | String | Product (FK) |
| `warehouse_id` | String | Warehouse (FK) |
| `on_hand_qty` | Integer | Physical inventory quantity |
| `reserved_qty` | Integer | Quantity reserved for existing commitments |

### 3.3 Procurement Table

#### `purchase_orders`

**Grain:** One record per purchase order line.

| Column | Data Type | Description |
|---|---|---|
| `po_line_id` | String | Unique purchase order line identifier (PK) |
| `po_id` | String | Purchase order identifier |
| `product_id` | String | Ordered product (FK) |
| `warehouse_id` | String | Destination warehouse (FK) |
| `order_date` | Date | Purchase order date |
| `ordered_qty` | Integer | Quantity ordered |
| `received_qty` | Integer | Quantity received to date |
| `expected_receipt_date` | Date | Expected delivery date |
| `status` | String | Open, partially received, received, or cancelled |

### 3.4 Sales Quotation Tables

#### `quotations`

**Grain:** One record per quotation.

| Column | Data Type | Description |
|---|---|---|
| `quotation_id` | String | Unique quotation identifier (PK) |
| `customer_id` | String | Customer (FK) |
| `quotation_date` | Date | Quotation creation date |
| `valid_until` | Date | Quotation validity end date |
| `status` | String | Open, won, or lost |
| `last_follow_up_date` | Date | Most recent follow-up date, if available |
| `closed_date` | Date | Date the quotation was closed, if applicable |

#### `quotation_items`

**Grain:** One record per product line within a quotation.

| Column | Data Type | Description |
|---|---|---|
| `quotation_item_id` | String | Unique quotation-line identifier (PK) |
| `quotation_id` | String | Parent quotation (FK) |
| `product_id` | String | Quoted product (FK) |
| `quantity` | Integer | Quoted quantity |
| `unit_price` | Decimal | Quoted price per unit |
| `discount_amount` | Decimal | Discount amount for the line |

Quotation value will be calculated from the line items after discounts. A won quotation will not automatically be treated as a completed sale or realized revenue.

### 3.5 Subscription Table

#### `subscriptions`

**Grain:** One record per subscription contract or renewal period.

| Column | Data Type | Description |
|---|---|---|
| `subscription_id` | String | Unique subscription-record identifier (PK) |
| `customer_id` | String | Customer (FK) |
| `product_id` | String | Subscription product (FK) |
| `start_date` | Date | Start of subscription period |
| `expiry_date` | Date | Subscription expiry date |
| `contract_value` | Decimal | Contract value for the recorded period |
| `renewal_status` | String | Upcoming, renewed, not renewed, or pending |
| `renewed_until` | Date | End date of the renewed period, if applicable |

Renewal status and contract value will be used only when their definitions and data consistency have been validated.

## 4. Table Relationships

The planned relationships are:

| Parent Table | Child Table | Relationship |
|---|---|---|
| `products` | `sales_transactions` | One-to-many |
| `products` | `inventory_snapshots` | One-to-many |
| `products` | `purchase_orders` | One-to-many |
| `products` | `quotation_items` | One-to-many |
| `products` | `subscriptions` | One-to-many |
| `customers` | `sales_transactions` | One-to-many |
| `customers` | `quotations` | One-to-many |
| `customers` | `subscriptions` | One-to-many |
| `warehouses` | `sales_transactions` | One-to-many |
| `warehouses` | `inventory_snapshots` | One-to-many |
| `warehouses` | `purchase_orders` | One-to-many |
| `quotations` | `quotation_items` | One-to-many |

Primary keys uniquely identify records. Foreign keys connect transactions to the relevant master records.

## 5. Data Design Assumptions

- The initial dataset will cover 36 months of synthetic business activity.
- Demand analysis will use weekly product-level quantities.
- The initial forecast horizon will be 13 weeks, approximately 90 days.
- Inventory will be tracked across multiple warehouses.
- Inventory snapshots will initially be generated weekly.
- Forecasting will focus on physical products with meaningful historical demand.
- Software licenses and subscriptions will be analyzed primarily through commercial and renewal data.
- Product, customer, and warehouse identifiers will remain consistent across related tables.
- The synthetic data will include realistic business scenarios for testing demand patterns, stock shortages, incoming purchase orders, quotation aging, and upcoming renewals.

These are design assumptions, not validated business facts.

## 6. Data Quality and Validation

The generated dataset will be checked for:

- Missing or duplicate primary keys.
- Invalid foreign-key references.
- Invalid dates and inconsistent date ranges.
- Negative or logically inconsistent quantities.
- Purchase orders where received quantities exceed ordered quantities without an explicit explanation.
- Inventory records where reserved quantities exceed on-hand quantities.
- Quotations with inconsistent status and closure dates.
- Subscription records with inconsistent renewal dates or statuses.
- Missing weekly observations and potential demand distortion caused by stockouts.

The data model and column definitions may be refined if validation reveals structural issues.

## 7. Known Limitations

- Synthetic data does not establish actual company performance or prove business impact.
- Sales transactions represent recorded sales, not necessarily unconstrained customer demand.
- Missing sales records do not automatically indicate zero demand.
- Purchase order quantities are not available inventory until received.
- Won quotations do not confirm realized revenue.
- Upcoming renewals do not guarantee future revenue.
- Initial reorder points and risk thresholds require validation and may be adjusted during testing.

## 8. Expected Output

This task will establish:

1. A documented data model with nine planned tables.
2. Standardized column names and definitions.
3. Primary-key and foreign-key relationships.
4. Data quality rules and validation requirements.
5. A foundation for synthetic data generation and subsequent analysis.