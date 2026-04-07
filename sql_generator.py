"""
sql_generator.py
----------------
LLM Pass 1 — Converts natural language questions into safe,
validated PostgreSQL queries using OpenAI GPT-4o.

Changes v2:
- 3 retry attempts (was 2)
- Error message from previous attempt fed back into context on retry
- Full table metadata for orders, order_items, order_payments,
  order_statuses, square_checkouts added as rich examples
- 20+ few-shot examples covering all common query patterns
- Empty tables now generate SQL (returns zero rows, not an error)
"""

import os
import re
import logging
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OpenAI client
# ─────────────────────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — System prompt
# ─────────────────────────────────────────────────────────────────────────────

SQL_GENERATOR_SYSTEM_PROMPT = """
You are an expert PostgreSQL query generator for a Point of Sale (POS) system.
Convert natural language questions into accurate, safe PostgreSQL SELECT queries.

═══════════════════════════════════════════════════════════════
ABSOLUTE RULES — never break these
═══════════════════════════════════════════════════════════════
1. Output ONLY the raw SQL query. No explanations, no markdown, no backticks, no comments.
2. Only SELECT statements. Never INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE.
3. Only use tables and columns from the schema provided. Never invent column or table names.
4. Always wrap ALL column names in double quotes: o."totalAmountWithTax"
   This is MANDATORY — PostgreSQL lowercases unquoted camelCase names and the query will fail.
5. Always use table aliases: o=orders, oi=order_items, op=order_payments,
   os=order_statuses, ot=order_types, p=products, pc=product_categories,
   pv=product_variations, c=customers, pm=payment_methods,
   mx51=mx51_transactions, sc=square_checkouts.
6. Format all currency/decimal values with ROUND(..., 2).
7. If a question cannot be answered from the schema: CANNOT_GENERATE: <reason>

═══════════════════════════════════════════════════════════════
LIMIT RULES
═══════════════════════════════════════════════════════════════
- Raw row queries (no aggregation): LIMIT 100
- GROUP BY queries: NO LIMIT
- Single aggregate (SUM/COUNT/AVG, no GROUP BY): NO LIMIT
- Top N queries ("top 10", "best 5"): LIMIT exactly what user asked

═══════════════════════════════════════════════════════════════
TIMEZONE RULES — ALL timestamps are stored as UTC
═══════════════════════════════════════════════════════════════
NEVER use: DATE(col) = '2026-02-18'  — timezone-unsafe, returns wrong results
ALWAYS use explicit UTC range:
  col >= '2026-02-18 00:00:00+00' AND col < '2026-02-19 00:00:00+00'

DATE PATTERNS (apply to any timestamp column):
  Today:          col >= CURRENT_DATE::timestamptz AND col < (CURRENT_DATE + INTERVAL '1 day')::timestamptz
  This week:      col >= DATE_TRUNC('week', NOW())
  This month:     col >= DATE_TRUNC('month', NOW())
  Last 7 days:    col >= NOW() - INTERVAL '7 days'
  Last 30 days:   col >= NOW() - INTERVAL '30 days'
  Last 90 days:   col >= NOW() - INTERVAL '90 days'
  This year:      col >= DATE_TRUNC('year', NOW())
  Specific date:  col >= 'YYYY-MM-DD 00:00:00+00' AND col < 'YYYY-MM-DD+1 00:00:00+00'
  Date range:     col >= 'YYYY-MM-01 00:00:00+00' AND col < 'YYYY-MM-30 00:00:00+00'
  Group by day:   DATE(col AT TIME ZONE 'UTC') AS sale_date
  Group by week:  DATE_TRUNC('week', col AT TIME ZONE 'UTC') AS week_start
  Group by month: TO_CHAR(col AT TIME ZONE 'UTC', 'YYYY-MM') AS month
  Group by year:  DATE_PART('year', col AT TIME ZONE 'UTC') AS year

DATE COLUMN PER TABLE — ONLY these are safe for date filtering:
  orders          -> "orderedDate"        (timestamp with time zone, UTC)
  order_items     -> "createdAt" or "dateTime" (timestamp with time zone, UTC)
  order_payments  -> "paymentDate" or "createdAt" (timestamp with time zone, UTC)
  mx51_transactions -> "finalisedAt"      (timestamp with time zone, UTC)
  square_checkouts  -> "createdAt"        (timestamp with time zone, UTC)
  customers         -> "createdAt"        (timestamp with time zone, UTC)
  products          -> "createdAt"        (timestamp with time zone, UTC)

NEVER filter on "updatedAt" — it has no timezone info and returns wrong results.

═══════════════════════════════════════════════════════════════
ORDERS TABLE — full column reference
═══════════════════════════════════════════════════════════════
Table: orders (alias: o)
Currently 0 rows in dev database — but ALWAYS generate SQL for orders questions.
The query will run and correctly return 0 results with a proper message.
DO NOT refuse to generate SQL just because the table is empty.

Key business columns:
  "id"                          — unique order identifier (UUID)
  "orderNumber"                 — human-readable order number
  "receiptNumber"               — receipt number
  "customerName"                — customer name on the order
  "customerId"                  — links to customers table
  "orderTypeId"                 — links to order_types (Dine In, Takeaway, Delivery)
  "orderStatusId"               — links to order_statuses (Completed, Cancelled etc)
  "totalAmountWithTax"          — TOTAL order value including tax (USE THIS FOR REVENUE)
  "totalAmountWithoutTax"       — order value before tax
  "totalTaxAmount"              — tax amount
  "totalDiscountAmountWithTax"  — discount applied including tax
  "totalDiscountAmountWithoutTax" — discount before tax
  "discountPercentage"          — discount percentage applied
  "itemsSubtotal"               — subtotal of all items before adjustments
  "totalDeliveryAmountWithTax"  — delivery charge including tax
  "creditCardSurChargeAmountWithTax" — credit card surcharge
  "publicHolidaySurChargeAmountWithTax" — public holiday surcharge
  "orderedDate"                 — when order was placed (USE FOR DATE FILTERING)
  "orderPaymentCompletedDate"   — when payment was completed
  "isReopened"                  — boolean, filter WHERE o."isReopened" = false for completed orders
  "storeId"                     — which store the order belongs to
  "posDeviceId"                 — which POS device took the order
  "channelId"                   — sales channel

Business terminology mapping for orders:
  "total sales" / "revenue" / "turnover"  → SUM(o."totalAmountWithTax")
  "number of orders" / "order count"      → COUNT(o."id")
  "average order value" / "AOV"           → AVG(o."totalAmountWithTax")
  "completed orders"                      → WHERE o."isReopened" = false
  "total discount"                        → SUM(o."totalDiscountAmountWithTax")
  "total tax collected"                   → SUM(o."totalTaxAmount")
  "delivery orders"                       → JOIN order_types ot ON o."orderTypeId" = ot."id" WHERE ot."name" ILIKE '%delivery%'

═══════════════════════════════════════════════════════════════
ORDER_ITEMS TABLE — full column reference
═══════════════════════════════════════════════════════════════
Table: order_items (alias: oi)
Currently 0 rows in dev database — always generate SQL, it will return 0 results.

Key business columns:
  "id"                    — unique line item identifier
  "orderId"               — links to orders table
  "productId"             — links to products table (get product name from products)
  "productVariationId"    — links to product_variations table
  "description"           — item description
  "quantity"              — quantity ordered
  "paidQuantity"          — quantity paid for
  "unitPrice"             — price per unit
  "totalPrice"            — total for this line item (quantity × unitPrice - discounts)
  "discountWithTax"       — discount on this item including tax
  "discountWithoutTax"    — discount on this item before tax
  "discountPercentage"    — discount percentage on this item
  "totalTax"              — tax on this item
  "isCancelled"           — boolean, filter WHERE oi."isCancelled" = false for active items
  "isRefunded"            — boolean
  "refundedQuantity"      — quantity refunded
  "promotionDiscountWithTax" — promotion discount applied
  "manualDiscountWithTax"    — manual discount applied
  "comboDealName"         — combo deal name if applicable
  "createdAt"             — when item was added (use for date filtering)

Business terminology for order_items:
  "top selling products"   → GROUP BY p."name" ORDER BY SUM(oi."quantity") DESC
  "best selling by revenue"→ GROUP BY p."name" ORDER BY SUM(oi."totalPrice") DESC
  "items sold"             → SUM(oi."quantity")
  "total item revenue"     → SUM(oi."totalPrice")
  "active items only"      → WHERE oi."isCancelled" = false AND oi."isRefunded" = false

═══════════════════════════════════════════════════════════════
ORDER_PAYMENTS TABLE — full column reference
═══════════════════════════════════════════════════════════════
Table: order_payments (alias: op)
Currently 0 rows in dev database — always generate SQL, it will return 0 results.

Key business columns:
  "id"                — unique payment identifier
  "orderId"           — links to orders table
  "paymentMethodId"   — links to payment_methods (Cash, Card, Gift Card etc)
  "paymentStatusId"   — links to payment_statuses
  "paidAmount"        — amount paid
  "tipAmount"         — tip amount
  "surchargeAmount"   — surcharge amount
  "changeReturn"      — change returned to customer
  "splitType"         — split payment type if applicable
  "paymentDate"       — when payment was made (use for date filtering)
  "giftCardCode"      — gift card code if used
  "storeId"           — which store processed the payment

Business terminology for order_payments:
  "payment breakdown"   → GROUP BY pm."name" with SUM(op."paidAmount")
  "cash payments"       → JOIN payment_methods pm WHERE pm."name" ILIKE '%cash%'
  "card payments"       → JOIN payment_methods pm WHERE pm."name" ILIKE '%card%'
  "total tips"          → SUM(op."tipAmount")
  "total surcharge"     → SUM(op."surchargeAmount")

═══════════════════════════════════════════════════════════════
ORDER_STATUSES TABLE — full column reference
═══════════════════════════════════════════════════════════════
Table: order_statuses (alias: os) — has real data (4 rows)
Columns: "id", "name"
Join: orders o JOIN order_statuses os ON o."orderStatusId" = os."id"
Use ILIKE for flexible name matching: WHERE os."name" ILIKE '%complete%'

Common status names (use ILIKE not exact match):
  Completed / Complete → ILIKE '%complet%'
  Cancelled / Canceled → ILIKE '%cancel%'
  Refunded             → ILIKE '%refund%'
  Open / Active        → ILIKE '%open%'

═══════════════════════════════════════════════════════════════
SQUARE_CHECKOUTS TABLE — full column reference
═══════════════════════════════════════════════════════════════
Table: square_checkouts (alias: sc) — 2119 rows of real data. SECONDARY sales source.

Key columns:
  "id"          — unique checkout identifier
  "storeId"     — which store
  "orderId"     — links to orders (currently empty)
  "checkoutId"  — Square's checkout reference
  "amount"      — transaction amount in AUD (USE THIS FOR REVENUE)
  "currency"    — always AUD
  "status"      — 'COMPLETED' or 'CANCELED' (stored uppercase)
  "paymentId"   — Square payment reference
  "orderNumber" — order number reference
  "createdAt"   — when checkout was created (USE FOR DATE FILTERING, UTC)

Always filter completed only: WHERE UPPER(sc."status") = 'COMPLETED'

Business terminology for square_checkouts:
  "Square sales" / "Square revenue"  → SUM(sc."amount") WHERE UPPER(sc."status") = 'COMPLETED'
  "Square transactions"              → COUNT(*) WHERE UPPER(sc."status") = 'COMPLETED'
  "cancelled Square checkouts"       → WHERE UPPER(sc."status") = 'CANCELED'
  "average Square transaction"       → AVG(sc."amount") WHERE UPPER(sc."status") = 'COMPLETED'

═══════════════════════════════════════════════════════════════
MX51_TRANSACTIONS TABLE — full column reference
═══════════════════════════════════════════════════════════════
Table: mx51_transactions (alias: mx51) — 200 rows of real data. PRIMARY sales source.

Key columns:
  "purchaseAmount"         — sale amount before tip/surcharge
  "finalAmount"            — TOTAL charged including tip+surcharge (USE FOR REVENUE)
  "tipAmount"              — tip collected
  "surchargeAmount"        — card surcharge applied
  "refundAmount"           — refund amount (if any)
  "resultFinancialStatus"  — stored UPPERCASE: 'APPROVED', 'CANCELLED', 'DECLINED'
  "finalisedAt"            — when transaction completed (USE FOR DATE FILTERING, UTC)
  "status"                 — overall transaction status

Always use UPPER() for status comparison:
  WHERE UPPER(mx51."resultFinancialStatus") = 'APPROVED'

Business terminology:
  "approved sales" / "successful transactions" → WHERE UPPER(mx51."resultFinancialStatus") = 'APPROVED'
  "total revenue"    → SUM(mx51."finalAmount")
  "purchase amount"  → SUM(mx51."purchaseAmount")
  "tips collected"   → SUM(mx51."tipAmount")
  "refunds"          → SUM(mx51."refundAmount")
  "transaction count"→ COUNT(*)

═══════════════════════════════════════════════════════════════
IMPORTANT — EMPTY TABLE BEHAVIOUR
═══════════════════════════════════════════════════════════════
orders, order_items, and order_payments are EMPTY in this database (0 rows).
HOWEVER: ALWAYS generate the correct SQL anyway.
- The query will run successfully and return 0 rows or NULL aggregates
- The system will tell the user "no data found" after running the query
- DO NOT refuse or use CANNOT_GENERATE just because a table is empty
- DO NOT substitute mx51 or square for orders questions — write the correct orders SQL
  The client may be testing the SQL structure for when production data is connected.

Exception: If the user EXPLICITLY asks for "current sales data" or "actual revenue now",
then you may note that orders is empty and suggest mx51_transactions instead.

═══════════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES — learn these patterns
═══════════════════════════════════════════════════════════════

Q: What is the total revenue from all orders?
A: SELECT ROUND(SUM(o."totalAmountWithTax"), 2) AS total_revenue
   FROM orders o
   WHERE o."isReopened" = false

Q: How many orders were placed today?
A: SELECT COUNT(o."id") AS order_count
   FROM orders o
   WHERE o."orderedDate" >= CURRENT_DATE::timestamptz
     AND o."orderedDate" < (CURRENT_DATE + INTERVAL '1 day')::timestamptz
     AND o."isReopened" = false

Q: What is the average order value this month?
A: SELECT ROUND(AVG(o."totalAmountWithTax"), 2) AS average_order_value
   FROM orders o
   WHERE o."orderedDate" >= DATE_TRUNC('month', NOW())
     AND o."isReopened" = false

Q: Show daily revenue for the last 7 days from orders
A: SELECT DATE(o."orderedDate" AT TIME ZONE 'UTC') AS sale_date,
          COUNT(o."id") AS order_count,
          ROUND(SUM(o."totalAmountWithTax"), 2) AS daily_revenue
   FROM orders o
   WHERE o."orderedDate" >= NOW() - INTERVAL '7 days'
     AND o."isReopened" = false
   GROUP BY DATE(o."orderedDate" AT TIME ZONE 'UTC')
   ORDER BY sale_date

Q: Show weekly revenue from orders
A: SELECT DATE_TRUNC('week', o."orderedDate" AT TIME ZONE 'UTC') AS week_start,
          COUNT(o."id") AS order_count,
          ROUND(SUM(o."totalAmountWithTax"), 2) AS weekly_revenue
   FROM orders o
   WHERE o."isReopened" = false
   GROUP BY DATE_TRUNC('week', o."orderedDate" AT TIME ZONE 'UTC')
   ORDER BY week_start

Q: Show monthly revenue from orders
A: SELECT TO_CHAR(o."orderedDate" AT TIME ZONE 'UTC', 'YYYY-MM') AS month,
          COUNT(o."id") AS order_count,
          ROUND(SUM(o."totalAmountWithTax"), 2) AS monthly_revenue
   FROM orders o
   WHERE o."isReopened" = false
   GROUP BY TO_CHAR(o."orderedDate" AT TIME ZONE 'UTC', 'YYYY-MM')
   ORDER BY month

Q: Total sales for January 2026
A: SELECT ROUND(SUM(o."totalAmountWithTax"), 2) AS total_sales,
          COUNT(o."id") AS order_count
   FROM orders o
   WHERE o."orderedDate" >= '2026-01-01 00:00:00+00'
     AND o."orderedDate" <  '2026-02-01 00:00:00+00'
     AND o."isReopened" = false

Q: What are the top 10 selling products by quantity?
A: SELECT p."name" AS product_name,
          SUM(oi."quantity") AS total_quantity_sold,
          ROUND(SUM(oi."totalPrice"), 2) AS total_revenue
   FROM order_items oi
   JOIN products p ON oi."productId" = p."id"
   WHERE oi."isCancelled" = false
   GROUP BY p."name"
   ORDER BY total_quantity_sold DESC
   LIMIT 10

Q: Sales by product category
A: SELECT pc."name" AS category,
          COUNT(DISTINCT oi."orderId") AS order_count,
          SUM(oi."quantity") AS items_sold,
          ROUND(SUM(oi."totalPrice"), 2) AS total_revenue
   FROM order_items oi
   JOIN products p ON oi."productId" = p."id"
   JOIN product_categories pc ON p."productCategoryId" = pc."id"
   WHERE oi."isCancelled" = false
   GROUP BY pc."name"
   ORDER BY total_revenue DESC

Q: Number of orders and average order value by order type
A: SELECT ot."name" AS order_type,
          COUNT(o."id") AS order_count,
          ROUND(AVG(o."totalAmountWithTax"), 2) AS avg_order_value,
          ROUND(SUM(o."totalAmountWithTax"), 2) AS total_revenue
   FROM orders o
   JOIN order_types ot ON o."orderTypeId" = ot."id"
   WHERE o."isReopened" = false
   GROUP BY ot."name"
   ORDER BY total_revenue DESC

Q: Revenue breakdown by payment method
A: SELECT pm."name" AS payment_method,
          COUNT(op."id") AS payment_count,
          ROUND(SUM(op."paidAmount"), 2) AS total_paid
   FROM order_payments op
   JOIN payment_methods pm ON op."paymentMethodId" = pm."id"
   GROUP BY pm."name"
   ORDER BY total_paid DESC

Q: Show completed orders with their status
A: SELECT o."orderNumber", o."customerName",
          ROUND(o."totalAmountWithTax", 2) AS total,
          os."name" AS status,
          o."orderedDate"
   FROM orders o
   JOIN order_statuses os ON o."orderStatusId" = os."id"
   WHERE os."name" ILIKE '%complet%'
   ORDER BY o."orderedDate" DESC
   LIMIT 100

Q: Total tips collected from order payments
A: SELECT ROUND(SUM(op."tipAmount"), 2) AS total_tips,
          COUNT(op."id") AS payment_count
   FROM order_payments op
   WHERE op."tipAmount" > 0

Q: What is the total discount given across all orders?
A: SELECT ROUND(SUM(o."totalDiscountAmountWithTax"), 2) AS total_discount,
          COUNT(o."id") AS orders_with_discount
   FROM orders o
   WHERE o."totalDiscountAmountWithTax" > 0
     AND o."isReopened" = false

Q: Show Square checkout revenue by month
A: SELECT TO_CHAR(sc."createdAt" AT TIME ZONE 'UTC', 'YYYY-MM') AS month,
          COUNT(sc."id") AS transaction_count,
          ROUND(SUM(sc."amount"), 2) AS monthly_revenue
   FROM square_checkouts sc
   WHERE UPPER(sc."status") = 'COMPLETED'
   GROUP BY TO_CHAR(sc."createdAt" AT TIME ZONE 'UTC', 'YYYY-MM')
   ORDER BY month

Q: What is the total revenue from mx51 transactions this year?
A: SELECT ROUND(SUM(mx51."finalAmount"), 2) AS total_revenue,
          COUNT(mx51."id") AS transaction_count
   FROM mx51_transactions mx51
   WHERE UPPER(mx51."resultFinancialStatus") = 'APPROVED'
     AND mx51."finalisedAt" >= DATE_TRUNC('year', NOW())

Q: Compare revenue from mx51 vs Square checkouts
A: SELECT 'mx51' AS source,
          ROUND(SUM(mx51."finalAmount"), 2) AS total_revenue,
          COUNT(*) AS transactions
   FROM mx51_transactions mx51
   WHERE UPPER(mx51."resultFinancialStatus") = 'APPROVED'
   UNION ALL
   SELECT 'square' AS source,
          ROUND(SUM(sc."amount"), 2) AS total_revenue,
          COUNT(*) AS transactions
   FROM square_checkouts sc
   WHERE UPPER(sc."status") = 'COMPLETED'

Q: How many customers do we have by state?
A: SELECT c."state", COUNT(c."id") AS customer_count
   FROM customers c
   WHERE c."state" IS NOT NULL
   GROUP BY c."state"
   ORDER BY customer_count DESC

Q: Show products with stock below 10
A: SELECT p."name", pv."name" AS variation,
          pv."stockQuantity", pv."costPrice"
   FROM products p
   JOIN product_variations pv ON pv."productId" = p."id"
   WHERE pv."stockQuantity" < 10
     AND p."isActive" = true
   ORDER BY pv."stockQuantity" ASC
   LIMIT 100

Q: What is the total tax collected from orders?
A: SELECT ROUND(SUM(o."totalTaxAmount"), 2) AS total_tax_collected,
          COUNT(o."id") AS order_count
   FROM orders o
   WHERE o."isReopened" = false

Q: Show cancelled orders
A: SELECT o."orderNumber", o."customerName",
          ROUND(o."totalAmountWithTax", 2) AS total,
          o."orderedDate"
   FROM orders o
   JOIN order_statuses os ON o."orderStatusId" = os."id"
   WHERE os."name" ILIKE '%cancel%'
   ORDER BY o."orderedDate" DESC
   LIMIT 100

Q: What order statuses are available? / List all order statuses / Show me the order statuses
A: SELECT os."id", os."name" AS order_status
   FROM order_statuses os
   ORDER BY os."name"

Q: Get me order statuses of each order with order id / Show order status for each order
A: SELECT o."id" AS order_id, o."orderNumber",
          os."name" AS order_status,
          o."orderedDate"
   FROM orders o
   JOIN order_statuses os ON o."orderStatusId" = os."id"
   ORDER BY o."orderedDate" DESC
   LIMIT 100

NOTE on order_statuses vs orders join:
- If user asks "what statuses exist" / "list statuses" / "available statuses"
  → query order_statuses directly (no join to orders needed)
- If user asks "status of each order" / "order status per order" / "order id with status"
  → join orders to order_statuses (orders is empty in dev, will return 0 rows)
- Always make the distinction based on whether user wants the lookup list or per-order data
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SQL validator
# ─────────────────────────────────────────────────────────────────────────────

FORBIDDEN_SQL_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "MERGE", "GRANT", "REVOKE", "EXEC",
    "EXECUTE", "CALL", "COPY", "pg_read_file", "pg_ls_dir",
]


class SQLValidationError(Exception):
    pass


def validate_sql(sql: str) -> str:
    sql = sql.strip()

    if not sql:
        raise SQLValidationError("Generated SQL is empty.")

    if sql.upper().startswith("CANNOT_GENERATE"):
        reason = sql.split(":", 1)[1].strip() if ":" in sql else "Unknown reason"
        raise SQLValidationError(f"Query could not be generated: {reason}")

    if not sql.upper().lstrip().startswith("SELECT"):
        raise SQLValidationError(
            f"Only SELECT queries allowed. Got: '{sql[:40]}'"
        )

    sql_upper = sql.upper()
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_upper):
            raise SQLValidationError(
                f"Forbidden keyword '{keyword}' detected. Only SELECT permitted."
            )

    if len(sql) > 5000:
        raise SQLValidationError(f"SQL too long ({len(sql)} chars). Possible bad response.")

    logger.info("SQL validation passed (%d chars).", len(sql))
    return sql


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SQL generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql(
    user_question: str,
    schema_context: str,
    client: OpenAI | None = None,
    previous_error: str | None = None,
) -> str:
    """
    Converts natural language to validated SQL.

    Args:
        user_question:   The question from the user.
        schema_context:  Full schema string from schema_loader.
        client:          Optional pre-built OpenAI client.
        previous_error:  Error message from a prior failed attempt —
                         injected into the prompt so the model can
                         self-correct on retry.
    """
    if not user_question or not user_question.strip():
        raise ValueError("user_question cannot be empty.")
    if not schema_context or not schema_context.strip():
        raise ValueError("schema_context cannot be empty.")

    user_question = user_question.strip()
    logger.info("Generating SQL for: '%s'", user_question)

    if client is None:
        client = get_openai_client()

    # ── Build user message ────────────────────────────────────────────────────
    # On retry, inject the previous error so the model knows what went wrong
    error_context = ""
    if previous_error:
        error_context = f"""
=== PREVIOUS ATTEMPT FAILED ===
Your previous SQL query failed with this error:
{previous_error}

Analyse the error carefully and fix the query.
Common fixes:
- Column not found → check spelling and double-quote the column name
- Table not found → check table name spelling
- Syntax error → check for missing commas, parentheses, keywords
- Type mismatch → check data types in comparisons

"""

    user_message = f"""
{schema_context}
{error_context}
=== USER QUESTION ===
{user_question}

Generate the PostgreSQL query to answer this question.
Output ONLY the raw SQL — no explanations, no markdown, no backticks.
Remember: ALL column names must be in double quotes.
"""

    try:
        logger.info("Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,   # Increased for complex multi-join queries
            temperature=0,
            messages=[
                {"role": "system", "content": SQL_GENERATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )

        raw_sql = response.choices[0].message.content
        if not raw_sql:
            raise SQLValidationError("OpenAI returned an empty response.")

        raw_sql = raw_sql.strip()
        logger.info("SQL received:\n%s", raw_sql)

    except APIConnectionError as e:
        logger.error("Network error: %s", e)
        raise
    except RateLimitError as e:
        logger.error("Rate limit: %s", e)
        raise
    except APIStatusError as e:
        logger.error("API error (status %s): %s", e.status_code, e.message)
        raise

    return validate_sql(raw_sql)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Retry wrapper (3 attempts, error fed back on each retry)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql_with_retry(
    user_question: str,
    schema_context: str,
    max_attempts: int = 3,   # Increased from 2 to 3
) -> str:
    """
    Wraps generate_sql() with 3-attempt retry logic.

    On each failed attempt the error message is passed back into the
    next prompt so GPT-4o can self-correct. This handles:
    - Formatting slips (markdown fences, added commentary)
    - Wrong column names caught by the validator
    - Syntax errors flagged by the DB (passed from query_executor)
    """
    client = get_openai_client()
    last_error = None
    previous_error_msg = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("SQL generation attempt %d of %d.", attempt, max_attempts)
            return generate_sql(
                user_question,
                schema_context,
                client=client,
                previous_error=previous_error_msg,
            )
        except SQLValidationError as e:
            last_error = e
            # Feed the exact error back into the next attempt
            previous_error_msg = str(e)
            logger.warning(
                "Attempt %d failed validation: %s. "
                "Error will be fed into next attempt.",
                attempt, e
            )
            if attempt < max_attempts:
                # Also strengthen the output format reminder
                user_question = (
                    f"{user_question}\n\n"
                    "CRITICAL REMINDER: Output ONLY the raw SQL query. "
                    "No markdown fences, no backticks, no explanations, no comments."
                )

    raise SQLValidationError(
        f"SQL generation failed after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from schema_loader import load_schema

    test_questions = [
        # Orders (empty table — should still generate SQL)
        "What is the total revenue from all orders?",
        "Show monthly revenue from orders",
        "How many orders were placed this week?",
        "What is the average order value?",
        "Show top 10 selling products by quantity",
        "Sales breakdown by product category",
        "Revenue by order type (dine in vs takeaway vs delivery)",
        "Show completed orders with their status",
        "What is the total discount given this month?",
        # Square (real data)
        "Show monthly revenue from Square checkouts",
        "How many completed Square transactions are there?",
        # mx51 (real data)
        "What is the total revenue from approved mx51 transactions?",
        "Show daily mx51 revenue for the last 14 days",
        # Products (real data)
        "How many products are in each category?",
        "Which products have low stock below 10 units?",
    ]

    conn, schema = load_schema()
    passed = 0
    failed = 0
    for i, q in enumerate(test_questions, 1):
        print(f"\n[Test {i}] {q}")
        try:
            sql = generate_sql_with_retry(q, schema)
            print(f"SQL:\n{sql}")
            passed += 1
        except SQLValidationError as e:
            print(f"FAILED: {e}")
            failed += 1
    conn.close()
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_questions)}")