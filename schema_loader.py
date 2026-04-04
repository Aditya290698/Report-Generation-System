"""
schema_loader.py
----------------
Connects to PostgreSQL and builds the schema context string
that gets injected into the LLM prompt.
"""

import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Database connection
# ─────────────────────────────────────────────────────────────────────────────

def get_db_connection():
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing  = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            cursor_factory=psycopg2.extras.RealDictCursor,
            connect_timeout=10,
        )
        logger.info("Database connection established.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error("DB connection failed: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Reporting tables
# ─────────────────────────────────────────────────────────────────────────────

REPORTING_TABLES = [
    # Core POS tables (empty in dev DB, ready for production)
    "orders",
    "order_items",
    "order_payments",
    "order_statuses",
    "order_types",
    # Product catalogue (real data)
    "products",
    "product_variations",
    "product_categories",
    # Customers (real data)
    "customers",
    "payment_methods",
    # Sales transaction tables (real data)
    "mx51_transactions",
    "square_checkouts",
]

TABLE_DESCRIPTIONS = {
    "orders": (
        "Master table for all customer orders. "
        "Revenue column: totalAmountWithTax. Date column: orderedDate. "
        "Filter: isReopened = false. NOTE: Empty in dev database."
    ),
    "order_items": (
        "Line items within each order. One row per product sold. "
        "Columns: quantity, unitPrice, totalPrice, discountWithTax, isCancelled, isRefunded. "
        "Join to orders via orderId, to products via productId. "
        "NOTE: Empty in dev database."
    ),
    "order_payments": (
        "Payment records per order. Columns: paidAmount, tipAmount, surchargeAmount, paymentMethodId. "
        "One order can have multiple rows (split payments). NOTE: Empty in dev database."
    ),
    "order_statuses": (
        "Lookup: order status names (Completed, Cancelled, Refunded etc). "
        "Join to orders via orders.orderStatusId = order_statuses.id."
    ),
    "order_types": (
        "Lookup: order type names (Dine In, Takeaway, Delivery etc). "
        "Join to orders via orders.orderTypeId = order_types.id."
    ),
    "products": (
        "Master product catalogue. 1950+ rows. "
        "Columns: name, code, productCategoryId, isActive, currentStock, costPrice. "
        "Join to order_items via productId."
    ),
    "product_variations": (
        "Variants of a product (Small, Medium, Large etc). 5787 rows. "
        "Columns: name, costPrice, gpMargin, stockQuantity, productId. "
        "Join to products via productId."
    ),
    "product_categories": (
        "Product category names. 123 rows. "
        "Join to products via products.productCategoryId = product_categories.id."
    ),
    "customers": (
        "Registered customers. 42 rows. "
        "Columns: name, email, phone, state, discountType, discountValue, "
        "totalOutstanding, maxCreditLimit, enableAccountPayment, isActive."
    ),
    "payment_methods": (
        "Lookup: payment method names (Cash, Card, Gift Card etc). 9 rows. "
        "Join to order_payments via order_payments.paymentMethodId = payment_methods.id."
    ),
    "mx51_transactions": (
        "EFTPOS/card terminal transactions. 200 rows. PRIMARY sales source. "
        "Key columns: purchaseAmount, finalAmount (use for revenue), tipAmount, "
        "surchargeAmount, refundAmount, resultFinancialStatus, finalisedAt. "
        "resultFinancialStatus values are UPPERCASE: APPROVED, CANCELLED, DECLINED. "
        "Always use UPPER(mx51.resultFinancialStatus) = 'APPROVED' for successful sales. "
        "finalisedAt is timestamp with time zone stored in UTC. "
        "Use range filter: finalisedAt >= 'date 00:00:00+00' AND finalisedAt < 'next_date 00:00:00+00'"
    ),
    "square_checkouts": (
        "Square POS terminal checkouts. 2119 rows. SECONDARY sales source. "
        "Key columns: amount (AUD), status, createdAt. "
        "status values: COMPLETED, CANCELED (uppercase). "
        "Always filter: UPPER(sc.status) = 'COMPLETED'. "
        "createdAt is timestamp with time zone stored in UTC."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Fetch schema from DB
# ─────────────────────────────────────────────────────────────────────────────

def fetch_table_schema(conn, table_name: str) -> list[dict]:
    query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = %s
        ORDER BY ordinal_position;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()
        if not columns:
            logger.warning("Table '%s' has no columns or does not exist.", table_name)
        return [dict(col) for col in columns]
    except psycopg2.Error as e:
        logger.error("Error fetching schema for '%s': %s", table_name, e)
        raise


def fetch_table_relationships(conn) -> list[dict]:
    query = """
        SELECT
            tc.table_name   AS source_table,
            kcu.column_name AS source_column,
            ccu.table_name  AS target_table,
            ccu.column_name AS target_column
        FROM information_schema.table_constraints      AS tc
        JOIN information_schema.key_column_usage       AS kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema    = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema    = 'public'
          AND tc.table_name      = ANY(%s)
        ORDER BY tc.table_name;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (REPORTING_TABLES,))
            rows = cursor.fetchall()
        rels = [dict(r) for r in rows]
        logger.info("Fetched %d foreign key relationships.", len(rels))
        return rels
    except psycopg2.Error as e:
        logger.error("Error fetching relationships: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Build schema context
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_context(conn) -> str:
    logger.info("Building schema context for %d tables...", len(REPORTING_TABLES))

    lines = []
    lines.append("=== DATABASE SCHEMA ===")
    lines.append(
        "PostgreSQL Point of Sale database. "
        "Use ONLY the tables and columns listed below.\n"
    )

    for table_name in REPORTING_TABLES:
        columns = fetch_table_schema(conn, table_name)
        if not columns:
            logger.warning("Skipping '%s' — no columns found.", table_name)
            continue

        lines.append(f"TABLE: {table_name}")
        desc = TABLE_DESCRIPTIONS.get(table_name, "No description.")
        lines.append(f"Description: {desc}")
        lines.append("Columns:")
        for col in columns:
            nullable = "nullable" if col["is_nullable"] == "YES" else "required"
            # Double quotes shown so LLM copies the pattern in queries
            lines.append(f'  - "{col["column_name"]}" ({col["data_type"]}, {nullable})')
        lines.append("")

    # Relationships
    relationships = fetch_table_relationships(conn)
    lines.append("=== TABLE RELATIONSHIPS (Foreign Keys) ===")
    lines.append("Use these for correct JOIN clauses:\n")

    if relationships:
        for rel in relationships:
            lines.append(
                f"  {rel['source_table']}.{rel['source_column']} "
                f"→ {rel['target_table']}.{rel['target_column']}"
            )
    else:
        logger.warning("No FK constraints found — using manual hints.")
        lines.append('  orders."customerId"               → customers.id')
        lines.append('  orders."orderStatusId"            → order_statuses.id')
        lines.append('  orders."orderTypeId"              → order_types.id')
        lines.append('  order_items."orderId"             → orders.id')
        lines.append('  order_items."productId"           → products.id')
        lines.append('  order_items."productVariationId"  → product_variations.id')
        lines.append('  order_payments."orderId"          → orders.id')
        lines.append('  order_payments."paymentMethodId"  → payment_methods.id')
        lines.append('  products."productCategoryId"      → product_categories.id')
        lines.append('  product_variations."productId"    → products.id')

    lines.append("")
    lines.append("=== QUERY RULES ===")
    lines.append("1. Always quote camelCase column names: p.\"productCategoryId\" not p.productCategoryId")
    lines.append("2. Use table aliases in all JOINs.")
    lines.append("3. Revenue = totalAmountWithTax (includes tax).")
    lines.append("4. JOIN order_items to products for product names.")
    lines.append("5. JOIN products to product_categories for category names.")
    lines.append("6. LIMIT rules:")
    lines.append("   - Raw detail queries: LIMIT 100")
    lines.append("   - GROUP BY queries: NO LIMIT")
    lines.append("   - Single aggregate (SUM/COUNT/AVG): NO LIMIT")
    lines.append("   - Top N: LIMIT exactly what user asked")
    lines.append("7. TIMEZONE: All timestamps are UTC. Never use DATE(col) = 'date'.")
    lines.append("   Always use: col >= 'YYYY-MM-DD 00:00:00+00' AND col < 'YYYY-MM-DD+1 00:00:00+00'")
    lines.append("8. mx51 status filter: UPPER(mx51.\"resultFinancialStatus\") = 'APPROVED'")
    lines.append("9. square status filter: UPPER(sc.\"status\") = 'COMPLETED'")

    schema = "\n".join(lines)
    logger.info("Schema context built (%d characters).", len(schema))
    return schema


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Public interface
# ─────────────────────────────────────────────────────────────────────────────

def load_schema() -> tuple[psycopg2.extensions.connection, str]:
    conn   = get_db_connection()
    schema = build_schema_context(conn)
    return conn, schema


if __name__ == "__main__":
    conn, schema = load_schema()
    print(schema)
    print(f"\nSchema length: {len(schema)} characters")
    conn.close()