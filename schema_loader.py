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
        "Master order table — 0 rows in dev database, will have data in production. "
        "ALWAYS generate SQL for orders questions — the query will run and return 0 with a clear message. "
        "Revenue column: totalAmountWithTax (use for all revenue/sales figures). "
        "Date column: orderedDate (timestamp with time zone, UTC). "
        "Filter completed orders: WHERE isReopened = false. "
        "Key columns: id, orderNumber, receiptNumber, customerName, customerId, "
        "orderTypeId, orderStatusId, totalAmountWithTax, totalAmountWithoutTax, "
        "totalTaxAmount, totalDiscountAmountWithTax, discountPercentage, "
        "itemsSubtotal, totalDeliveryAmountWithTax, orderedDate, isReopened, storeId."
    ),
    "order_items": (
        "Line items within each order — 0 rows in dev, will have data in production. "
        "ALWAYS generate SQL for order_items questions. "
        "One row per product sold. Use for top-selling products and category sales. "
        "Key columns: id, orderId, productId, productVariationId, description, "
        "quantity, paidQuantity, unitPrice, totalPrice, discountWithTax, "
        "discountPercentage, totalTax, isCancelled, isRefunded, refundedQuantity, "
        "promotionDiscountWithTax, manualDiscountWithTax, comboDealName, createdAt. "
        "Filter active items: WHERE isCancelled = false AND isRefunded = false. "
        "Join to orders via orderId, to products via productId."
    ),
    "order_payments": (
        "Payment records per order — 0 rows in dev, will have data in production. "
        "ALWAYS generate SQL for payment questions. "
        "One order can have multiple payment rows (split payments). "
        "Key columns: id, orderId, paymentMethodId, paymentStatusId, paidAmount, "
        "tipAmount, surchargeAmount, changeReturn, splitType, paymentDate, "
        "giftCardCode, storeId. "
        "Join to payment_methods via paymentMethodId."
    ),
    "order_statuses": (
        "Lookup table — order status names. 4 rows. Has real data. "
        "Columns: id, name. "
        "Common values: Completed, Cancelled, Refunded, Open (use ILIKE for matching). "
        "Join to orders: orders.orderStatusId = order_statuses.id."
    ),
    "order_types": (
        "Lookup table — order type names. 22 rows. Has real data. "
        "Columns: id, name. "
        "Common values: Dine In, Takeaway, Delivery, Drive Thru. "
        "Join to orders: orders.orderTypeId = order_types.id."
    ),
    "products": (
        "Master product catalogue. 1950+ rows. Has real data. "
        "Key columns: id, name, code, productCategoryId, productSubCategoryId, "
        "isActive, currentStock, currentCostPerUnit, minimumStockAlert, "
        "isStockedItem, sellByWeight, colorCode, createdAt. "
        "Join to order_items via productId. "
        "Join to product_categories via productCategoryId."
    ),
    "product_variations": (
        "Product variants — Small, Medium, Large etc. 5787 rows. Has real data. "
        "Key columns: id, name, code, productId, costPrice, gpMargin, "
        "stockQuantity, isActive, isDefault, barCodeNumber, createdAt. "
        "Join to products via productId."
    ),
    "product_categories": (
        "Product category names. 123 rows. Has real data. "
        "Columns: id, name, isActive, sortOrder, description, colorCode. "
        "Join to products: products.productCategoryId = product_categories.id."
    ),
    "customers": (
        "Registered customer records. 42 rows. Has real data. "
        "Key columns: id, name, email, phone, streetAddress, suburb, state, "
        "postCode, isActive, customerGroupId, discountType, discountValue, "
        "totalOutstanding, maxOrderLimit, maxCreditLimit, "
        "enableAccountPayment, followGroupSettings, createdAt."
    ),
    "payment_methods": (
        "Lookup — payment method names. 9 rows. Has real data. "
        "Columns: id, name. "
        "Join to order_payments: order_payments.paymentMethodId = payment_methods.id."
    ),
    "mx51_transactions": (
        "EFTPOS/card terminal transactions. 200 rows. PRIMARY sales source with real data. "
        "Key columns: id, purchaseAmount, finalAmount (revenue = finalAmount), "
        "tipAmount, surchargeAmount, refundAmount, resultFinancialStatus, "
        "finalisedAt (UTC timestamp), status, storeId, posDeviceId. "
        "resultFinancialStatus is UPPERCASE: APPROVED, CANCELLED, DECLINED. "
        "Always filter: WHERE UPPER(resultFinancialStatus) = 'APPROVED' for successful sales. "
        "Date filter: finalisedAt >= 'date 00:00:00+00' AND finalisedAt < 'next_date 00:00:00+00'."
    ),
    "square_checkouts": (
        "Square POS terminal checkouts. 2119 rows. SECONDARY sales source with real data. "
        "Full columns: id (uuid), storeId (uuid), orderId (uuid nullable), "
        "checkoutId (varchar), amount (numeric — USE FOR REVENUE in AUD), "
        "currency (varchar, always AUD), status (varchar UPPERCASE: COMPLETED/CANCELED), "
        "paymentId (varchar nullable), referenceId (varchar nullable), "
        "orderNumber (varchar nullable), updatedAt (no timezone — do not filter), "
        "createdAt (timestamp with timezone UTC — USE FOR DATE FILTERING). "
        "Always filter successful: WHERE UPPER(status) = 'COMPLETED'. "
        "Use storeId to group by store. "
        "Date filter: createdAt >= 'date 00:00:00+00' AND createdAt < 'next_date 00:00:00+00'."
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


# ─────────────────────────────────────────────────────────────────────────────────
# SECTION 3b — Live row counts
# ─────────────────────────────────────────────────────────────────────────────────

def fetch_row_counts(conn, table_names):
    query = """
        SELECT relname AS table_name, n_live_tup AS row_count
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
          AND relname = ANY(%s);
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (list(table_names),))
            rows = cursor.fetchall()
        counts = {row["table_name"]: int(row["row_count"]) for row in rows}
        for t in table_names:
            counts.setdefault(t, 0)
        logger.info("Row counts: %s", counts)
        return counts
    except psycopg2.Error as e:
        logger.warning("Row count fetch failed (%s) -- defaulting to 0.", e)
        return {t: 0 for t in table_names}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Build schema context
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_context(conn) -> str:
    logger.info("Building schema context for %d tables...", len(REPORTING_TABLES))

    # Fetch live row counts so the LLM knows which tables have data
    row_counts = fetch_row_counts(conn, REPORTING_TABLES)

    lines = []
    lines.append("=== DATABASE SCHEMA ===")
    lines.append(
        "PostgreSQL Point of Sale database. "
        "Use ONLY the tables and columns listed below.\n"
    )

    # Tables are marked [EMPTY] or [X rows] so the LLM knows what has data.
    # IMPORTANT: Even EMPTY tables must have SQL generated for them —
    # the query will run and return 0 rows with a clear message to the user.
    # Never refuse to generate SQL just because a table is empty.
    lines.append("TABLE ROW COUNTS (live from database):")
    for table_name in REPORTING_TABLES:
        count = row_counts.get(table_name, 0)
        status = "EMPTY — 0 rows" if count == 0 else f"{count:,} rows"
        lines.append(f"  {table_name}: {status}")
    lines.append("")

    for table_name in REPORTING_TABLES:
        columns = fetch_table_schema(conn, table_name)
        if not columns:
            logger.warning("Skipping '%s' — no columns found.", table_name)
            continue

        count = row_counts.get(table_name, 0)
        row_status = "EMPTY — 0 rows in this database" if count == 0 else f"{count:,} rows"

        # Table header clearly shows row count
        lines.append(f"TABLE: {table_name}  [{row_status}]")
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
    lines.append("")
    lines.append("=== EMPTY TABLE STRATEGY ===")
    lines.append("Check TABLE ROW COUNTS above before writing any query.")
    lines.append("PRIORITY 1 — Prefer tables with data.")
    lines.append("  If the question can be answered using tables with rows > 0, use those.")
    lines.append("  Example: 'order statuses list' -> query order_statuses (4 rows), no join to orders needed.")
    lines.append("  Example: 'total revenue' -> use mx51_transactions (has data), not orders (0 rows).")
    lines.append("  Example: 'payment methods' -> query payment_methods (9 rows) directly.")
    lines.append("PRIORITY 2 — If empty table is specifically required, generate SQL with a comment.")
    lines.append("  Add: -- NOTE: this table is empty in dev, query will return 0 rows")
    lines.append("  The system shows the user a helpful message about zero results.")
    lines.append("PRIORITY 3 — Use CANNOT_GENERATE only if truly no table can answer the question.")

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