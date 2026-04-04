import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Logging setup ────────────────────────────────────────────────────────────
# Using Python's built-in logging instead of print() so log levels
# (DEBUG, INFO, WARNING, ERROR) can be controlled without touching code.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Load environment variables ────────────────────────────────────────────────
# Reads DB credentials from the .env file so they're never hardcoded
# in source code. Always load at module level so it runs once on import.
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Database connection
# ─────────────────────────────────────────────────────────────────────────────

def get_db_connection():
    """
    Creates and returns a PostgreSQL connection using credentials
    from environment variables.

    Returns:
        psycopg2.connection: An active database connection.

    Raises:
        EnvironmentError: If any required env variable is missing.
        psycopg2.OperationalError: If the database is unreachable.
    """

    # Collect all required env vars and check they exist before connecting.
    # Failing early with a clear message is much better than a cryptic
    # psycopg2 error later.
    required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file."
        )

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            # cursor_factory returns rows as dicts instead of plain tuples,
            # making downstream code much more readable: row["column_name"]
            # instead of row[0].
            cursor_factory=psycopg2.extras.RealDictCursor,
            # Connection-level timeout: fail fast if DB is unreachable
            # rather than hanging indefinitely.
            connect_timeout=10,
        )
        logger.info("Database connection established successfully.")
        return conn

    except psycopg2.OperationalError as e:
        logger.error("Failed to connect to the database: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — The 10 core tables we care about
# ─────────────────────────────────────────────────────────────────────────────

# These are the only tables the LLM is allowed to query.
# Keeping this list explicit prevents the LLM from accidentally touching
# unrelated tables (e.g. internal config tables, audit logs, etc.)
REPORTING_TABLES = [
    # Core POS tables (currently empty in dev DB, ready for production)
    "orders",
    "order_items",
    "order_payments",
    "order_statuses",
    "order_types",
    # Product catalogue (1950+ rows of real data)
    "products",
    "product_variations",
    "product_categories",
    # Customers (42 rows of real data)
    "customers",
    "payment_methods",
    # Sales transaction tables (real data available now)
    "mx51_transactions",
    "square_checkouts",
]

# Human-friendly descriptions for each table.
# These are injected into the LLM prompt so it understands the *purpose*
# of each table, not just its structure. This dramatically improves
# the quality of generated SQL.
TABLE_DESCRIPTIONS = {
    "orders": (
        "The master table for all customer orders. Contains order totals, "
        "dates, status, customer details, and links to order type and channel. "
        "Use orderedDate for date-range filtering and totalAmountWithTax for revenue. "
        "NOTE: Currently empty in the development database."
    ),
    "order_items": (
        "Line items within each order. One row per product sold. "
        "Contains quantity, unitPrice, totalPrice, and discount fields. "
        "Join to orders via orderId and to products via productId. "
        "NOTE: Currently empty in the development database."
    ),
    "order_payments": (
        "Payment records for each order. Contains paidAmount, tipAmount, "
        "surchargeAmount, and the paymentMethodId linking to payment_methods. "
        "One order can have multiple payment rows (e.g. split payments). "
        "NOTE: Currently empty in the development database."
    ),
    "order_statuses": (
        "Lookup table for order status names (e.g. Completed, Cancelled, Refunded). "
        "Join to orders via orders.orderStatusId = order_statuses.id."
    ),
    "order_types": (
        "Lookup table for order type names (e.g. Dine In, Takeaway, Delivery). "
        "Join to orders via orders.orderTypeId = order_types.id."
    ),
    "products": (
        "Master product catalogue. Contains product name, code, category, "
        "and stock information. Join to order_items via productId. "
        "Has 1950+ rows of real product data."
    ),
    "product_variations": (
        "Variants of a product (e.g. Small, Medium, Large). "
        "Contains costPrice, gpMargin, stockQuantity. "
        "Join to products via productId and to order_items via productVariationId. "
        "Has 5787 rows of real data."
    ),
    "product_categories": (
        "Product category names (e.g. Coffee, Food, Beverages). "
        "Join to products via products.productCategoryId = product_categories.id. "
        "Has 123 rows of real data."
    ),
    "customers": (
        "Registered customer records. Contains name, email, phone, "
        "discount settings, and account credit limits. "
        "Join to orders via orders.customerId = customers.id. "
        "Has 42 rows of real data."
    ),
    "payment_methods": (
        "Lookup table for payment method names (e.g. Cash, Card, Gift Card). "
        "Join to order_payments via order_payments.paymentMethodId = payment_methods.id. "
        "Has 9 rows of real data."
    ),
    "mx51_transactions": (
        "EFTPOS/card terminal transaction records from the mx51 payment device. "
        "This is the PRIMARY source for real sales numbers in this database. "
        "Key columns for sales reporting: "
        "purchaseAmount = the original sale amount before tip/surcharge. "
        "finalAmount = total charged including tip and surcharge (use this for revenue). "
        "tipAmount = tip collected on the transaction. "
        "surchargeAmount = card surcharge applied. "
        "refundAmount = amount refunded (if applicable). "
        "resultFinancialStatus = filter to approved for successful sales only. "
        "finalisedAt = the datetime the transaction completed, use for date filtering. "
        "Has 200 rows of real transaction data. "
        "IMPORTANT: Always filter WHERE mx51.resultFinancialStatus = approved "
        "to exclude failed or cancelled transactions."
    ),
    "square_checkouts": (
        "Square POS terminal checkout records. Secondary source for sales data. "
        "Key columns: "
        "amount = transaction amount in AUD. "
        "status = filter to COMPLETED for successful sales only. "
        "createdAt = transaction date, use for date filtering. "
        "currency = always AUD. "
        "orderId = links to orders table (currently empty in dev DB). "
        "Has 2119 rows but many are CANCELED. "
        "IMPORTANT: Always filter WHERE sc.status = COMPLETED "
        "to exclude cancelled transactions."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Fetch schema from DB
# ─────────────────────────────────────────────────────────────────────────────

def fetch_table_schema(conn, table_name: str) -> list[dict]:
    """
    Fetches column metadata for a single table from PostgreSQL's
    information_schema — the standard system view that holds column
    definitions for every table in the database.

    Args:
        conn: Active psycopg2 database connection.
        table_name: Name of the table to inspect.

    Returns:
        List of dicts, one per column, with keys:
        column_name, data_type, is_nullable, column_default.

    Raises:
        psycopg2.Error: On any database query failure.
    """
    query = """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = %s
        ORDER BY ordinal_position;
    """

    try:
        with conn.cursor() as cursor:
            # Using parameterised query (%s) — never use f-strings or
            # string concatenation with SQL. Parameterised queries are
            # immune to SQL injection.
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()

        if not columns:
            logger.warning("Table '%s' has no columns or does not exist.", table_name)

        return [dict(col) for col in columns]

    except psycopg2.Error as e:
        logger.error("Error fetching schema for table '%s': %s", table_name, e)
        raise


def fetch_table_relationships(conn) -> list[dict]:
    """
    Fetches all foreign key relationships between our 10 reporting tables.
    These relationships are included in the schema context so the LLM
    knows exactly how to JOIN tables correctly.

    Args:
        conn: Active psycopg2 connection.

    Returns:
        List of dicts with: source_table, source_column,
        target_table, target_column.
    """
    query = """
        SELECT
            tc.table_name        AS source_table,
            kcu.column_name      AS source_column,
            ccu.table_name       AS target_table,
            ccu.column_name      AS target_column
        FROM information_schema.table_constraints      AS tc
        JOIN information_schema.key_column_usage       AS kcu
          ON tc.constraint_name  = kcu.constraint_name
         AND tc.table_schema     = kcu.table_schema
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

        relationships = [dict(row) for row in rows]
        logger.info("Fetched %d foreign key relationships.", len(relationships))
        return relationships

    except psycopg2.Error as e:
        logger.error("Error fetching relationships: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Build the schema context string
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_context(conn) -> str:
    """
    Builds the complete schema context string that gets injected into
    the LLM system prompt.

    The context includes:
    - A description of each table's purpose
    - All column names with data types and nullable flags
    - Foreign key relationships for correct JOINs

    This is the single most important input to the LLM. A well-structured
    schema context produces accurate SQL; a poor one produces hallucinated
    column names and wrong JOINs.

    Args:
        conn: Active psycopg2 connection.

    Returns:
        A formatted multi-line string ready to be inserted into a prompt.
    """
    logger.info("Building schema context for %d tables...", len(REPORTING_TABLES))

    lines = []
    lines.append("=== DATABASE SCHEMA ===")
    lines.append(
        "You have access to a Point of Sale (POS) PostgreSQL database. "
        "Use ONLY the tables and columns listed below. "
        "Do not reference any table or column not listed here.\n"
    )

    # ── Per-table schema block ────────────────────────────────────────────────
    for table_name in REPORTING_TABLES:
        columns = fetch_table_schema(conn, table_name)

        if not columns:
            # Skip tables that don't exist yet (e.g. lookup tables not yet created)
            logger.warning("Skipping table '%s' — no columns found.", table_name)
            continue

        # Table header
        lines.append(f"TABLE: {table_name}")

        # Human description so the LLM understands the table's purpose
        description = TABLE_DESCRIPTIONS.get(table_name, "No description available.")
        lines.append(f"Description: {description}")

        # Column definitions — one per line
        lines.append("Columns:")
        for col in columns:
            nullable = "nullable" if col["is_nullable"] == "YES" else "required"
            # Column names are shown in double quotes to remind the LLM
            # that camelCase names must always be quoted in queries.
            lines.append(f'  - "{col["column_name"]}" ({col["data_type"]}, {nullable})')

        lines.append("")  # blank line between tables

    # ── Relationships block ───────────────────────────────────────────────────
    relationships = fetch_table_relationships(conn)

    lines.append("=== TABLE RELATIONSHIPS (Foreign Keys) ===")
    lines.append("Use these to write correct JOIN clauses:\n")

    if relationships:
        for rel in relationships:
            lines.append(
                f"  {rel['source_table']}.{rel['source_column']} "
                f"→ {rel['target_table']}.{rel['target_column']}"
            )
    else:
        # Some databases don't enforce FK constraints even if the
        # logical relationships exist. We add manual hints as fallback.
        logger.warning("No FK constraints found. Adding manual relationship hints.")
        lines.append('  orders."customerId"              → customers.id')
        lines.append('  orders."orderStatusId"           → order_statuses.id')
        lines.append('  orders."orderTypeId"             → order_types.id')
        lines.append('  order_items."orderId"            → orders.id')
        lines.append('  order_items."productId"          → products.id')
        lines.append('  order_items."productVariationId" → product_variations.id')
        lines.append('  order_payments."orderId"         → orders.id')
        lines.append('  order_payments."paymentMethodId" → payment_methods.id')
        lines.append('  products."productCategoryId"     → product_categories.id')
        lines.append('  product_variations."productId"   → products.id')

    lines.append("")
    lines.append("=== IMPORTANT RULES ===")
    lines.append("1. Always filter out cancelled orders: WHERE orders.isReopened = false")
    lines.append("2. Use orderedDate for date filtering, not createdAt.")
    lines.append("3. Revenue = totalAmountWithTax (includes tax).")
    lines.append("4. For product names, always JOIN order_items to products.")
    lines.append("5. For category names, always JOIN products to product_categories.")
    lines.append("6. Always use table aliases in JOINs for readability.")
    lines.append("7. LIMIT rules:")
    lines.append("   - Raw detail queries (SELECT * or multi-column rows): add LIMIT 100")
    lines.append("   - Grouped/aggregated queries (GROUP BY): NO LIMIT")
    lines.append("   - Single aggregate (SUM/COUNT/AVG no GROUP BY): NO LIMIT")
    lines.append("   - Top N queries (user says top 10, best 5): LIMIT to what user asked")

    schema_context = "\n".join(lines)

    logger.info(
        "Schema context built successfully (%d characters).", len(schema_context)
    )
    return schema_context


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Public interface
# ─────────────────────────────────────────────────────────────────────────────

def load_schema() -> tuple[psycopg2.extensions.connection, str]:
    """
    Main entry point for this module. Creates a DB connection and
    returns both the connection (for later query execution) and the
    fully built schema context string.

    Returns:
        Tuple of (db_connection, schema_context_string).

    Usage:
        conn, schema = load_schema()
        # Pass schema to the LLM, conn to the query executor.
    """
    conn = get_db_connection()
    schema_context = build_schema_context(conn)
    return conn, schema_context


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly to verify the schema loads correctly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        conn, schema = load_schema()
        print("\n" + "─" * 60)
        print(schema)
        print("─" * 60)
        print(f"\n✓ Schema loaded successfully ({len(schema)} characters)")
        conn.close()
    except Exception as e:
        logger.error("Schema load failed: %s", e)
        raise