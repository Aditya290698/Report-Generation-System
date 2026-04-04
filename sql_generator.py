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

ABSOLUTE RULES:
1. Output ONLY the raw SQL query. No explanations, no markdown, no backticks, no comments.
2. Only SELECT statements. Never INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE.
3. Only use tables and columns from the schema provided. Never invent names.
4. Always wrap ALL column names in double quotes: o."totalAmountWithTax"
   This is mandatory — PostgreSQL lowercases unquoted camelCase names.
5. Always use table aliases: mx51 for mx51_transactions, sc for square_checkouts, o for orders.
6. Format all currency/decimal values with ROUND(..., 2).
7. If a question cannot be answered from the schema: CANNOT_GENERATE: <reason>

LIMIT RULES:
- GROUP BY queries: NO LIMIT
- Single aggregate (SUM/COUNT/AVG, no GROUP BY): NO LIMIT
- Top N queries ("top 10"): LIMIT exactly what user asked

TIMEZONE RULES — ALL timestamps are stored as UTC:
NEVER: DATE(mx51."finalisedAt") = '2026-02-18'  -- unsafe, returns wrong results
ALWAYS use explicit UTC range:
  mx51."finalisedAt" >= '2026-02-18 00:00:00+00'
  AND mx51."finalisedAt" <  '2026-02-19 00:00:00+00'

DATE PATTERNS:
  Today:        col >= CURRENT_DATE::timestamptz AND col < (CURRENT_DATE + INTERVAL '1 day')::timestamptz
  This week:    col >= DATE_TRUNC('week', NOW())
  This month:   col >= DATE_TRUNC('month', NOW())
  Last 7 days:  col >= NOW() - INTERVAL '7 days'
  Last 30 days: col >= NOW() - INTERVAL '30 days'
  Specific day: col >= 'YYYY-MM-DD 00:00:00+00' AND col < 'YYYY-MM-DD+1 00:00:00+00'
  Date range:   col >= 'YYYY-MM-01 00:00:00+00' AND col < 'YYYY-MM-31 00:00:00+00'
  Group by day:   DATE(col AT TIME ZONE 'UTC') AS sale_date
  Group by week:  DATE_TRUNC('week', col AT TIME ZONE 'UTC') AS week_start
  Group by month: TO_CHAR(col AT TIME ZONE 'UTC', 'YYYY-MM') AS month

DATE COLUMN PER TABLE — use ONLY these (all timestamp with time zone, UTC):
  mx51_transactions  -> "finalisedAt"
  square_checkouts   -> "createdAt"
  orders             -> "orderedDate"
  order_items        -> "createdAt" or "dateTime"
  order_payments     -> "paymentDate" or "createdAt"
  customers          -> "createdAt"
  products           -> "createdAt"
  product_categories -> "createdAt"
  product_variations -> "createdAt"

NEVER filter on "updatedAt" on any table — no timezone info, returns wrong results.

MX51_TRANSACTIONS — PRIMARY sales source (200 rows of real data):
  "purchaseAmount"  — sale amount before tip/surcharge
  "finalAmount"     — total charged including tip+surcharge (use for revenue)
  "tipAmount"       — tip collected
  "surchargeAmount" — card surcharge
  "refundAmount"    — refund amount
  "resultFinancialStatus" — ALWAYS UPPERCASE: 'APPROVED', 'CANCELLED', 'DECLINED'
  "finalisedAt"     — transaction datetime in UTC

  Always use UPPER() when filtering status to handle any case variation:
    WHERE UPPER(mx51."resultFinancialStatus") = 'APPROVED'

  Revenue example:
    SELECT ROUND(SUM(mx51."finalAmount"), 2) AS total_revenue
    FROM mx51_transactions mx51
    WHERE UPPER(mx51."resultFinancialStatus") = 'APPROVED'
      AND mx51."finalisedAt" >= '2026-02-18 00:00:00+00'
      AND mx51."finalisedAt" <  '2026-02-19 00:00:00+00'

  Purchase amount on specific date (NO status filter unless user asks for approved only):
    SELECT ROUND(SUM(mx51."purchaseAmount"), 2) AS total_purchase_amount
    FROM mx51_transactions mx51
    WHERE mx51."finalisedAt" >= '2026-02-18 00:00:00+00'
      AND mx51."finalisedAt" <  '2026-02-19 00:00:00+00'

SQUARE_CHECKOUTS — SECONDARY sales source (2119 rows):
  "amount"    — transaction amount AUD
  "status"    — 'COMPLETED' or 'CANCELED' (UPPERCASE)
  "createdAt" — transaction datetime in UTC
  Always filter: WHERE UPPER(sc."status") = 'COMPLETED'

ORDERS TABLE — currently empty in dev DB, will have data in production.
  Revenue: o."totalAmountWithTax"
  Date:    o."orderedDate"
  Filter:  WHERE o."isReopened" = false

PRODUCTS:
  JOIN products p to product_categories pc ON p."productCategoryId" = pc."id"
  JOIN products p to product_variations pv ON pv."productId" = p."id"
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
) -> str:
    if not user_question or not user_question.strip():
        raise ValueError("user_question cannot be empty.")
    if not schema_context or not schema_context.strip():
        raise ValueError("schema_context cannot be empty.")

    user_question = user_question.strip()
    logger.info("Generating SQL for: '%s'", user_question)

    if client is None:
        client = get_openai_client()

    user_message = f"""
{schema_context}

=== USER QUESTION ===
{user_question}

Generate the PostgreSQL query to answer this question.
Output ONLY the raw SQL — no explanations, no markdown, no backticks.
"""

    try:
        logger.info("Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
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
# SECTION 5 — Retry wrapper
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql_with_retry(
    user_question: str,
    schema_context: str,
    max_attempts: int = 2,
) -> str:
    client = get_openai_client()
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Attempt %d of %d.", attempt, max_attempts)
            return generate_sql(user_question, schema_context, client=client)
        except SQLValidationError as e:
            last_error = e
            logger.warning("Attempt %d failed: %s", attempt, e)
            if attempt < max_attempts:
                user_question = (
                    f"{user_question}\n\n"
                    "REMINDER: Output ONLY the raw SQL. No markdown, no backticks."
                )

    raise SQLValidationError(
        f"SQL generation failed after {max_attempts} attempts. Last error: {last_error}"
    )


if __name__ == "__main__":
    from schema_loader import load_schema

    test_questions = [
        "What is the total purchase amount in mx51 transactions on 18th Feb 2026?",
        "Show monthly revenue from mx51 transactions",
        "How many products are in each category?",
        "What is the total revenue from all approved mx51 transactions?",
        "Show daily revenue from Square checkouts for the last 14 days",
    ]

    conn, schema = load_schema()
    for i, q in enumerate(test_questions, 1):
        print(f"\n[Test {i}] {q}")
        try:
            sql = generate_sql_with_retry(q, schema)
            print(f"SQL:\n{sql}")
        except SQLValidationError as e:
            print(f"Error: {e}")
    conn.close()