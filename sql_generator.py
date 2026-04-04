import os
import re
import logging
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from dotenv import load_dotenv

# ── Logging setup ─────────────────────────────────────────────────────────────
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
    """
    Creates and returns an OpenAI API client.

    The API key is read from the OPENAI_API_KEY environment variable.
    We validate its presence here so failures are caught early with a
    clear message rather than inside a deeply nested API call.

    Returns:
        OpenAI: Configured API client.

    Raises:
        EnvironmentError: If the API key is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    return OpenAI(api_key=api_key)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — System prompt for SQL generation
# ─────────────────────────────────────────────────────────────────────────────

# This is the instruction set that tells Claude exactly how to behave
# when generating SQL. A well-written system prompt is the difference
# between a reliable system and one that hallucinates column names.
SQL_GENERATOR_SYSTEM_PROMPT = """
You are an expert PostgreSQL query generator for a Point of Sale (POS) system.

Your job is to convert natural language questions into accurate, safe SQL queries.

STRICT RULES — follow these without exception:
1. Output ONLY the raw SQL query. No explanations, no markdown, no backticks.
2. Only use SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, 
   ALTER, TRUNCATE, CREATE, or any other data-modifying statement.
3. Only reference tables and columns from the schema provided. Never invent 
   column or table names.
4. Always wrap ALL column names in double quotes to preserve case sensitivity.
   e.g. o."totalAmountWithTax" not o.totalAmountWithTax
   This is required because the database uses camelCase column names.
5. Always use table aliases (e.g. o for orders, oi for order_items).
6. LIMIT rules — follow these carefully:
   - Raw row queries (SELECT * or multi-column detail): LIMIT 100
   - Aggregated/grouped queries (GROUP BY): NO LIMIT — grouping already reduces rows
   - Single aggregate (SUM, COUNT, AVG with no GROUP BY): NO LIMIT — always 1 row
   - Top N ranking queries (user says "top 10", "best 5"): LIMIT to exactly what user asked
7. For revenue figures, use totalAmountWithTax from the orders table.
8. For date filtering, use the orderedDate column on the orders table.
9. When joining order_items to get product names, always JOIN to the products table.
10. When grouping by category, JOIN products to product_categories.
11. Format currency columns with ROUND(..., 2) for clean decimal output.
12. If the question is ambiguous or cannot be answered with the given schema,
    return exactly this text and nothing else:
    CANNOT_GENERATE: <brief reason>

COMMON DATE PATTERNS TO USE:
- Today:          DATE(o.orderedDate) = CURRENT_DATE
- This week:      o.orderedDate >= DATE_TRUNC('week', CURRENT_DATE)
- This month:     o.orderedDate >= DATE_TRUNC('month', CURRENT_DATE)
- Last 7 days:    o.orderedDate >= CURRENT_DATE - INTERVAL '7 days'
- Last 30 days:   o.orderedDate >= CURRENT_DATE - INTERVAL '30 days'
- Date range:     o.orderedDate BETWEEN '2024-01-01' AND '2024-01-31'
- Daily group:    DATE(o.orderedDate) AS sale_date
- Weekly group:   DATE_TRUNC('week', o.orderedDate) AS week_start
- Monthly group:  TO_CHAR(o.orderedDate, 'YYYY-MM') AS month

SALES DATA SOURCES — use these when orders table is empty:
- mx51_transactions: PRIMARY sales source. Use "finalAmount" for revenue.
  Always filter: WHERE mx51."resultFinancialStatus" = 'approved'
  Use "finalisedAt" for date filtering.
  Example: SELECT ROUND(SUM(mx51."finalAmount"),2) FROM mx51_transactions mx51
           WHERE mx51."resultFinancialStatus" = 'approved'

- square_checkouts: SECONDARY sales source. Use "amount" for revenue.
  Always filter: WHERE sc."status" = 'COMPLETED'
  Use "createdAt" for date filtering.
  Example: SELECT ROUND(SUM(sc."amount"),2) FROM square_checkouts sc
           WHERE sc."status" = 'COMPLETED'
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SQL safety validator
# ─────────────────────────────────────────────────────────────────────────────

# These SQL keywords must never appear in a reporting query.
# We block them regardless of what the LLM generates.
FORBIDDEN_SQL_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "MERGE", "GRANT", "REVOKE", "EXEC",
    "EXECUTE", "CALL", "COPY", "pg_read_file", "pg_ls_dir",
]


class SQLValidationError(Exception):
    """
    Raised when generated SQL fails safety or syntax validation.
    Inheriting from Exception allows callers to catch it specifically.
    """
    pass


def validate_sql(sql: str) -> str:
    """
    Validates a generated SQL query for safety before execution.

    Checks performed:
    1. Not empty
    2. Starts with SELECT (read-only)
    3. Contains no forbidden data-modifying keywords
    4. Not excessively long (guards against prompt injection attacks
       that try to embed huge payloads in the query)

    Args:
        sql: The SQL string generated by the LLM.

    Returns:
        The cleaned, validated SQL string (stripped of whitespace).

    Raises:
        SQLValidationError: If any check fails, with a descriptive message.
    """
    # Strip whitespace and normalise
    sql = sql.strip()

    # ── Check 1: Not empty ────────────────────────────────────────────────────
    if not sql:
        raise SQLValidationError("Generated SQL is empty.")

    # ── Check 2: LLM signalled it couldn't generate ───────────────────────────
    # The system prompt tells Claude to return CANNOT_GENERATE: <reason>
    # when the question is unanswerable. We surface this as a clean error.
    if sql.upper().startswith("CANNOT_GENERATE"):
        reason = sql.split(":", 1)[1].strip() if ":" in sql else "Unknown reason"
        raise SQLValidationError(f"Query could not be generated: {reason}")

    # ── Check 3: Must be a SELECT statement ───────────────────────────────────
    # Normalise to uppercase for the check, but keep original for execution.
    sql_upper = sql.upper()
    if not sql_upper.lstrip().startswith("SELECT"):
        raise SQLValidationError(
            "Only SELECT queries are allowed. "
            f"Query starts with: '{sql[:30]}...'"
        )

    # ── Check 4: No forbidden keywords ────────────────────────────────────────
    # Using word-boundary regex so 'CREATED_AT' doesn't falsely match 'CREATE'.
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, sql_upper):
            raise SQLValidationError(
                f"Forbidden SQL keyword detected: '{keyword}'. "
                "Only read-only SELECT queries are permitted."
            )

    # ── Check 5: Length guard ─────────────────────────────────────────────────
    # A legitimate reporting query should never be 5000+ characters.
    # An unusually long query suggests something went wrong (e.g. the LLM
    # included explanation text, or a prompt injection attempt).
    if len(sql) > 5000:
        raise SQLValidationError(
            f"Generated SQL is unusually long ({len(sql)} chars). "
            "This may indicate a malformed response."
        )

    logger.info("SQL validation passed (%d characters).", len(sql))
    return sql


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SQL generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql(
    user_question: str,
    schema_context: str,
    client: OpenAI | None = None,
) -> str:
    """
    Converts a natural language question into a validated SQL query
    using OpenAI (LLM Pass 1).

    Flow:
        user_question + schema_context
              ↓
        OpenAI (SQL generator system prompt)
              ↓
        raw SQL string
              ↓
        SQL validator
              ↓
        validated SQL (ready for execution)

    Args:
        user_question: The natural language question from the user.
                       e.g. "What were the top 5 products last week?"
        schema_context: The full schema string from schema_loader.py.
        client: Optional pre-built OpenAI client. If None, one is
                created automatically. Pass one in when reusing across
                multiple calls to avoid re-initialising on every request.

    Returns:
        A validated SQL string ready to be executed.

    Raises:
        SQLValidationError: If the generated SQL fails validation.
        openai.APIError: On API communication failures.
        ValueError: If inputs are empty.
    """

    # ── Input validation ──────────────────────────────────────────────────────
    if not user_question or not user_question.strip():
        raise ValueError("user_question cannot be empty.")
    if not schema_context or not schema_context.strip():
        raise ValueError("schema_context cannot be empty.")

    user_question = user_question.strip()
    logger.info("Generating SQL for question: '%s'", user_question)

    # ── Build the API client ──────────────────────────────────────────────────
    if client is None:
        client = get_openai_client()

    # ── Build the user message ────────────────────────────────────────────────
    # We combine the schema context and the question in one message.
    # The schema goes first so the model has full context before it sees
    # the question — this mirrors how a human analyst would work.
    user_message = f"""
{schema_context}

=== USER QUESTION ===
{user_question}

Generate the PostgreSQL query to answer this question.
Remember: output ONLY the raw SQL, nothing else.
"""

    # ── Call the OpenAI API ───────────────────────────────────────────────────
    # OpenAI uses the Chat Completions format: system prompt goes in the
    # "system" role message, user question in the "user" role message.
    # This is different from Anthropic where system is a separate parameter.
    try:
        logger.info("Calling OpenAI API for SQL generation...")
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,       # SQL queries don't need more than this
            messages=[
                {"role": "system", "content": SQL_GENERATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )

        # Extract the text content from the response.
        # OpenAI returns choices[0].message.content as a plain string.
        raw_sql = response.choices[0].message.content
        if not raw_sql:
            raise SQLValidationError("OpenAI returned an empty response.")

        raw_sql = raw_sql.strip()
        logger.info("Raw SQL received from OpenAI:\n%s", raw_sql)

    except APIConnectionError as e:
        logger.error("Network error connecting to OpenAI API: %s", e)
        raise
    except RateLimitError as e:
        logger.error("OpenAI API rate limit hit: %s", e)
        raise
    except APIStatusError as e:
        logger.error("OpenAI API error (status %s): %s", e.status_code, e.message)
        raise

    # ── Validate before returning ─────────────────────────────────────────────
    # This is the safety gate — the SQL never reaches the database
    # without passing all validation checks.
    validated_sql = validate_sql(raw_sql)

    return validated_sql


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Convenience wrapper with retry logic
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql_with_retry(
    user_question: str,
    schema_context: str,
    max_attempts: int = 2,
) -> str:
    """
    Wraps generate_sql() with simple retry logic. If the first attempt
    produces invalid SQL, we ask Claude again with an explicit reminder
    to output only SQL. This handles occasional LLM formatting slips
    (e.g. Claude wrapping the query in markdown code fences).

    Args:
        user_question: Natural language question.
        schema_context: Schema context string.
        max_attempts: How many times to try before giving up (default 2).

    Returns:
        Validated SQL string.

    Raises:
        SQLValidationError: After all attempts fail.
    """
    client = get_openai_client()  # Create once, reuse across retries
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("SQL generation attempt %d of %d.", attempt, max_attempts)
            return generate_sql(user_question, schema_context, client=client)

        except SQLValidationError as e:
            last_error = e
            logger.warning("Attempt %d failed validation: %s", attempt, e)

            if attempt < max_attempts:
                # Append a stricter instruction for the retry
                user_question = (
                    f"{user_question}\n\n"
                    "REMINDER: Your previous response was not valid SQL. "
                    "Output ONLY the raw SQL query with no explanation, "
                    "no markdown, and no code fences."
                )

    # All attempts exhausted
    raise SQLValidationError(
        f"SQL generation failed after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from schema_loader import load_schema

    # A set of sample questions to test the generator
    test_questions = [
        "What is the total revenue for this month?",
        "Show me the top 5 best selling products this week.",
        "How many orders were placed each day in the last 7 days?",
        "What is the average order value for this month?",
        "Show revenue broken down by product category.",
    ]

    try:
        print("Loading schema from database...")
        conn, schema = load_schema()

        print(f"\n{'─'*60}")
        print("Running SQL generation tests...")
        print(f"{'─'*60}\n")

        for i, question in enumerate(test_questions, 1):
            print(f"[Test {i}] Question: {question}")
            try:
                sql = generate_sql_with_retry(question, schema)
                print(f"Generated SQL:\n{sql}")
            except SQLValidationError as e:
                print(f"Validation Error: {e}")
            except Exception as e:
                print(f"Error: {e}")
            print(f"{'─'*60}\n")

        conn.close()

    except Exception as e:
        logger.error("Test run failed: %s", e)
        raise