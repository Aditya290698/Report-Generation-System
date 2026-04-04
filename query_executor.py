import logging
import psycopg2
import psycopg2.extras

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class QueryExecutionError(Exception):
    """
    Raised when a query fails at the database level.
    Distinct from SQLValidationError (which catches bad SQL before
    execution) — this catches runtime failures like type mismatches,
    missing tables at runtime, or timeouts.
    """
    pass


class EmptyResultError(Exception):
    """
    Raised when a query returns zero rows. This is a separate exception
    from a query failure — the SQL ran correctly, there's just no data
    matching the criteria (e.g. no orders in the selected date range).
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Query executor
# ─────────────────────────────────────────────────────────────────────────────

def execute_query(conn, sql: str) -> dict:
    """
    Executes a validated SQL query and returns structured results.

    The result dict contains:
    - "columns": list of column names in the order they appear
    - "rows":    list of dicts, one per result row
    - "row_count": number of rows returned

    Using dicts for rows (via RealDictCursor) means downstream code
    accesses values by name — row["total_revenue"] — not by index,
    which makes the report generator much more readable.

    Args:
        conn: Active psycopg2 connection (from schema_loader.load_schema).
        sql:  Validated SQL string from sql_generator.generate_sql.

    Returns:
        dict with keys: columns, rows, row_count.

    Raises:
        QueryExecutionError: On database-level failures.
        EmptyResultError: When the query returns zero rows.
    """
    logger.info("Executing SQL query...")
    logger.debug("Query:\n%s", sql)

    try:
        # Use a separate cursor from the connection.
        # We use RealDictCursor here specifically (even if the connection
        # was opened with it as default) to be explicit and safe.
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

            # Set a statement timeout on this specific query.
            # This prevents runaway queries from locking the DB.
            # 30 seconds is generous for any reporting query.
            cursor.execute("SET statement_timeout = '30s';")

            # Execute the actual reporting query
            cursor.execute(sql)

            rows = cursor.fetchall()

            # Extract column names from the cursor description.
            # cursor.description is a list of Column objects; [0] is the name.
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Convert RealDictRow objects to plain Python dicts
            # so the results are easy to serialise (e.g. to JSON)
            # and work naturally with all downstream code.
            plain_rows = [dict(row) for row in rows]

    except psycopg2.errors.QueryCanceled as e:
        # This triggers when statement_timeout is exceeded
        logger.error("Query timed out after 30 seconds: %s", e)
        raise QueryExecutionError(
            "The query took too long to execute (>30s). "
            "Try narrowing your date range or adding more filters."
        )

    except psycopg2.errors.UndefinedTable as e:
        logger.error("Query references a table that doesn't exist: %s", e)
        raise QueryExecutionError(
            f"Query references an unknown table. Details: {e}"
        )

    except psycopg2.errors.UndefinedColumn as e:
        logger.error("Query references a column that doesn't exist: %s", e)
        raise QueryExecutionError(
            f"Query references an unknown column. Details: {e}"
        )

    except psycopg2.Error as e:
        # Catch-all for any other psycopg2 error
        logger.error("Database error during query execution: %s", e)
        raise QueryExecutionError(f"Database error: {e}")

    # ── Check for empty results ───────────────────────────────────────────────
    if not plain_rows:
        logger.info("Query returned 0 rows.")
        raise EmptyResultError(
            "The query returned no results. "
            "There may be no data matching your criteria "
            "(e.g. no orders in the selected date range)."
        )

    logger.info(
        "Query executed successfully. Returned %d rows with %d columns.",
        len(plain_rows),
        len(columns),
    )

    return {
        "columns":   columns,
        "rows":      plain_rows,
        "row_count": len(plain_rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Result formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_results_for_llm(query_result: dict, user_question: str) -> str:
    """
    Converts raw query results into a clean text block that gets passed
    to LLM Pass 2 (the report writer).

    We deliberately format this as structured text rather than JSON
    because the LLM reads and reasons about structured text more reliably
    than parsing raw JSON, especially for numeric analysis.

    Null handling:
    - A single-row result where ALL values are None means an aggregation
      (e.g. SUM, COUNT) ran on an empty table — we surface this clearly
      rather than showing a confusing table of N/A values.
    - Individual None values in multi-row results are shown as "No data"
      with a note explaining what that means.

    Args:
        query_result: The dict returned by execute_query().
        user_question: The original question — included so the report
                       writer knows what was asked.

    Returns:
        A formatted string ready to be injected into the report-writer prompt.
    """
    columns   = query_result["columns"]
    rows      = query_result["rows"]
    row_count = query_result["row_count"]

    # ── Detect all-null result ────────────────────────────────────────────────
    # This happens when an aggregation like SUM() or AVG() runs against
    # an empty table — PostgreSQL returns one row with all NULLs.
    # Example: SELECT SUM("totalAmountWithTax") FROM orders → {total: None}
    # We catch this early and return a clear human-readable message
    # instead of a table full of "No data" which is confusing.
    if row_count == 1:
        all_values = list(rows[0].values())
        if all(v is None for v in all_values):
            logger.info(
                "Query returned a single all-null row — "
                "aggregation on empty data detected."
            )
            return (
                f"=== QUERY RESULTS ===\n"
                f"Original question: {user_question}\n\n"
                f"No data found.\n"
                f"The query ran successfully but the relevant table(s) "
                f"contain no records matching your criteria.\n"
                f"This usually means there are no transactions in the "
                f"database yet for the selected time period or filters."
            )

    lines = []
    lines.append("=== QUERY RESULTS ===")
    lines.append(f"Original question: {user_question}")
    lines.append(f"Rows returned: {row_count}")

    # ── Track how many nulls we find across the whole result ─────────────────
    null_columns = set()

    lines.append("")

    # ── Column headers ────────────────────────────────────────────────────────
    header = " | ".join(columns)
    separator = "-" * len(header)
    lines.append(header)
    lines.append(separator)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row in rows:
        formatted_values = []
        for col in columns:
            val = row.get(col)
            if val is None:
                # Track which columns have nulls so we can explain them
                null_columns.add(col)
                formatted_values.append("No data")
            elif isinstance(val, float):
                # Round floats to 2dp — prevents values like 48250.000000001
                formatted_values.append(f"{val:.2f}")
            else:
                formatted_values.append(str(val))

        lines.append(" | ".join(formatted_values))

    lines.append("")
    lines.append(f"Total rows: {row_count}")

    # ── Null column explanation ───────────────────────────────────────────────
    # If any columns had null values, add a clear note at the bottom
    # explaining what "No data" means for each affected column.
    if null_columns:
        lines.append("")
        lines.append("Note — the following columns had missing values (shown as 'No data'):")
        for col in sorted(null_columns):
            lines.append(
                f"  - {col}: this field was not recorded for some records, "
                f"or the related table has no matching entry."
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from schema_loader import load_schema
    from sql_generator import generate_sql_with_retry

    test_question = "How many orders were placed today?"

    try:
        print("Loading schema...")
        conn, schema = load_schema()

        print(f"Generating SQL for: '{test_question}'")
        sql = generate_sql_with_retry(test_question, schema)
        print(f"\nGenerated SQL:\n{sql}\n")

        print("Executing query...")
        result = execute_query(conn, sql)

        print("\nFormatted results:")
        print(format_results_for_llm(result, test_question))

        conn.close()

    except EmptyResultError as e:
        print(f"No data: {e}")
    except QueryExecutionError as e:
        print(f"Execution error: {e}")
    except Exception as e:
        logger.error("Test failed: %s", e)
        raise