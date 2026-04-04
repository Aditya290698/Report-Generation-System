import re
import sys
import logging
import argparse

from schema_loader    import load_schema
from sql_generator    import generate_sql_with_retry, SQLValidationError
from query_executor   import execute_query, format_results_for_llm, \
                             QueryExecutionError, EmptyResultError
from report_generator import generate_report, export_report_to_pdf

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_report(user_question: str, schema_context: str, conn, export_pdf: bool = False) -> dict:
    """
    Runs the full two-pass pipeline for a single question:
        Question → SQL (Pass 1) → Execute → Report + Chart (Pass 2) → PDF

    Args:
        user_question:  Natural language question from the user.
        schema_context: Schema string from schema_loader.
        conn:           Active DB connection from schema_loader.
        export_pdf:     If True, exports a PDF report to ./reports/

    Returns:
        dict with keys: question, sql, raw_result, report, pdf_path

    Raises:
        SQLValidationError, QueryExecutionError, EmptyResultError
    """
    sep = "─"*60
    print(f"\n{sep}")
    print(f"Question: {user_question}")
    sep = "─"*60
    print(f"{sep}")

    # ── Pass 1: Generate SQL ──────────────────────────────────────────────────
    print("Generating SQL (Pass 1)...")
    sql = generate_sql_with_retry(user_question, schema_context)
    print(f"\nGenerated SQL:\n{sql}")

    # ── Execute ───────────────────────────────────────────────────────────────
    print("\nExecuting query...")
    raw_result = execute_query(conn, sql)
    row_count = raw_result["row_count"]
    print(f"Query returned {row_count} rows.")

    # ── Pass 2: Generate report ───────────────────────────────────────────────
    print("\nGenerating report (Pass 2)...")
    report = generate_report(user_question, sql, raw_result)

    # ── Print report to terminal ──────────────────────────────────────────────
    sep2 = "═"*60
    print(f"\n{sep2}")
    print(f"REPORT")
    sep2 = "═"*60
    print(f"{sep2}")
    print(f"\nSummary:\n{report.summary}")
    print(f"\nKey Insights:")
    for insight in report.key_insights:
        if insight:
            print(f"  • {insight}")
    print(f"\nAnalysis:\n{report.narrative}")
    print(f"\nChart recommendation: {report.chart.chart_type}")
    print(f"Reason: {report.chart.reasoning}")

    # ── Optional PDF export ───────────────────────────────────────────────────
    pdf_path = None
    if export_pdf and report.has_data:
        import os
        os.makedirs("reports", exist_ok=True)
        safe_name = re.sub(r"[^a-z0-9]+", "_", user_question.lower())[:50]
        pdf_path = f"reports/{safe_name}.pdf"
        export_report_to_pdf(report, pdf_path)
        print(f"\nPDF saved to: {pdf_path}")

    return {
        "question":   user_question,
        "sql":        sql,
        "raw_result": raw_result,
        "report":     report,
        "pdf_path":   pdf_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Interactive CLI loop
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "What is the total revenue for this month?",
    "Show the top 10 best selling products this week.",
    "How many orders were placed each day in the last 7 days?",
    "What is the average order value for this month?",
    "Show revenue broken down by product category.",
    "How many orders were placed today?",
    "Which customers have spent the most this month?",
    "Show daily revenue for the last 30 days.",
    "What percentage of orders were paid by card vs cash?",
    "Show the number of orders and revenue by order type.",
]


def print_sample_questions():
    print("\nSample questions you can ask:")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"  {i:2}. {q}")
    print()


def run_interactive(schema_context: str, conn):
    """
    Runs an interactive CLI loop where the user can type questions
    and get SQL + results back in real time.

    Type 'exit' or 'quit' to stop.
    Type 'examples' to see sample questions.
    """
    print("\n" + "=" * 60)
    print("  POS LLM Reporting System — SQL Generator")
    print("=" * 60)
    print("Type your question in plain English and press Enter.")
    print("Type 'examples' to see sample questions.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            # Prompt the user for a question
            user_input = input("Your question: ").strip()

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D gracefully
            print("\nExiting. Goodbye!")
            break

        # ── Handle special commands ───────────────────────────────────────────
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye!")
            break

        if user_input.lower() in ("examples", "sample", "help"):
            print_sample_questions()
            continue

        # ── Run the report pipeline ───────────────────────────────────────────
        try:
            run_report(user_input, schema_context, conn)

        except SQLValidationError as e:
            # SQL generation or validation failed — surface clearly
            print(f"\n[Error] Could not generate a valid SQL query: {e}")
            print("Try rephrasing your question.\n")

        except EmptyResultError as e:
            # Query ran fine but no data matched
            print(f"\n[No Results] {e}\n")

        except QueryExecutionError as e:
            # Database-level failure
            print(f"\n[Database Error] {e}\n")

        except Exception as e:
            # Unexpected error — log full details but show user a clean message
            logger.error("Unexpected error: %s", e, exc_info=True)
            print(f"\n[Unexpected Error] {e}")
            print("Check the logs for details.\n")

        print()  # Blank line between questions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Supports:
      --question   Run a single question and exit (non-interactive mode)
      --debug      Enable DEBUG level logging for detailed output
    """
    parser = argparse.ArgumentParser(
        description="POS LLM Reporting System — Natural language to SQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --question "Total revenue this month"
  python main.py --question "Top products this week" --debug
        """
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Run a single question and exit (non-interactive mode).",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Export the report as a PDF file into the ./reports/ folder.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging for detailed output.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Adjust log level if --debug flag is passed
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    # ── Step 1: Load schema (one-time setup) ──────────────────────────────────
    # This connects to the DB and builds the schema context string.
    # We do this once at startup so every subsequent question reuses
    # the same connection and schema — no reconnecting per query.
    print("Connecting to database and loading schema...")
    try:
        conn, schema_context = load_schema()
        print("Schema loaded successfully.")
    except EnvironmentError as e:
        print(f"[Config Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Startup Error] Failed to load schema: {e}")
        logger.error("Schema load failed: %s", e, exc_info=True)
        sys.exit(1)

    # ── Step 2: Run either single-question or interactive mode ────────────────
    try:
        if args.question:
            # Non-interactive: run one question and exit.
            # Useful for scripting or testing.
            try:
                run_report(args.question, schema_context, conn, export_pdf=args.pdf)
            except SQLValidationError as e:
                print(f"\n[Error] {e}")
                sys.exit(1)
            except EmptyResultError as e:
                print(f"\n[No Results] {e}")
            except QueryExecutionError as e:
                print(f"\n[Database Error] {e}")
                sys.exit(1)
        else:
            # Interactive mode: loop until user exits
            run_interactive(schema_context, conn)

    finally:
        # Always close the DB connection cleanly — even if something crashes
        if conn and not conn.closed:
            conn.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    main()