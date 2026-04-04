"""
api.py
------
FastAPI application for the POS LLM Reporting System.

Exposes the full two-pass LLM pipeline (SQL generation + report writing)
as a REST API. The web UI and any other client calls this API — they
never touch the database or LLM directly.

Endpoints:
    POST /report         → Run a question through the full pipeline
    GET  /report/pdf     → Download the last generated report as PDF
    GET  /health         → Check API, DB, and OpenAI connectivity
    GET  /tables         → List the reporting tables in scope
    GET  /docs           → Auto-generated Swagger UI (built into FastAPI)

Run with:
    uvicorn app:app --reload --port 8000

Then open:
    http://localhost:8000/docs   ← Interactive API playground
"""

import os
import re
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import secrets
import base64

from schema_loader    import load_schema, REPORTING_TABLES
from sql_generator    import generate_sql_with_retry, SQLValidationError
from query_executor   import execute_query, QueryExecutionError, EmptyResultError
from report_generator import generate_report, export_report_to_pdf, ReportOutput

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — App state
# Shared objects that live for the lifetime of the server process.
# We load the DB connection and schema once at startup — not on every request.
# This saves ~2 seconds per request (no reconnect, no schema rebuild).
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """Holds resources shared across all requests."""
    conn         = None   # psycopg2 connection
    schema       = None   # Schema context string
    pdf_store: dict[str, str] = {}  # report_id → pdf file path


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1b — Authentication
# Manual Basic Auth parsing — we deliberately do NOT use FastAPI's HTTPBasic
# security scheme because that sends WWW-Authenticate: Basic in the response
# header, which causes browsers to show their native login popup dialog.
# Instead we parse the Authorization header ourselves and return a plain 401
# JSON response that our custom login UI handles gracefully.
# ─────────────────────────────────────────────────────────────────────────────

def verify_credentials(request: Request):
    """
    Manually parses the Authorization: Basic header and verifies credentials
    against APP_USERNAME and APP_PASSWORD environment variables.

    Returns plain 401 JSON (no WWW-Authenticate header) so the browser
    never shows its native HTTP Basic Auth popup dialog.

    Raises:
        HTTPException 401: If credentials are missing or wrong.
    """
    expected_user = os.getenv("APP_USERNAME", "admin")
    expected_pass = os.getenv("APP_PASSWORD", "admin")

    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Basic "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    try:
        decoded    = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format.",
        )

    user_ok = secrets.compare_digest(
        username.encode("utf-8"),
        expected_user.encode("utf-8"),
    )
    pass_ok = secrets.compare_digest(
        password.encode("utf-8"),
        expected_pass.encode("utf-8"),
    )

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return username


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Lifespan (startup + shutdown)
# FastAPI's lifespan replaces the old @app.on_event("startup") pattern.
# Everything before `yield` runs at startup; everything after at shutdown.
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect to DB and build schema context.
    Shutdown: close DB connection cleanly.

    IMPORTANT: We do NOT raise on DB failure — if we crash at startup
    the health check endpoint never responds and Railway marks it unhealthy.
    Instead we log the error and let the app start in degraded mode.
    The /health endpoint will report the actual status.
    """
    logger.info("Starting POS Reporting API...")

    try:
        state.conn, state.schema = load_schema()
        logger.info("Database connected and schema loaded successfully.")
    except Exception as e:
        # Log but do NOT raise — let the server start so /health can respond
        logger.error("Could not connect to database at startup: %s", e)
        logger.warning("Server starting in degraded mode — DB unavailable.")
        state.conn   = None
        state.schema = ""

    # Create PDF output directory
    Path("reports").mkdir(exist_ok=True)

    logger.info("POS Reporting API is ready.")
    yield   # Server is running — handle requests

    # Shutdown
    if state.conn and not state.conn.closed:
        state.conn.close()
        logger.info("Database connection closed.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="POS LLM Reporting API",
    description=(
        "Natural language reporting for the CygenPOS system. "
        "Ask a question in plain English, get SQL, results, a report, and a chart."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Read allowed origins from environment variable so we can lock it down
# in production without changing code.
# .env: ALLOWED_ORIGINS=https://yourapp.com,https://www.yourapp.com
# Development default: * (allow all)
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS_LIST = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Request / Response models
# Pydantic models give us automatic validation, clear API documentation,
# and IDE autocompletion — all from just defining the shape of the data.
# ─────────────────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    """Body of POST /report"""
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language question about your POS data.",
        examples=["How many products are in each category?",
                  "What is the total revenue from mx51 transactions?"],
    )
    export_pdf: bool = Field(
        default=False,
        description="If true, a PDF is generated and its download ID is returned.",
    )


class ChartDataset(BaseModel):
    """One dataset within a chart (one line, one set of bars, etc.)"""
    label:           str
    data:            list[float]
    backgroundColor: Any        # str for bar/line, list[str] for pie
    borderColor:     str
    borderWidth:     int
    fill:            bool


class ChartConfigResponse(BaseModel):
    """Chart configuration — passed directly to Chart.js in the frontend."""
    chart_type: str
    title:      str
    labels:     list[str]
    datasets:   list[ChartDataset]
    x_label:    str
    y_label:    str
    reasoning:  str             # Why this chart type was chosen


class ReportResponse(BaseModel):
    """Full response from POST /report"""
    report_id:    str           # UUID — use this to download the PDF
    question:     str
    sql:          str           # Generated SQL (useful for debugging)
    summary:      str           # 2-3 sentence executive summary
    narrative:    str           # Full report text
    key_insights: list[str]    # 3 bullet points
    chart:        ChartConfigResponse
    row_count:    int
    has_data:     bool
    generated_at: str           # ISO timestamp
    pdf_url:      str | None    # e.g. "/report/pdf/{report_id}" if exported


class HealthResponse(BaseModel):
    """Response from GET /health"""
    status:      str
    database:    str
    schema_size: int            # Characters in the schema context
    openai:      str
    timestamp:   str


class ErrorResponse(BaseModel):
    """Standard error shape returned on 4xx/5xx"""
    error:   str
    detail:  str
    code:    str


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Helper: reconnect if DB connection dropped
# Long-running servers sometimes lose their DB connection. This helper
# detects a closed connection and transparently reconnects before the
# next query, so users never see a "connection closed" error.
# ─────────────────────────────────────────────────────────────────────────────

def ensure_db_connection():
    """
    Checks if the DB connection is alive and reconnects if not.
    Called at the start of every endpoint that touches the database.
    """
    if state.conn is None or state.conn.closed:
        logger.warning("DB connection lost — reconnecting...")
        state.conn, state.schema = load_schema()
        logger.info("Reconnected successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Helper: convert ReportOutput → ReportResponse
# ─────────────────────────────────────────────────────────────────────────────

def build_response(
    report: ReportOutput,
    report_id: str,
    pdf_url: str | None = None,
) -> ReportResponse:
    """
    Converts the internal ReportOutput dataclass into the API response model.
    Keeps the internal domain model decoupled from the API contract.
    """
    chart = report.chart
    datasets = [
        ChartDataset(
            label=ds["label"],
            data=ds["data"],
            backgroundColor=ds["backgroundColor"],
            borderColor=ds["borderColor"],
            borderWidth=ds["borderWidth"],
            fill=ds["fill"],
        )
        for ds in chart.datasets
    ]

    return ReportResponse(
        report_id=report_id,
        question=report.question,
        sql=report.sql,
        summary=report.summary,
        narrative=report.narrative,
        key_insights=report.key_insights,
        chart=ChartConfigResponse(
            chart_type=chart.chart_type,
            title=chart.title,
            labels=chart.labels,
            datasets=datasets,
            x_label=chart.x_label,
            y_label=chart.y_label,
            reasoning=chart.reasoning,
        ),
        row_count=report.row_count,
        has_data=report.has_data,
        generated_at=datetime.utcnow().isoformat() + "Z",
        pdf_url=pdf_url,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Endpoints
# ─────────────────────────────────────────────────────────────────────────────

# ── POST /report ──────────────────────────────────────────────────────────────
@app.post(
    "/report",
    response_model=ReportResponse,
    summary="Generate a report from a natural language question",
    tags=["Reporting"],
)
async def create_report(
    request: ReportRequest,
    http_request: Request,
):
    verify_credentials(http_request)
    """
    The main endpoint. Full pipeline:

    1. Validate the question
    2. Generate SQL (LLM Pass 1)
    3. Execute SQL against PostgreSQL
    4. Generate report + select chart type (LLM Pass 2)
    5. Optionally export PDF
    6. Return everything the UI needs

    **Example request:**
    ```json
    { "question": "How many products are in each category?", "export_pdf": true }
    ```
    """
    ensure_db_connection()

    report_id = str(uuid.uuid4())
    logger.info("[%s] New report request: '%s'", report_id, request.question)

    # ── Pass 1: Generate SQL ──────────────────────────────────────────────────
    try:
        sql = generate_sql_with_retry(request.question, state.schema)
        logger.info("[%s] SQL generated: %s", report_id, sql[:80])
    except SQLValidationError as e:
        logger.warning("[%s] SQL validation failed: %s", report_id, e)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "sql_generation_failed",
                "detail": str(e),
                "code": "SQL_INVALID",
            },
        )
    except Exception as e:
        logger.error("[%s] Unexpected SQL generation error: %s", report_id, e)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "sql_generation_error",
                "detail": "Failed to generate SQL query. Please try again.",
                "code": "INTERNAL_ERROR",
            },
        )

    # ── Execute SQL ───────────────────────────────────────────────────────────
    try:
        query_result = execute_query(state.conn, sql)
        logger.info("[%s] Query returned %d rows.", report_id,
                    query_result["row_count"])
    except EmptyResultError as e:
        # Not a server error — the query worked, there's just no data
        logger.info("[%s] Query returned no results: %s", report_id, e)
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_data",
                "detail": str(e),
                "code": "NO_DATA",
            },
        )
    except QueryExecutionError as e:
        logger.error("[%s] Query execution failed: %s", report_id, e)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "query_execution_failed",
                "detail": str(e),
                "code": "DB_ERROR",
            },
        )

    # ── Pass 2: Generate report ───────────────────────────────────────────────
    try:
        report = generate_report(request.question, sql, query_result)
        logger.info("[%s] Report generated. Chart type: %s",
                    report_id, report.chart.chart_type)
    except Exception as e:
        logger.error("[%s] Report generation failed: %s", report_id, e)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "report_generation_failed",
                "detail": "Failed to generate report narrative. Please try again.",
                "code": "LLM_ERROR",
            },
        )

    # ── Optional PDF export ───────────────────────────────────────────────────
    pdf_url = None
    if request.export_pdf and report.has_data:
        try:
            safe_name = re.sub(r"[^a-z0-9]+", "_", request.question.lower())[:50]
            pdf_path  = f"reports/{report_id}_{safe_name}.pdf"
            export_report_to_pdf(report, pdf_path)
            state.pdf_store[report_id] = pdf_path
            pdf_url = f"/report/pdf/{report_id}"
            logger.info("[%s] PDF exported: %s", report_id, pdf_path)
        except Exception as e:
            # PDF failure is non-fatal — return the report without PDF
            logger.warning("[%s] PDF export failed (non-fatal): %s", report_id, e)

    return build_response(report, report_id, pdf_url)


# ── GET /report/pdf/{report_id} ───────────────────────────────────────────────
@app.get(
    "/report/pdf/{report_id}",
    summary="Download a generated report as PDF",
    tags=["Reporting"],
    response_class=FileResponse,
)
async def download_pdf(
    report_id: str,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    verify_credentials(http_request)
    """
    Downloads the PDF for a previously generated report.

    The `report_id` comes from the `report_id` field in the POST /report response.
    PDFs are kept for the lifetime of the server process — they are cleaned up
    when the server restarts.

    **Note:** You must have requested `export_pdf: true` in POST /report first.
    """
    pdf_path = state.pdf_store.get(report_id)

    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "pdf_not_found",
                "detail": (
                    f"No PDF found for report_id '{report_id}'. "
                    "Make sure you set export_pdf: true when creating the report."
                ),
                "code": "PDF_NOT_FOUND",
            },
        )

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=410,
            detail={
                "error": "pdf_expired",
                "detail": "The PDF file no longer exists. Please regenerate the report.",
                "code": "PDF_EXPIRED",
            },
        )

    # Use a clean filename for the download dialog
    filename = f"report_{report_id[:8]}.pdf"

    logger.info("Serving PDF download: %s", pdf_path)

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
    )


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API, database, and OpenAI connectivity",
    tags=["System"],
)
async def health_check():
    """
    Checks all three components the API depends on:
    - Database: runs a lightweight SELECT 1 query
    - Schema: confirms the schema context was loaded
    - OpenAI: confirms the API key is present (does not make an API call)

    Returns overall `status: "ok"` only if all three are healthy.
    """
    db_status     = "unknown"
    openai_status = "unknown"

    # ── Check database ────────────────────────────────────────────────────────
    try:
        ensure_db_connection()
        with state.conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:60]}"
        logger.error("Health check — DB error: %s", e)
        # Try to reconnect once more in case it was a transient failure
        try:
            state.conn, state.schema = load_schema()
            with state.conn.cursor() as cur:
                cur.execute("SELECT 1")
            db_status = "connected"
            logger.info("Health check — DB reconnected successfully.")
        except Exception as e2:
            db_status = f"error: {str(e2)[:60]}"

    # ── Check OpenAI key ──────────────────────────────────────────────────────
    if os.getenv("OPENAI_API_KEY"):
        openai_status = "api_key_present"
    else:
        openai_status = "error: OPENAI_API_KEY not set"

    # ── Overall status ────────────────────────────────────────────────────────
    all_ok = (db_status == "connected" and openai_status == "api_key_present")

    if not all_ok:
        # Return 503 so load balancers / monitoring tools can detect unhealthy
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="degraded",
                database=db_status,
                schema_size=len(state.schema) if state.schema else 0,
                openai=openai_status,
                timestamp=datetime.utcnow().isoformat() + "Z",
            ).model_dump(),
        )

    return HealthResponse(
        status="ok",
        database=db_status,
        schema_size=len(state.schema) if state.schema else 0,
        openai=openai_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ── GET /tables ───────────────────────────────────────────────────────────────
@app.get(
    "/tables",
    summary="List the tables available for reporting",
    tags=["System"],
)
async def list_tables():
    """
    Returns the list of database tables the LLM is allowed to query.
    Useful for the UI to show users what data is available.
    """
    return {
        "tables": REPORTING_TABLES,
        "count":  len(REPORTING_TABLES),
    }


# ── POST /login ──────────────────────────────────────────────────────────────
@app.post("/login", summary="Verify credentials", tags=["Auth"])
async def login(http_request: Request):
    """
    Verifies username and password. Returns user info on success.
    The web UI calls this on the login form submit to confirm credentials
    before showing the main dashboard.
    """
    username = verify_credentials(http_request)
    return {
        "status":   "authenticated",
        "username": username,
        "message":  "Login successful.",
    }


# ── GET / ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    """Serve the web UI."""
    index_path = Path("index.html")
    if index_path.exists():
        return FileResponse("index.html", media_type="text/html")
    return {
        "message": "POS LLM Reporting API",
        "docs":    "/docs",
        "health":  "/health",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # Auto-restart on file changes during development
        log_level="info",
    )