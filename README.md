# LLM-Based POS Reporting System

AI-powered natural language reporting on a Point of Sale PostgreSQL database. Ask questions in plain English — the system generates SQL, runs it, writes a report, selects the right chart, and exports a PDF automatically.

**Live demo:** https://web-production-a461.up.railway.app

---

## How it works

```
User question  →  Schema context injected  →  LLM Pass 1 (SQL generator)
      →  SQL validator  →  PostgreSQL query  →  LLM Pass 2 (report writer)
      →  Report + Chart + PDF
```

1. **LLM Pass 1** — GPT-4o reads the database schema and converts the question into a safe PostgreSQL SELECT query
2. **SQL Validator** — blocks any destructive keywords before the query touches the database
3. **Query Executor** — runs the validated SQL against PostgreSQL with a 30-second timeout
4. **LLM Pass 2** — GPT-4o reads the raw results and writes an executive summary, 3 key insights, and a detailed narrative
5. **Report renderer** — selects the right chart type automatically (bar, line, pie, horizontal bar) and generates a downloadable PDF

---

## Project structure

```
├── app.py                 # FastAPI application — all API endpoints
├── schema_loader.py       # Connects to DB, builds schema context string
├── sql_generator.py       # LLM Pass 1 — natural language → SQL
├── query_executor.py      # Runs SQL, formats results
├── report_generator.py    # LLM Pass 2 — results → report + chart + PDF
├── main.py                # CLI entry point for local testing
├── index.html             # Web UI — served at /
├── requirements.txt       # Python dependencies
├── Procfile               # Railway start command
├── railway.toml           # Railway config
├── .env.example           # Environment variable template
└── .gitignore
```

---

## Local setup

### Prerequisites

- Python 3.11+
- PostgreSQL access (credentials below)
- OpenAI API key

### 1. Clone the repo

```bash
git clone https://github.com/Aditya290698/Report-Generation-System.git
cd Report-Generation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file

Copy the example and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-openai-key-here

DB_HOST=54.153.162.250
DB_PORT=5432
DB_NAME=test
DB_USER=postgres
DB_PASSWORD=1234

APP_USERNAME=admin
APP_PASSWORD=admin

ALLOWED_ORIGINS=*
```

### 4. Run via CLI (quickest way to test)

```bash
# Single question
python main.py --question "How many products are in each category?"

# Single question with PDF export
python main.py --question "Show monthly revenue from mx51 transactions" --pdf

# Interactive mode — type questions one by one
python main.py
```

**Sample output:**

```
────────────────────────────────────────────────────────────
Question: How many products are in each category?
────────────────────────────────────────────────────────────
Generating SQL (Pass 1)...
SQL:
SELECT pc."name" AS category_name, COUNT(p."id") AS product_count
FROM product_categories pc
LEFT JOIN products p ON p."productCategoryId" = pc."id"
GROUP BY pc."name"
ORDER BY product_count DESC

Query returned 72 rows.

Generating report (Pass 2)...

════════════════════════════════════════════════════════════
REPORT
════════════════════════════════════════════════════════════

Summary:
Spirits leads all categories with 167 products, followed by Domestic Red
Wine with 155. The catalogue spans 72 distinct categories.

Key Insights:
  • Spirits is the largest category with 167 products
  • Domestic Red Wine and Domestic White Wine together account for 277 products
  • 4 categories have only 1 product each, indicating areas for expansion

Chart recommendation: horizontalBar
```

### 5. Run the API locally

```bash
python app.py
```

Server starts at `http://localhost:8000`

Open the web UI: http://localhost:8000

Open API docs: http://localhost:8000/docs

---

## API reference

Base URL (production): `https://web-production-a461.up.railway.app`

Base URL (local): `http://localhost:8000`

All protected endpoints require HTTP Basic Auth:

```
Authorization: Basic base64(username:password)
```

---

### `POST /report`

Generate a full report from a natural language question.

**Auth required:** Yes

**Request body:**

```json
{
  "question": "Show monthly revenue from mx51 transactions",
  "export_pdf": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | Natural language question (5–500 chars) |
| `export_pdf` | boolean | No | Generate a downloadable PDF (default: false) |

**Response:**

```json
{
  "report_id": "3f8a2c1d-...",
  "question": "Show monthly revenue from mx51 transactions",
  "sql": "SELECT TO_CHAR(mx51.\"finalisedAt\" AT TIME ZONE 'UTC', 'YYYY-MM') ...",
  "summary": "Revenue peaked in February 2026 at AUD 4,820...",
  "narrative": "Analysis of mx51 transaction data reveals...",
  "key_insights": [
    "February 2026 was the highest revenue month at AUD 4,820",
    "Average monthly revenue across all approved transactions is AUD 3,210",
    "Transaction volume increased 34% month-on-month from January to February"
  ],
  "chart": {
    "chart_type": "line",
    "title": "Monthly Revenue From Mx51 Transactions",
    "labels": ["2026-01", "2026-02", "2026-03"],
    "datasets": [{ "label": "Final Amount", "data": [2840.50, 4820.00, 3100.75] }],
    "x_label": "Month",
    "y_label": "Final Amount",
    "reasoning": "Time-based label column — line chart shows trends over time"
  },
  "row_count": 3,
  "has_data": true,
  "generated_at": "2026-04-04T12:00:00Z",
  "pdf_url": "/report/pdf/3f8a2c1d-..."
}
```

**Error responses:**

| Status | Code | Meaning |
|---|---|---|
| 401 | — | Invalid credentials |
| 404 | `no_data` | Query ran but returned no results |
| 422 | `sql_generation_failed` | LLM could not generate valid SQL |
| 500 | `query_execution_failed` | Database error |
| 500 | `report_generation_failed` | LLM report writing failed |

---

### `GET /report/pdf/{report_id}`

Download the PDF for a previously generated report.

**Auth required:** Yes

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `report_id` | string | The `report_id` from `POST /report` response |

**Response:** PDF file (`application/pdf`)

**Note:** You must have set `export_pdf: true` in the original `POST /report` request.

```bash
curl -u admin:admin \
  https://web-production-a461.up.railway.app/report/pdf/3f8a2c1d-... \
  --output report.pdf
```
---

### `GET /health`

Check system status. No auth required.

**Response:**

```json
{
  "status": "ok",
  "database": "connected",
  "schema_size": 16571,
  "openai": "api_key_present",
  "timestamp": "2026-04-04T12:00:00Z"
}
```

Returns `503` with `status: "degraded"` if the database or OpenAI key is unavailable.

---

### `GET /tables`

List all 12 reporting tables. No auth required.

**Response:**

```json
{
  "tables": ["orders", "order_items", "products", "..."],
  "count": 12
}
```

---

### `GET /docs`

Interactive Swagger UI — test all endpoints directly in the browser. No auth required.

---

## Sample questions to try

**Sales:**
```
What is the total purchase amount in mx51 transactions on 18th Feb 2026?
What is the total mx51 revenue from January 2026 to March 2026?
Show monthly revenue from approved mx51 transactions
What is the total revenue from all completed Square checkouts?
What is the average transaction value from approved mx51 transactions?
```

**Products:**
```
How many products are in each category?
What are the top 10 most expensive products by cost price?
How many active vs inactive products are there?
Which products have low stock below 10 units?
Show the top 10 products with the highest GP margin
```

**Customers:**
```
How many customers do we have in total?
Show me customers grouped by state
Which customers have an outstanding balance greater than zero?
How many customers have account payment enabled?
What is the total outstanding balance across all customers?
```

**Orders (config data — orders table empty in dev DB):**
```
What order types are configured in the system?
What order statuses are available?
What payment methods are available in the system?
```

---

## Testing with curl

```bash
# Health check
curl https://web-production-a461.up.railway.app/health

# Generate a report
curl -X POST https://web-production-a461.up.railway.app/report \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic YWRtaW46YWRtaW4=" \
  -d '{"question": "How many products are in each category?", "export_pdf": false}'

# Generate with PDF
curl -X POST https://web-production-a461.up.railway.app/report \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic YWRtaW46YWRtaW4=" \
  -d '{"question": "Show monthly revenue from mx51 transactions", "export_pdf": true}'
```

`YWRtaW46YWRtaW4=` is `admin:admin` base64 encoded.

---

## Tech stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o |
| Database | PostgreSQL (psycopg2) |
| PDF generation | ReportLab + matplotlib |
| Web UI | Vanilla HTML/CSS/JS + Chart.js |
| Deployment | Railway |

---

## Deployment (Railway)

1. Push code to GitHub
2. Connect repo to Railway
3. Add environment variables in Railway → Variables
4. Railway auto-deploys on every push

Required Railway variables:
```
OPENAI_API_KEY
DB_HOST
DB_PORT
DB_NAME
DB_USER
DB_PASSWORD
APP_USERNAME
APP_PASSWORD
ALLOWED_ORIGINS
```