"""
report_generator.py
-------------------
LLM Pass 2 — Takes the raw query results from query_executor.py and:

1. Analyses the data structure to recommend the best chart type
2. Calls OpenAI to write a professional business report narrative
3. Bundles everything into a ReportOutput object that the UI layer
   (web app) uses to render the chart and export the PDF

Chart selection logic:
- Time series data (dates/months)  → Line chart
- Category comparisons (<=8 items) → Bar chart
- Category comparisons (>8 items)  → Horizontal bar chart
- Part-of-whole / proportions      → Pie chart
- Single aggregate value           → Metric card (no chart)
- Ranking / top N                  → Bar chart
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
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
# SECTION 1 — Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChartConfig:
    """
    Describes the chart the UI should render.
    Passed directly to the frontend — all fields map to Chart.js options.
    """
    chart_type: str          # "line" | "bar" | "horizontalBar" | "pie" | "none"
    title: str               # Chart title shown above the chart
    labels: list             # X-axis labels or pie segment labels
    datasets: list           # List of {label, data, backgroundColor} dicts
    x_label: str = ""        # X-axis label (for bar/line)
    y_label: str = ""        # Y-axis label (for bar/line)
    reasoning: str = ""      # Why this chart type was chosen (shown in UI)


@dataclass
class ReportOutput:
    """
    The complete output of Pass 2 — everything the UI needs to:
    - Render the report narrative
    - Draw the chart
    - Generate the PDF
    """
    question: str            # Original user question
    sql: str                 # The SQL that was executed
    summary: str             # 2-3 sentence executive summary
    narrative: str           # Full report text (markdown)
    key_insights: list       # Bullet point insights extracted by LLM
    chart: ChartConfig       # Chart configuration for the frontend
    row_count: int           # Number of rows returned
    has_data: bool           # False if query returned no results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Chart type selector
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that suggest time-series data in column names
TIME_KEYWORDS = [
    "date", "month", "week", "year", "day", "period",
    "time", "created", "finalised", "ordered"
]

# Keywords that suggest proportional / share data
PROPORTION_KEYWORDS = [
    "percentage", "percent", "share", "ratio", "proportion", "pct"
]

# Keywords that suggest a ranking / top-N query
RANKING_KEYWORDS = [
    "top", "best", "most", "highest", "lowest", "rank", "count", "total"
]


def detect_chart_type(
    columns: list,
    rows: list,
    user_question: str,
) -> tuple[str, str]:
    """
    Analyses the query result shape and the user's question to pick
    the most appropriate chart type.

    Decision logic (in priority order):
    1. Single row + single value  → no chart (metric card)
    2. Time-based label column    → line chart
    3. Proportion keywords        → pie chart (only if <=8 categories)
    4. <=8 categories             → bar chart
    5. >8 categories              → horizontal bar (labels need more space)

    Args:
        columns:       List of column names from the query result.
        rows:          List of row dicts from the query result.
        user_question: The original natural language question.

    Returns:
        Tuple of (chart_type, reasoning_string).
    """
    row_count = len(rows)
    col_count = len(columns)
    question_lower = user_question.lower()

    # ── Single aggregate value — just show a metric card ─────────────────────
    if row_count == 1 and col_count == 1:
        return "none", (
            "Single aggregate value — displayed as a metric card. "
            "No chart needed."
        )

    # ── Identify label column and value column ────────────────────────────────
    # Heuristic: label column is text, value column is numeric.
    # First column is usually the label (category/date), rest are values.
    label_col = columns[0] if columns else ""
    label_col_lower = label_col.lower()

    # ── Check for time series ─────────────────────────────────────────────────
    is_time_series = any(kw in label_col_lower for kw in TIME_KEYWORDS)
    if is_time_series:
        return "line", (
            f"Column '{label_col}' contains time-based data — "
            "line chart shows trends over time most clearly."
        )

    # ── Check for proportion / percentage data ────────────────────────────────
    is_proportion = (
        any(kw in question_lower for kw in PROPORTION_KEYWORDS)
        or any(kw in col.lower() for col in columns for kw in PROPORTION_KEYWORDS)
    )
    if is_proportion and row_count <= 8:
        return "pie", (
            "Data represents proportions or percentages — "
            "pie chart shows part-of-whole relationships clearly."
        )

    # ── Category comparison ───────────────────────────────────────────────────
    if row_count <= 8:
        return "bar", (
            f"{row_count} categories — vertical bar chart "
            "allows easy side-by-side comparison."
        )

    # ── Many categories — horizontal bar gives labels room ───────────────────
    return "horizontalBar", (
        f"{row_count} categories — horizontal bar chart used so "
        "category labels have enough space to be readable."
    )


def build_chart_config(
    columns: list,
    rows: list,
    chart_type: str,
    reasoning: str,
    user_question: str,
) -> ChartConfig:
    """
    Builds the ChartConfig object from query results.

    Extracts labels (first column) and numeric values (remaining columns),
    assigns colours, and returns a ChartConfig ready for the frontend.

    Args:
        columns:       Column names from the query.
        rows:          Row dicts from the query.
        chart_type:    One of "line", "bar", "horizontalBar", "pie", "none".
        reasoning:     Why this chart type was chosen.
        user_question: Original question — used to derive chart title.

    Returns:
        ChartConfig instance.
    """
    if chart_type == "none" or not rows:
        return ChartConfig(
            chart_type="none",
            title="",
            labels=[],
            datasets=[],
            reasoning=reasoning,
        )

    # ── Extract labels and value columns ─────────────────────────────────────
    label_col = columns[0]
    value_cols = columns[1:] if len(columns) > 1 else columns

    labels = [str(row.get(label_col, "")) for row in rows]

    # Colour palette — distinct colours for up to 8 datasets/segments
    COLORS = [
        "#378ADD",  # blue
        "#1D9E75",  # teal
        "#EF9F27",  # amber
        "#D85A30",  # coral
        "#7F77DD",  # purple
        "#639922",  # green
        "#D4537E",  # pink
        "#888780",  # gray
    ]

    datasets = []
    for i, val_col in enumerate(value_cols):
        color = COLORS[i % len(COLORS)]

        # Extract numeric values — default to 0 for nulls
        data = []
        for row in rows:
            val = row.get(val_col)
            try:
                data.append(round(float(val), 2) if val is not None else 0)
            except (ValueError, TypeError):
                data.append(0)

        # Pie charts need a list of colours (one per segment)
        bg_color = (
            [COLORS[j % len(COLORS)] for j in range(len(rows))]
            if chart_type == "pie"
            else color
        )

        datasets.append({
            "label": val_col.replace("_", " ").title(),
            "data": data,
            "backgroundColor": bg_color,
            "borderColor": color,
            "borderWidth": 1,
            "fill": chart_type == "line",  # Fill area under line charts
        })

    # Derive axis labels from column names
    x_label = label_col.replace("_", " ").title()
    y_label = value_cols[0].replace("_", " ").title() if value_cols else ""

    # Chart title — clean up the question into a title
    title = user_question.strip().rstrip("?").title()

    return ChartConfig(
        chart_type=chart_type,
        title=title,
        labels=labels,
        datasets=datasets,
        x_label=x_label,
        y_label=y_label,
        reasoning=reasoning,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LLM Pass 2 system prompt
# ─────────────────────────────────────────────────────────────────────────────

REPORT_WRITER_SYSTEM_PROMPT = """
You are a professional business analyst writing reports for a Point of Sale system.

You will be given:
1. A question the user asked
2. The raw data returned from the database

Your job is to write a clear, professional business report.

OUTPUT FORMAT — respond with valid JSON only, no markdown, no backticks:
{
  "summary": "2-3 sentence executive summary of the key finding",
  "narrative": "Full report in plain text. 3-5 paragraphs. Professional tone. Include specific numbers from the data.",
  "key_insights": [
    "Insight 1 — specific, data-backed observation",
    "Insight 2 — specific, data-backed observation",
    "Insight 3 — specific, data-backed observation"
  ]
}

RULES:
1. Always reference specific numbers from the data — never be vague.
2. If data shows a clear winner/leader, name it explicitly.
3. If data is empty or all zeros, say so clearly and suggest the date range may have no data.
4. Keep the narrative professional but readable — avoid jargon.
5. key_insights must be exactly 3 bullet points, each starting with a specific finding.
6. Currency values are in AUD unless stated otherwise.
7. Never invent data that isn't in the results provided.
8. Output ONLY valid JSON — no preamble, no explanation outside the JSON.
9. For sales/revenue reports, always mention the time period covered if visible in the data.
10. If a value is NULL or N/A in the data, note it as unavailable — do not treat it as zero.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    user_question: str,
    sql: str,
    query_result: dict,
    client: OpenAI | None = None,
) -> ReportOutput:
    """
    LLM Pass 2 — the report writer.

    Takes the raw query result and generates:
    - A chart configuration (type auto-selected based on data shape)
    - An executive summary
    - A full narrative report
    - Key bullet-point insights

    Args:
        user_question: Original natural language question.
        sql:           The SQL that was executed (included in PDF output).
        query_result:  Dict with columns, rows, row_count from execute_query().
        client:        Optional pre-built OpenAI client.

    Returns:
        ReportOutput — everything the UI needs to render and export.

    Raises:
        ValueError: If query_result is malformed.
        openai.APIError: On API failures.
    """
    columns   = query_result.get("columns", [])
    rows      = query_result.get("rows", [])
    row_count = query_result.get("row_count", 0)

    logger.info(
        "Generating report for '%s' — %d rows, %d columns.",
        user_question, row_count, len(columns)
    )

    # ── Handle empty data ─────────────────────────────────────────────────────
    if not rows or row_count == 0:
        return ReportOutput(
            question=user_question,
            sql=sql,
            summary="No data was found matching your query criteria.",
            narrative=(
                "The query executed successfully but returned no results. "
                "This typically means there are no records in the database "
                "matching the specified filters, such as no transactions "
                "in the selected date range."
            ),
            key_insights=[
                "No data available for this query.",
                "Check that the date range contains transaction data.",
                "The relevant tables may not have been populated yet.",
            ],
            chart=ChartConfig(
                chart_type="none", title="", labels=[], datasets=[], reasoning=""
            ),
            row_count=0,
            has_data=False,
        )

    # ── Handle all-NULL result (aggregation on empty dataset) ─────────────────
    # When SUM/AVG/COUNT runs on rows that match no records, PostgreSQL returns
    # one row with NULL. We catch this here before sending to LLM so it doesn't
    # hallucinate explanations like "system downtime".
    if row_count == 1:
        all_values = list(rows[0].values())
        if all(v is None for v in all_values):
            logger.info("All-NULL result detected — aggregation matched no records.")
            return ReportOutput(
                question=user_question,
                sql=sql,
                summary=(
                    "The query returned no matching records for the specified filters. "
                    "This means no transactions matched all the conditions in your question."
                ),
                narrative=(
                    "The SQL query ran successfully but the aggregation (SUM/COUNT/AVG) "
                    "returned NULL, which means zero rows matched all the filter conditions combined. "
                    "For mx51 transactions, only 2 out of 200 records have status APPROVED. "
                    "If you are filtering by both date and APPROVED status, there may be no APPROVED "
                    "transactions on that specific date. "
                    "Try asking for the total purchase amount without specifying a status, "
                    "or broaden the date range to find records."
                ),
                key_insights=[
                    "No records matched all filter conditions in the query.",
                    "For mx51 transactions, only 2 of 200 records are APPROVED status.",
                    "Try broadening the date range or removing the status filter to find data.",
                ],
                chart=ChartConfig(
                    chart_type="none", title="", labels=[], datasets=[], reasoning=""
                ),
                row_count=0,
                has_data=False,
            )

    # ── Step 1: Select chart type ─────────────────────────────────────────────
    chart_type, reasoning = detect_chart_type(columns, rows, user_question)
    logger.info("Chart type selected: %s — %s", chart_type, reasoning)

    # ── Step 2: Build chart config ────────────────────────────────────────────
    chart = build_chart_config(columns, rows, chart_type, reasoning, user_question)

    # ── Step 3: Format data for LLM ───────────────────────────────────────────
    # Build a readable table of the results to inject into the prompt.
    # Cap at 50 rows — the LLM doesn't need all 2000 rows to write a summary.
    MAX_ROWS_FOR_LLM = 50
    display_rows = rows[:MAX_ROWS_FOR_LLM]

    header = " | ".join(columns)
    separator = "-" * len(header)
    data_lines = [header, separator]
    for row in display_rows:
        vals = []
        for col in columns:
            v = row.get(col)
            if v is None:
                vals.append("N/A")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        data_lines.append(" | ".join(vals))

    if len(rows) > MAX_ROWS_FOR_LLM:
        data_lines.append(f"... and {len(rows) - MAX_ROWS_FOR_LLM} more rows")

    formatted_data = "\n".join(data_lines)

    user_message = f"""
Question: {user_question}

Data returned ({row_count} rows):
{formatted_data}

Write the business report as JSON following the required format.
"""

    # ── Step 4: Call OpenAI ───────────────────────────────────────────────────
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)

    try:
        logger.info("Calling OpenAI API for report generation...")
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,
            temperature=0.3,    # Lower temperature = more consistent, factual output
            messages=[
                {"role": "system", "content": REPORT_WRITER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )

        raw_content = response.choices[0].message.content.strip()
        logger.info("Report content received from OpenAI.")

    except APIConnectionError as e:
        logger.error("Network error calling OpenAI for report: %s", e)
        raise
    except RateLimitError as e:
        logger.error("OpenAI rate limit hit during report generation: %s", e)
        raise
    except APIStatusError as e:
        logger.error("OpenAI API error during report: status=%s", e.status_code)
        raise

    # ── Step 5: Parse the JSON response ───────────────────────────────────────
    # Strip markdown fences if the LLM added them despite instructions
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM report JSON: %s\nRaw: %s", e, raw_content)
        # Graceful fallback — use raw text as narrative
        parsed = {
            "summary": "Report generated successfully.",
            "narrative": raw_content,
            "key_insights": ["See narrative for details.", "", ""],
        }

    # ── Step 6: Build and return ReportOutput ─────────────────────────────────
    return ReportOutput(
        question=user_question,
        sql=sql,
        summary=parsed.get("summary", ""),
        narrative=parsed.get("narrative", ""),
        key_insights=parsed.get("key_insights", []),
        chart=chart,
        row_count=row_count,
        has_data=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PDF export
# ─────────────────────────────────────────────────────────────────────────────

def render_chart_to_image(chart: ChartConfig) -> str | None:
    """
    Renders the chart to a temporary PNG image using matplotlib.
    Returns the temp file path, or None if chart_type is 'none'.

    Supports: bar, horizontalBar, line, pie.
    The image is saved to a temp file so ReportLab can embed it.

    Args:
        chart: ChartConfig built by build_chart_config().

    Returns:
        Path to the temporary PNG file, or None if no chart.
    """
    if chart.chart_type == "none" or not chart.labels or not chart.datasets:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")           # Non-interactive backend — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import tempfile
    except ImportError:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install it with: pip install matplotlib"
        )

    labels   = chart.labels
    datasets = chart.datasets
    ct       = chart.chart_type

    # ── Figure sizing ─────────────────────────────────────────────────────────
    # Horizontal bar charts need more height when there are many categories
    if ct == "horizontalBar":
        fig_height = max(6, min(len(labels) * 0.35, 20))
        fig, ax = plt.subplots(figsize=(10, fig_height))
    elif ct == "pie":
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    # Colour palette — consistent with ChartConfig
    COLORS = [
        "#378ADD", "#1D9E75", "#EF9F27", "#D85A30",
        "#7F77DD", "#639922", "#D4537E", "#888780",
    ]

    # ── Draw the chart ────────────────────────────────────────────────────────
    if ct in ("bar", "horizontalBar"):
        values = datasets[0]["data"] if datasets else []
        bar_colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

        if ct == "bar":
            bars = ax.bar(range(len(labels)), values, color=bar_colors,
                          width=0.6, edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(chart.y_label, fontsize=9, color="#555555")
            ax.set_xlabel(chart.x_label, fontsize=9, color="#555555")
            # Value labels on top of each bar
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(values) * 0.01,
                        str(int(val)) if val == int(val) else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7, color="#333333"
                    )
        else:
            # Horizontal bar — sort by value descending for readability
            paired = sorted(zip(values, labels), reverse=True)
            values_sorted  = [p[0] for p in paired]
            labels_sorted  = [p[1] for p in paired]
            bar_colors_sorted = [COLORS[i % len(COLORS)] for i in range(len(labels_sorted))]

            bars = ax.barh(range(len(labels_sorted)), values_sorted,
                           color=bar_colors_sorted, edgecolor="white",
                           linewidth=0.5, height=0.7)
            ax.set_yticks(range(len(labels_sorted)))
            ax.set_yticklabels(labels_sorted, fontsize=8)
            ax.set_xlabel(chart.y_label, fontsize=9, color="#555555")
            ax.invert_yaxis()   # Highest value at top
            # Value labels at end of each bar
            for bar, val in zip(bars, values_sorted):
                if val > 0:
                    ax.text(
                        val + max(values_sorted) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(val)) if val == int(val) else f"{val:.1f}",
                        va="center", fontsize=7, color="#333333"
                    )

    elif ct == "line":
        for i, ds in enumerate(datasets):
            color = COLORS[i % len(COLORS)]
            ax.plot(range(len(labels)), ds["data"], color=color,
                    linewidth=2, marker="o", markersize=4, label=ds["label"])
            ax.fill_between(range(len(labels)), ds["data"],
                            alpha=0.08, color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(chart.y_label, fontsize=9, color="#555555")
        ax.set_xlabel(chart.x_label, fontsize=9, color="#555555")
        if len(datasets) > 1:
            ax.legend(fontsize=8)

    elif ct == "pie":
        values = datasets[0]["data"] if datasets else []
        pie_colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,            # Use legend instead — labels on pie get crowded
            colors=pie_colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=140,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        for t in autotexts:
            t.set_fontsize(7)
            t.set_color("#333333")
        # Legend with truncated labels
        legend_labels = [
            f"{l[:28]}..." if len(l) > 28 else l
            for l in labels
        ]
        ax.legend(wedges, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=7,
                  frameon=False)
        ax.axis("equal")

    # ── Styling ───────────────────────────────────────────────────────────────
    ax.set_title(chart.title, fontsize=12, fontweight="bold",
                 color="#1a1a2e", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.tick_params(colors="#555555")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.5,
                  linestyle="--") if ct != "pie" else None

    plt.tight_layout(pad=1.5)

    # ── Save to temp file ─────────────────────────────────────────────────────
    # Use delete=False so we control deletion after ReportLab is done.
    # Explicitly close the file handle before saving — on Windows, an open
    # NamedTemporaryFile handle blocks any other process from reading the file.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()     # Release the handle immediately so matplotlib can write freely

    plt.savefig(tmp_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Chart rendered to temp file: %s", tmp_path)
    return tmp_path


def export_report_to_pdf(report: ReportOutput, output_path: str) -> str:
    """
    Exports the ReportOutput to a professional PDF file using ReportLab.

    The PDF includes:
    - Header with report title and date
    - Executive summary box
    - Key insights as bullet points
    - Full narrative
    - Embedded matplotlib chart image
    - SQL query used (in monospace font)
    - Footer

    Args:
        report:      ReportOutput from generate_report().
        output_path: File path for the PDF.

    Returns:
        The output_path string.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable, Image as RLImage, KeepTogether
        )
        from datetime import datetime
        import os
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF export. "
            "Install it with: pip install reportlab"
        )

    logger.info("Exporting report to PDF: %s", output_path)

    # ── Render chart to image first ───────────────────────────────────────────
    chart_image_path = None
    if report.chart.chart_type != "none":
        try:
            chart_image_path = render_chart_to_image(report.chart)
        except Exception as e:
            logger.warning("Chart rendering failed, skipping: %s", e)

    # ── Document setup ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    PAGE_W = A4[0] - 4*cm       # Usable width inside margins

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"],
        fontSize=20, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#666666"), spaceAfter=20,
    )
    section_style = ParagraphStyle(
        "SectionHeader", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#185FA5"),
        spaceBefore=16, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=16, textColor=colors.HexColor("#2c2c2c"),
    )
    insight_style = ParagraphStyle(
        "Insight", parent=styles["Normal"],
        fontSize=10, leading=16, leftIndent=12,
        textColor=colors.HexColor("#2c2c2c"),
    )
    summary_style = ParagraphStyle(
        "Summary", parent=styles["Normal"],
        fontSize=11, leading=18, textColor=colors.HexColor("#1a1a2e"),
        leftIndent=12, rightIndent=12,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#888888"),
        alignment=1,            # Centre aligned
        spaceAfter=8,
    )

    # ── Build story ───────────────────────────────────────────────────────────
    story = []
    now = datetime.now().strftime("%d %B %Y, %H:%M")

    # Title & date
    story.append(Paragraph(report.question.title(), title_style))
    story.append(Paragraph(f"Generated on {now}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 12))

    # Executive summary
    story.append(Paragraph("Executive Summary", section_style))
    summary_tbl = Table([[Paragraph(report.summary, summary_style)]],
                        colWidths=[PAGE_W])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#EEF4FB")),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#185FA5")),
        ("TOPPADDING",    (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ("RIGHTPADDING",  (0,0), (-1,-1), 14),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 16))

    # Key insights
    story.append(Paragraph("Key Insights", section_style))
    for insight in report.key_insights:
        if insight:
            story.append(Paragraph(f"• {insight}", insight_style))
            story.append(Spacer(1, 4))
    story.append(Spacer(1, 12))

    # ── Chart image ───────────────────────────────────────────────────────────
    if chart_image_path and os.path.exists(chart_image_path):
        story.append(Paragraph("Visualisation", section_style))

        # Scale image to fit page width, preserving aspect ratio
        from PIL import Image as PILImage
        with PILImage.open(chart_image_path) as img:
            img_w, img_h = img.size

        max_w   = PAGE_W
        scale   = max_w / img_w
        img_h_scaled = img_h * scale

        # For very tall charts (horizontal bar with many categories)
        # cap at 2/3 of A4 page height to avoid a single chart filling a page
        MAX_H = A4[1] * 0.65
        if img_h_scaled > MAX_H:
            scale        = MAX_H / img_h
            max_w        = img_w * scale
            img_h_scaled = MAX_H

        rl_img = RLImage(chart_image_path, width=max_w, height=img_h_scaled)
        story.append(KeepTogether([
            rl_img,
            Paragraph(
                f"Chart type: {report.chart.chart_type.replace('Bar', ' Bar').title()} — "
                f"{report.chart.reasoning}",
                caption_style,
            ),
        ]))
        story.append(Spacer(1, 16))

    # Full narrative
    story.append(Paragraph("Detailed Analysis", section_style))
    for para in report.narrative.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 8))
    story.append(Spacer(1, 12))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated by POS LLM Reporting System. "
        f"Data source: PostgreSQL. Rows returned: {report.row_count}.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#999999")),
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(story)

    # Clean up temp chart image
    # On Windows, files can stay locked after writing — retry with a short
    # delay to let the OS release the handle before deleting.
    if chart_image_path and os.path.exists(chart_image_path):
        import time
        for attempt in range(3):
            try:
                time.sleep(0.2)
                os.unlink(chart_image_path)
                logger.info("Temp chart image cleaned up.")
                break
            except PermissionError:
                if attempt == 2:
                    logger.warning(
                        "Could not delete temp chart file (Windows lock): %s. "
                        "It will be cleaned on next system restart.",
                        chart_image_path,
                    )

    logger.info("PDF exported successfully: %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from schema_loader import load_schema
    from sql_generator import generate_sql_with_retry
    from query_executor import execute_query

    test_question = "How many products are in each category?"

    try:
        print("Loading schema...")
        conn, schema = load_schema()

        print(f"Generating SQL for: '{test_question}'")
        sql = generate_sql_with_retry(test_question, schema)
        print(f"SQL: {sql}\n")

        print("Executing query...")
        result = execute_query(conn, sql)

        print("Generating report (Pass 2)...")
        report = generate_report(test_question, sql, result)

        print("\n" + "="*60)
        print("REPORT OUTPUT")
        print("="*60)
        print(f"\nQuestion:  {report.question}")
        print(f"Has data:  {report.has_data}")
        print(f"Rows:      {report.row_count}")
        print(f"Chart:     {report.chart.chart_type} — {report.chart.reasoning}")
        print(f"\nSummary:\n{report.summary}")
        print(f"\nKey Insights:")
        for i in report.key_insights:
            print(f"  • {i}")
        print(f"\nNarrative:\n{report.narrative}")

        # Export PDF
        pdf_path = "test_report.pdf"
        export_report_to_pdf(report, pdf_path)
        print(f"\nPDF exported to: {pdf_path}")

        conn.close()

    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        raise