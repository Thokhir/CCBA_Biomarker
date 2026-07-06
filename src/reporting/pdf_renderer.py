"""Renders a ReportContent object to PDF via reportlab."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=18, spaceAfter=4, textColor=colors.HexColor("#0F6E68")))
    styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=12, spaceAfter=12, textColor=colors.grey))
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=13, spaceBefore=14, spaceAfter=6,
                               textColor=colors.HexColor("#0F6E68")))
    return styles


def _dataframe_to_table(df, styles) -> Table:
    data = [list(df.columns)] + df.round(4).astype(str).values.tolist()
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E68")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E6E5")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    return table


def render_pdf(content, out_path: Path) -> None:
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = _styles()
    story = [Paragraph(content.title, styles["ReportTitle"]),
             Paragraph(f"{content.subtitle} - Generated {content.generated_at}", styles["ReportSubtitle"])]

    for section in content.sections:
        story.append(Paragraph(section.title, styles["SectionHeader"]))
        for paragraph in section.paragraphs:
            story.append(Paragraph(paragraph, styles["Normal"]))
            story.append(Spacer(1, 4))

        if section.table is not None:
            story.append(Spacer(1, 4))
            story.append(_dataframe_to_table(section.table, styles))
            if section.table_caption:
                story.append(Paragraph(f"<i>{section.table_caption}</i>", styles["Normal"]))

        if section.figure_path:
            fig_file = RESULTS_DIR / section.figure_path
            if fig_file.exists():
                story.append(Spacer(1, 6))
                story.append(Image(str(fig_file), width=5 * inch, height=3.5 * inch, kind="proportional"))
                if section.figure_caption:
                    story.append(Paragraph(f"<i>{section.figure_caption}</i>", styles["Normal"]))

    doc.build(story)
