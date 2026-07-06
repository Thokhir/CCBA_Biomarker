"""Single-patient clinical decision-support PDF report.

SCOPE NOTE: this generates a PDF for ONE patient only (prediction,
explanation, relevant biomarker/pathway/drug context, risk placement if
applicable, a model-validation footnote). It is explicitly NOT the
whole-platform academic/reviewer package (manuscript tables, supplementary
tables, whole-cohort findings) - that is Module 14's separate scope, built
later and separately. Do not extend this function to cover multi-patient or
whole-platform content; add a new module for that instead.
"""
import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results"

DISCLAIMER = "Research use only. Not a diagnostic device. For investigational purposes."
SURVIVAL_CAVEAT = (
    "Survival findings are exploratory and hypothesis-generating, based on 34 patients with only "
    "18 observed deaths. This risk placement is not a validated clinical-grade prognostic output."
)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=16, spaceAfter=6, textColor=colors.HexColor("#0F6E68")))
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=12, spaceBefore=12, spaceAfter=6,
                               textColor=colors.HexColor("#0F6E68")))
    styles.add(ParagraphStyle(name="Disclaimer", fontSize=8, textColor=colors.grey))
    return styles


def _fig_to_image(fig, width=5.5 * inch):
    png_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
    return Image(io.BytesIO(png_bytes), width=width, height=width * 500 / 900)


def generate_patient_report(session, biomarker_annotation, therapeutic_priority, drug_repurposing,
                             overall_metrics, waterfall_fig) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = _styles()
    story = []

    # Header
    story.append(Paragraph("CCA-BDP Clinical Decision Support Report", styles["ReportTitle"]))
    story.append(Paragraph(DISCLAIMER, styles["Disclaimer"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Sample ID: <b>{session.sample_id}</b> &nbsp;&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"],
    ))

    # Prediction summary
    story.append(Paragraph("Prediction Summary", styles["SectionHeader"]))
    label_text = "Tumor" if session.prediction.predicted_label == 1 else "Normal"
    summary_data = [
        ["Predicted label", label_text],
        ["Tumor probability", f"{session.prediction.tumor_probability:.1%}"],
        ["Confidence", session.prediction.confidence],
        ["Genes provided", str(len(session.patient_matrix.genes_provided))],
        ["Genes imputed (not measured)",
         ", ".join(session.patient_matrix.genes_imputed) or "None"],
    ]
    table = Table(summary_data, colWidths=[2.2 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F7F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E6E5")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    if session.patient_matrix.genes_imputed:
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Note: predictions based on a partial panel are weaker evidence than a complete panel.",
            styles["Disclaimer"],
        ))

    # Top contributing biomarkers
    story.append(Paragraph("Top Contributing Biomarkers", styles["SectionHeader"]))
    top_contributions = session.explanation.contributions.head(8)
    contrib_data = [["Gene", "SHAP value", "Direction", "Measured?"]]
    for _, row in top_contributions.iterrows():
        direction = "Toward Tumor" if row["shap_value"] > 0 else "Toward Normal"
        measured = "Imputed" if row["is_imputed"] else "Measured"
        contrib_data.append([row["gene_name"], f"{row['shap_value']:+.4f}", direction, measured])
    contrib_table = Table(contrib_data, colWidths=[1.3 * inch, 1.3 * inch, 1.6 * inch, 1.3 * inch])
    contrib_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E68")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E6E5")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(contrib_table)

    if waterfall_fig is not None:
        story.append(Spacer(1, 8))
        story.append(_fig_to_image(waterfall_fig))

    # Biomarker context for top genes
    story.append(Paragraph("Biomarker Context (top contributing genes)", styles["SectionHeader"]))
    top_gene_names = top_contributions["gene_name"].head(5).tolist()
    context_rows = [["Gene", "STRING hub", "Top disease", "Priority rank", "Known drug"]]
    for gene in top_gene_names:
        bio_row = biomarker_annotation[biomarker_annotation["gene_name"] == gene]
        priority_row = therapeutic_priority[therapeutic_priority["gene_name"] == gene]
        drug_rows = drug_repurposing[drug_repurposing["gene_name"] == gene]

        is_hub = "Yes" if not bio_row.empty and bool(bio_row.iloc[0].get("is_hub")) else "No"
        top_disease = bio_row.iloc[0].get("top_disease_name", "-") if not bio_row.empty else "-"
        rank = str(int(priority_row.index[0]) + 1) if not priority_row.empty else "-"
        drug_name = drug_rows.iloc[0]["drug_name"] if not drug_rows.empty else "None known"
        context_rows.append([gene, is_hub, str(top_disease)[:30], rank, drug_name])

    context_table = Table(context_rows, colWidths=[0.9 * inch, 0.8 * inch, 1.8 * inch, 0.9 * inch, 1.1 * inch])
    context_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E68")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E6E5")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(context_table)

    # Survival context (only if risk placement was computed)
    if session.risk_placement is not None:
        story.append(Paragraph("Survival Context", styles["SectionHeader"]))
        story.append(Paragraph(f"Composite risk score (ITIH4-derived): {session.risk_placement:.3f}", styles["Normal"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(SURVIVAL_CAVEAT, styles["Disclaimer"]))

    # Model validation footnote
    story.append(Paragraph("Model Validation Context", styles["SectionHeader"]))
    story.append(Paragraph(
        f"This model's underlying diagnostic performance on independent external cohorts "
        f"(n={int(overall_metrics['Samples'])}): ROC-AUC={overall_metrics['ROC_AUC']:.3f}, "
        f"Sensitivity={overall_metrics['Sensitivity']:.3f}, Specificity={overall_metrics['Specificity']:.3f}.",
        styles["Normal"],
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(DISCLAIMER, styles["Disclaimer"]))

    doc.build(story)
    return buffer.getvalue()
