"""Automated reporting orchestrator.

Aggregates real headline results from every completed module (8-12) and
generates the full documentation set: manuscript-style report in three
formats (PDF/DOCX/HTML) built from one shared content structure, curated
manuscript tables + complete supplementary tables (Excel), a short
reviewer-focused PDF, and a machine-readable JSON summary.

Distinct in scope from Module 13's report_generator.py (single-patient
clinical PDF, generated on demand from a live upload): this module covers
the whole platform's research findings, generated once from the
already-computed results/ artifacts, not per-patient.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from . import data_aggregator, content_builder, pdf_renderer, docx_renderer, html_renderer
    from . import table_export, reviewer_report
except ImportError:
    import data_aggregator
    import content_builder
    import pdf_renderer
    import docx_renderer
    import html_renderer
    import table_export
    import reviewer_report

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "reports"


class ReportingEngine:
    def __init__(self):
        self.execution_start = datetime.now()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.summary = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ReportingEngine")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(OUTPUT_DIR / "execution_log.txt", mode="w")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def load_summary(self) -> None:
        self.logger.info("Aggregating results from Modules 8-12...")
        self.summary = data_aggregator.build_platform_summary()
        self.logger.info("Aggregated: %d training samples, ROC-AUC=%.3f, top biomarker=%s",
                          self.summary.model_metadata["training_samples"],
                          self.summary.overall_metrics["ROC_AUC"],
                          self.summary.shap_global_importance.iloc[0]["gene_name"])

    def export_final_report(self) -> None:
        self.logger.info("Building manuscript report content...")
        content = content_builder.build_manuscript_content(self.summary)

        self.logger.info("Rendering Final_Report.pdf...")
        pdf_renderer.render_pdf(content, OUTPUT_DIR / "Final_Report.pdf")

        self.logger.info("Rendering Final_Report.docx...")
        docx_renderer.render_docx(content, OUTPUT_DIR / "Final_Report.docx")

        self.logger.info("Rendering Final_Report.html...")
        html_renderer.render_html(content, OUTPUT_DIR / "Final_Report.html")

    def export_tables(self) -> None:
        self.logger.info("Exporting manuscript and supplementary table workbooks...")
        table_export.export_manuscript_tables(self.summary, OUTPUT_DIR / "Manuscript_Tables.xlsx")
        table_export.export_supplementary_tables(self.summary, OUTPUT_DIR / "Supplementary_Tables.xlsx")

    def export_reviewer_package(self) -> None:
        self.logger.info("Exporting reviewer PDF and JSON summary...")
        reviewer_report.export_reviewer_pdf(self.summary, OUTPUT_DIR / "Reviewer_Report.pdf")
        reviewer_report.export_json_summary(self.summary, OUTPUT_DIR / "JSON_Summary.json")

    def run(self) -> None:
        self.logger.info("Automated Reporting Engine Started")
        self.load_summary()
        self.export_final_report()
        self.export_tables()
        self.export_reviewer_package()
        self.logger.info("Execution Time: %s", datetime.now() - self.execution_start)


if __name__ == "__main__":
    ReportingEngine().run()
