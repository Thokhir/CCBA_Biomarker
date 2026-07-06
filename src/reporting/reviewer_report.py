"""Reviewer package: a short reviewer-focused PDF (methods, results, caveats,
reproducibility) plus a machine-readable JSON summary of the whole platform.

Distinct in scope from the manuscript Final_Report (comprehensive, all
modules, full tables/figures): this is deliberately short and leads with
statistical caveats/limitations, since that's what a reviewer needs first.
"""
import json
from datetime import datetime
from pathlib import Path

try:
    from . import content_builder, pdf_renderer
except ImportError:
    import content_builder
    import pdf_renderer


def export_reviewer_pdf(summary, out_path: Path) -> None:
    content = content_builder.build_reviewer_content(summary)
    pdf_renderer.render_pdf(content, out_path)


def export_json_summary(summary, out_path: Path) -> None:
    json_summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "training_samples": summary.model_metadata["training_samples"],
            "training_features": summary.model_metadata["training_features"],
            "training_date": summary.model_metadata["training_date"],
            "sklearn_version": summary.model_metadata["sklearn_version"],
        },
        "external_validation": {
            "roc_auc": summary.overall_metrics["ROC_AUC"],
            "sensitivity": summary.overall_metrics["Sensitivity"],
            "specificity": summary.overall_metrics["Specificity"],
            "n_samples": int(summary.overall_metrics["Samples"]),
        },
        "top_biomarker_shap": summary.shap_global_importance.iloc[0]["gene_name"],
        "explainability_ranking_correlations": summary.explainability_summary["ranking_correlations"],
        "biological_validation": {
            "string_network_edges": summary.biological_validation_summary["string_network"]["n_edges"],
            "string_hub_genes": summary.biological_validation_summary["string_network"]["hub_genes"],
        },
        "survival": {
            "cohort_size": summary.survival_summary["cohort_size"],
            "n_deaths": summary.survival_summary["n_deaths"],
            "km_significant_genes": summary.survival_summary["km_significant_genes"],
        },
        "top_drug_target": summary.therapeutic_priority.iloc[0]["gene_name"],
        "top_repurposing_candidate": {
            "gene_name": summary.drug_repurposing.iloc[0]["gene_name"],
            "drug_name": summary.drug_repurposing.iloc[0]["drug_name"],
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4, default=str)
