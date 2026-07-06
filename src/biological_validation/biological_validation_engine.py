"""Biological validation orchestrator.

Answers: are the 20 clinical-panel genes biologically meaningful, or just
statistically associated? Runs GO/KEGG/Reactome/WikiPathways enrichment,
STRING PPI/hub analysis, Open Targets disease association, and PubMed
literature co-occurrence mining, then consolidates everything into one
per-gene summary table.

Findings are reported dynamically from whatever the real run produces
(per-library significance, actual edge/hub counts, actual novel-gene list)
rather than a fixed narrative decided during planning - a GO library, for
example, may reach significance even when KEGG/Reactome/WikiPathways don't.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from . import gene_ontology as go
    from . import pathway_enrichment as pe
    from . import protein_interaction as pi
    from . import disease_association as da
    from . import literature_mining as lm
    from . import panel_annotation as pa
    from . import figures as fg
except ImportError:
    import gene_ontology as go
    import pathway_enrichment as pe
    import protein_interaction as pi
    import disease_association as da
    import literature_mining as lm
    import panel_annotation as pa
    import figures as fg

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "pathway_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

SIGNIFICANCE_THRESHOLD = 0.05


class BiologicalValidationEngine:
    def __init__(self):
        self.execution_start = datetime.now()
        for d in (OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        self.genes = None
        self.go_results = None
        self.pathway_results = None
        self.string_network = None
        self.string_degree = None
        self.disease_associations = None
        self.literature = None
        self.panel_summary = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("BiologicalValidationEngine")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(REPORT_DIR / "execution_log.txt", mode="w")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def run_go_analysis(self) -> None:
        self.logger.info("Loading clinical panel genes...")
        self.genes = go.load_panel_genes()
        self.logger.info("Panel size: %d genes", len(self.genes))

        self.logger.info("Running GO enrichment (BP/CC/MF)...")
        self.go_results = go.run_go_analysis(self.genes)
        for namespace, df in self.go_results.items():
            df.to_csv(OUTPUT_DIR / f"GO_{namespace}.csv", index=False)
            n_sig = int((df["Adjusted P-value"] < SIGNIFICANCE_THRESHOLD).sum())
            self.logger.info("GO_%s: %d terms, %d significant (padj<0.05), min padj=%.4f",
                              namespace, len(df), n_sig, df["Adjusted P-value"].min())
            fg.plot_enrichment_barplot(df, FIGURE_DIR / f"GO_{namespace}_barplot.png", f"GO {namespace}")
            fg.plot_go_dotplot(df, FIGURE_DIR / f"GO_{namespace}_dotplot.png", f"GO {namespace}")

    def run_pathway_analysis(self) -> None:
        self.logger.info("Running KEGG/Reactome/WikiPathways enrichment...")
        self.pathway_results = pe.run_pathway_analysis(self.genes)
        for name, df in self.pathway_results.items():
            df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
            n_sig = int((df["Adjusted P-value"] < SIGNIFICANCE_THRESHOLD).sum())
            self.logger.info("%s: %d terms, %d significant (padj<0.05), min padj=%.4f",
                              name, len(df), n_sig, df["Adjusted P-value"].min())
            fg.plot_enrichment_barplot(df, FIGURE_DIR / f"{name}.png", name)

    def run_protein_interaction(self) -> None:
        self.logger.info("Fetching STRING PPI network...")
        self.string_network = pi.fetch_string_network(self.genes)
        self.string_network.to_csv(OUTPUT_DIR / "STRING_network.csv", index=False)

        self.string_degree = pi.compute_degree_centrality(self.string_network, self.genes)
        self.string_degree.to_csv(OUTPUT_DIR / "STRING_degree.csv", index=False)

        n_hubs = int(self.string_degree["is_hub"].sum())
        n_isolated = int((self.string_degree["string_degree"] == 0).sum())
        self.logger.info("STRING network: %d edges among %d genes, %d hub genes, %d isolated genes",
                          len(self.string_network), len(self.genes), n_hubs, n_isolated)

        fg.plot_string_network(self.string_network, self.string_degree, FIGURE_DIR / "STRING_network.png")

    def run_disease_association(self) -> None:
        self.logger.info("Fetching Open Targets disease associations...")
        self.disease_associations = da.run_disease_association(self.genes)
        self.disease_associations.to_csv(OUTPUT_DIR / "Disease_Associations.csv", index=False)

        n_resolved = int(self.disease_associations.groupby("gene_name")["disease_name"].apply(
            lambda s: s.notna().any()).sum())
        self.logger.info("Disease associations resolved for %d / %d genes", n_resolved, len(self.genes))

    def run_literature_mining(self) -> None:
        self.logger.info("Mining PubMed literature co-occurrence...")
        self.literature = lm.run_literature_mining(self.genes)
        self.literature.to_csv(OUTPUT_DIR / "Literature_Summary.csv", index=False)

        n_novel = int(self.literature["novel_in_cca_literature"].sum())
        self.logger.info("%d / %d genes have zero CCA-specific PubMed co-occurrence (novel/understudied)",
                          n_novel, len(self.genes))

    def run_panel_annotation(self) -> None:
        self.logger.info("Building consolidated panel annotation table...")
        self.panel_summary = pa.build_panel_annotation(
            self.genes, self.go_results, self.pathway_results,
            self.string_degree, self.disease_associations, self.literature,
        )
        self.panel_summary.to_csv(OUTPUT_DIR / "Biomarker_Annotation.csv", index=False)
        self.logger.info("Panel annotation table: %d rows (expected %d)",
                          len(self.panel_summary), len(self.genes))

    def export_reports(self) -> None:
        enrichment_results = {**{f"GO_{ns}": df for ns, df in self.go_results.items()}, **self.pathway_results}
        significance_summary = {
            name: {
                "n_terms": len(df),
                "n_significant": int((df["Adjusted P-value"] < SIGNIFICANCE_THRESHOLD).sum()),
                "min_adjusted_pvalue": float(df["Adjusted P-value"].min()),
                "top_term": df.iloc[0]["Term"] if len(df) else None,
                "top_term_genes": df.iloc[0]["Genes"] if len(df) else None,
            }
            for name, df in enrichment_results.items()
        }

        summary = {
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "panel_size": len(self.genes),
            "enrichment_significance_summary": significance_summary,
            "string_network": {
                "n_edges": len(self.string_network),
                "n_hub_genes": int(self.string_degree["is_hub"].sum()),
                "n_isolated_genes": int((self.string_degree["string_degree"] == 0).sum()),
                "hub_genes": self.string_degree.loc[self.string_degree["is_hub"], "gene_name"].tolist(),
            },
            "literature_mining": {
                "n_novel_in_cca_literature": int(self.literature["novel_in_cca_literature"].sum()),
                "novel_genes": self.literature.loc[
                    self.literature["novel_in_cca_literature"], "gene_name"].tolist(),
                "most_studied_gene": self.literature.sort_values(
                    "cca_hit_count", ascending=False).iloc[0]["gene_name"],
            },
        }
        with open(REPORT_DIR / "biological_validation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, default=str)

        lines = [
            "=" * 70,
            "BIOLOGICAL VALIDATION REPORT",
            "=" * 70,
            "",
            f"Generated : {datetime.now()}",
            f"Panel size : {len(self.genes)} genes",
            "",
            "Enrichment Significance Summary (GO / KEGG / Reactome / WikiPathways)",
            "-" * 70,
        ]
        for name, s in significance_summary.items():
            lines.append(
                f"{name:<14} {s['n_terms']:>4} terms, {s['n_significant']:>3} significant (padj<0.05), "
                f"min padj={s['min_adjusted_pvalue']:.4f}, top term: {s['top_term']} ({s['top_term_genes']})"
            )

        lines += [
            "",
            "Protein-Protein Interaction (STRING, required_score=400)",
            "-" * 70,
            f"{len(self.string_network)} edges among {len(self.genes)} genes "
            f"({int(self.string_degree['is_hub'].sum())} hub genes at degree>=2, "
            f"{int((self.string_degree['string_degree']==0).sum())} isolated genes).",
            "A sparse/mostly-disconnected network is an expected result for this panel:",
            "the 20 genes were selected via independent differential-expression + Random",
            "Forest + XGBoost consensus scoring, not via known-pathway curation, so strong",
            "prior PPI connectivity was never a selection criterion.",
            f"Hub genes: {', '.join(self.string_degree.loc[self.string_degree['is_hub'], 'gene_name'])}",
            "",
            "Literature Mining (PubMed co-occurrence)",
            "-" * 70,
            f"Most-studied gene in CCA context: "
            f"{self.literature.sort_values('cca_hit_count', ascending=False).iloc[0]['gene_name']} "
            f"({int(self.literature['cca_hit_count'].max())} hits)",
            f"Genes with zero CCA-specific PubMed co-occurrence "
            f"({int(self.literature['novel_in_cca_literature'].sum())}/{len(self.genes)}, "
            "novel/understudied candidates, not an error):",
            ", ".join(self.literature.loc[self.literature["novel_in_cca_literature"], "gene_name"]),
        ]
        (REPORT_DIR / "biological_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        self.logger.info("Biological Validation Engine Started")
        self.run_go_analysis()
        self.run_pathway_analysis()
        self.run_protein_interaction()
        self.run_disease_association()
        self.run_literature_mining()
        self.run_panel_annotation()
        self.export_reports()
        self.logger.info("Execution Time: %s", datetime.now() - self.execution_start)


if __name__ == "__main__":
    BiologicalValidationEngine().run()
