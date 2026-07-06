"""Drug target discovery orchestrator.

Translates the clinical biomarker panel into potential therapeutic
opportunities: composite target prioritization (combining predictive
importance, external-cohort relevance, biological relevance, and survival
association from Modules 9-11), DGIdb + Open Targets drug-gene evidence,
and repurposing candidates. DrugBank is intentionally not integrated (its
API now requires a paid academic license, confirmed during Module 10).
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from . import open_targets_drugs as otd
    from . import dgidb_search as ds
    from . import target_prioritization as tp
    from . import drug_repurposing as dr
    from . import figures as fg
except ImportError:
    import open_targets_drugs as otd
    import dgidb_search as ds
    import target_prioritization as tp
    import drug_repurposing as dr
    import figures as fg

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "drug_targets"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"


class DrugTargetEngine:
    def __init__(self):
        self.execution_start = datetime.now()
        for d in (OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        self.genes = None
        self.priority_table = None
        self.dgidb_results = None
        self.open_targets_drugs = None
        self.tractability = None
        self.repurposing_candidates = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DrugTargetEngine")
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

    def load_genes(self) -> None:
        self.genes = pd.read_csv(otd.CLINICAL_PANEL_FILE)["gene_name"].tolist()
        self.logger.info("Panel size: %d genes", len(self.genes))

    def run_target_prioritization(self) -> None:
        self.logger.info("Building composite target priority table...")
        self.priority_table = tp.build_target_priority_table(self.genes)
        self.priority_table.to_csv(OUTPUT_DIR / "TherapeuticPriority.csv", index=False)
        fg.plot_target_priority(self.priority_table, FIGURE_DIR / "TargetPriority.png")
        top5 = self.priority_table.head(5)["gene_name"].tolist()
        self.logger.info("Top 5 prioritized targets: %s", top5)

    def run_dgidb_search(self) -> None:
        self.logger.info("Querying DGIdb for drug-gene interactions...")
        self.dgidb_results = ds.run_dgidb_search(self.genes)
        self.dgidb_results.to_csv(OUTPUT_DIR / "DGIdb.csv", index=False)
        n_genes = self.dgidb_results.groupby("gene_name")["drug_name"].apply(lambda s: s.notna().any()).sum()
        self.logger.info("DGIdb: %d rows, %d / %d genes have known interactions",
                          len(self.dgidb_results), n_genes, len(self.genes))

    def run_open_targets_drugs(self) -> None:
        self.logger.info("Querying Open Targets for known drugs and tractability...")
        self.open_targets_drugs = otd.run_known_drugs(self.genes)
        self.open_targets_drugs.to_csv(OUTPUT_DIR / "OpenTargets.csv", index=False)

        self.tractability = otd.run_tractability(self.genes)
        self.tractability.to_csv(OUTPUT_DIR / "Tractability.csv", index=False)
        n_druggable = int(self.tractability["is_druggable_family_sm"].sum())
        self.logger.info("Open Targets: %d / %d genes flagged as small-molecule druggable family",
                          n_druggable, len(self.genes))

    def run_drug_repurposing(self) -> None:
        self.logger.info("Building drug repurposing candidate table...")
        self.repurposing_candidates = dr.build_repurposing_candidates(
            self.dgidb_results, self.open_targets_drugs, self.priority_table
        )
        self.repurposing_candidates.to_csv(OUTPUT_DIR / "DrugRepurposing.csv", index=False)
        fg.plot_drug_network(self.repurposing_candidates, FIGURE_DIR / "DrugNetwork.png")
        self.logger.info("Repurposing candidates: %d drug-gene pairs across %d genes",
                          len(self.repurposing_candidates), self.repurposing_candidates["gene_name"].nunique())

    def run_biomarker_drug_table(self) -> None:
        self.logger.info("Building consolidated biomarker-drug table...")
        table = self.priority_table.merge(self.tractability, on="gene_name", how="left")
        n_drugs_per_gene = self.repurposing_candidates.groupby("gene_name")["drug_name"].nunique()
        table["n_known_drugs"] = table["gene_name"].map(n_drugs_per_gene).fillna(0).astype(int)
        assert len(table) == len(self.genes)
        table.to_csv(OUTPUT_DIR / "BiomarkerDrugTable.csv", index=False)

    def export_reports(self) -> None:
        summary = {
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "panel_size": len(self.genes),
            "top5_priority_targets": self.priority_table.head(5)[
                ["gene_name", "target_priority_score"]].to_dict("records"),
            "n_genes_with_dgidb_interactions": int(
                self.dgidb_results.groupby("gene_name")["drug_name"].apply(lambda s: s.notna().any()).sum()),
            "n_genes_with_open_targets_drugs": int(
                self.open_targets_drugs.groupby("gene_name")["drug_name"].apply(lambda s: s.notna().any()).sum()),
            "n_druggable_family_sm": int(self.tractability["is_druggable_family_sm"].sum()),
            "n_repurposing_candidates": len(self.repurposing_candidates),
            "top_repurposing_candidates": self.repurposing_candidates.head(5)[
                ["gene_name", "drug_name", "source", "target_priority_score"]].to_dict("records"),
            "note": "DrugBank not integrated - its API requires a paid academic license.",
        }
        with open(REPORT_DIR / "drug_target_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, default=str)

        lines = [
            "=" * 70,
            "DRUG TARGET DISCOVERY REPORT",
            "=" * 70,
            "",
            f"Generated : {datetime.now()}",
            f"Panel size : {len(self.genes)} genes",
            "",
            "Top 5 Prioritized Targets (composite score: predictive + external +",
            "biological relevance + survival association, equally weighted)",
            "-" * 70,
        ]
        for _, row in self.priority_table.head(5).iterrows():
            lines.append(f"{row['gene_name']}: score={row['target_priority_score']:.3f}")

        lines += [
            "",
            "Drug-Gene Evidence",
            "-" * 70,
            f"DGIdb: {int(self.dgidb_results.groupby('gene_name')['drug_name'].apply(lambda s: s.notna().any()).sum())} / {len(self.genes)} genes have known interactions",
            f"Open Targets: {int(self.open_targets_drugs.groupby('gene_name')['drug_name'].apply(lambda s: s.notna().any()).sum())} / {len(self.genes)} genes have known drugs/clinical candidates",
            f"Small-molecule druggable family: {int(self.tractability['is_druggable_family_sm'].sum())} / {len(self.genes)} genes",
            "",
            "Top Repurposing Candidates (ranked by target priority score)",
            "-" * 70,
        ]
        for _, row in self.repurposing_candidates.head(10).iterrows():
            lines.append(f"{row['gene_name']} -> {row['drug_name']} ({row['source']}, "
                          f"priority={row['target_priority_score']:.3f})")

        lines += [
            "",
            "Note: DrugBank was not integrated - its API now requires a paid academic",
            "license. DGIdb and Open Targets (both free) provide the drug-gene evidence",
            "in this report.",
        ]
        (REPORT_DIR / "drug_target_report.txt").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        self.logger.info("Drug Target Discovery Engine Started")
        self.load_genes()
        self.run_target_prioritization()
        self.run_dgidb_search()
        self.run_open_targets_drugs()
        self.run_drug_repurposing()
        self.run_biomarker_drug_table()
        self.export_reports()
        self.logger.info("Execution Time: %s", datetime.now() - self.execution_start)


if __name__ == "__main__":
    DrugTargetEngine().run()
