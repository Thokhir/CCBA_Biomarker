from pathlib import Path
import pandas as pd
import numpy as np


class FeatureAlignmentEngine:

    def __init__(self):

        self.base = Path(__file__).resolve().parents[3]

        self.cohort_dir = (
            self.base /
            "results" /
            "external_cohorts"
        )

        self.biomarker_file = (
            self.base /
            "results" /
            "clinical_biomarker_panel.csv"
        )

        self.output_dir = (
            self.base /
            "results" /
            "aligned_external_cohorts"
        )

        self.report_dir = (
            self.base /
            "results" /
            "feature_alignment_reports"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_biomarkers(self):

        biomarkers = pd.read_csv(self.biomarker_file)

        self.required_genes = biomarkers["gene_name"].tolist()

        print("=" * 90)
        print("CONSENSUS BIOMARKERS")
        print("=" * 90)

        for gene in self.required_genes:
            print(gene)

        print()

    def process_dataset(self, file):

        gse = file.stem.replace("_external_cohort", "")

        print("=" * 90)
        print(gse)

        df = pd.read_csv(file)

        metadata_cols = [

            "sample_id",
            "phenotype",
            "final_label",
            "disease",
            "tissue",
            "subtype",
            "confidence",
            "matched_column",
            "matched_keyword",
            "annotation_status"

        ]

        available = []
        missing = []

        report = []

        for gene in self.required_genes:

            if gene in df.columns:

                available.append(gene)

                values = pd.to_numeric(
                    df[gene],
                    errors="coerce"
                )

                report.append({

                    "gene": gene,

                    "present": True,

                    "missing_values":
                        values.isna().sum(),

                    "zero_variance":
                        values.nunique() == 1

                })

            else:

                missing.append(gene)

                report.append({

                    "gene": gene,

                    "present": False,

                    "missing_values": np.nan,

                    "zero_variance": np.nan

                })

        print("Available Biomarkers :", len(available))
        print("Missing Biomarkers   :", len(missing))

        if len(missing):

            print("\nMissing Genes")

            for gene in missing:
                print(" -", gene)

        aligned = df[
            metadata_cols + available
        ]

        out_file = (
            self.output_dir /
            f"{gse}_aligned.csv"
        )

        aligned.to_csv(
            out_file,
            index=False
        )

        report_df = pd.DataFrame(report)

        report_file = (
            self.report_dir /
            f"{gse}_feature_report.csv"
        )

        report_df.to_csv(
            report_file,
            index=False
        )

        print()

        print("Aligned Dataset Shape :", aligned.shape)

        print("Saved :", out_file)

        print("Feature Report :", report_file)

        print()

    def run(self):

        self.load_biomarkers()

        files = sorted(

            self.cohort_dir.glob(
                "*_external_cohort.csv"
            )

        )

        if len(files) == 0:

            print("No external cohorts found.")

            return

        print("=" * 90)
        print("EXTERNAL COHORTS FOUND")
        print("=" * 90)

        for f in files:
            print(f.name)

        print()

        for f in files:

            self.process_dataset(f)

        print("=" * 90)
        print("FEATURE ALIGNMENT COMPLETED")
        print("=" * 90)


if __name__ == "__main__":

    FeatureAlignmentEngine().run()