from pathlib import Path
import pandas as pd


class ExternalCohortBuilder:

    def __init__(self):

        self.base = Path(__file__).resolve().parents[3]

        self.expression_dir = (
            self.base /
            "data" /
            "geo" /
            "harmonized_expression"
        )

        self.annotation_dir = (
            self.base /
            "results" /
            "sample_annotation"
        )

        self.output_dir = (
            self.base /
            "results" /
            "external_cohorts"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def build_dataset(self, expression_file):

        # ----------------------------------------------------
        # Get GSE ID
        # ----------------------------------------------------

        gse = expression_file.stem

        gse = gse.replace("_harmonized_expression", "")
        gse = gse.replace("_harmonized", "")

        print("=" * 90)
        print(gse)

        annotation_file = (
            self.annotation_dir /
            f"{gse}_annotation.csv"
        )

        if not annotation_file.exists():

            print("Annotation file not found:")
            print(annotation_file)
            print()

            return

        expr = pd.read_csv(expression_file)
        ann = pd.read_csv(annotation_file)

        print("Expression Shape :", expr.shape)
        print("Annotation Shape :", ann.shape)

        # ----------------------------------------------------
        # Detect gene column automatically
        # ----------------------------------------------------

        gene_column = None

        if "gene_name" in expr.columns:
            gene_column = "gene_name"

        elif "gene_symbol" in expr.columns:
            gene_column = "gene_symbol"

        elif "Gene Symbol" in expr.columns:
            gene_column = "Gene Symbol"

        else:

            raise ValueError(
                "No gene column found in expression matrix."
            )

        print("Gene Column :", gene_column)

        # ----------------------------------------------------
        # Convert genes x samples
        # into samples x genes
        # ----------------------------------------------------

        expr = expr.set_index(gene_column).T

        expr.index.name = "sample_id"

        expr.reset_index(inplace=True)

        print("Transposed Shape :", expr.shape)

        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        merged = pd.merge(

            ann,

            expr,

            on="sample_id",

            how="inner"

        )

        # ----------------------------------------------------
        # QC REPORT
        # ----------------------------------------------------

        print()
        print("QC REPORT")
        print("-" * 50)

        print("Expression Samples :", expr.shape[0])
        print("Annotation Samples :", ann.shape[0])
        print("Matched Samples    :", merged.shape[0])
        print("Genes              :", expr.shape[1] - 1)

        print("-" * 50)
        print()

        print("Merged Shape :", merged.shape)

        # ----------------------------------------------------
        # Keep only resolved samples
        # ----------------------------------------------------

        merged = merged[
            merged["annotation_status"] == "Resolved"
        ]

        print("Resolved Samples :", len(merged))

        outfile = (

            self.output_dir /

            f"{gse}_external_cohort.csv"

        )

        merged.to_csv(
            outfile,
            index=False
        )

        print("Saved :", outfile)
        print()

    def run(self):

        files = sorted(

            list(
                self.expression_dir.glob(
                    "*_harmonized.csv"
                )
            )

            +

            list(
                self.expression_dir.glob(
                    "*_harmonized_expression.csv"
                )
            )

        )

        print("=" * 90)
        print("HARMONIZED DATASETS FOUND")
        print("=" * 90)

        if len(files) == 0:

            print("No harmonized datasets found.")
            return

        for f in files:
            print(f.name)

        print()

        for f in files:

            self.build_dataset(f)

        print("=" * 90)
        print("External Cohort Building Completed")
        print("=" * 90)


if __name__ == "__main__":

    ExternalCohortBuilder().run()