from pathlib import Path
import pandas as pd


class ProbeMapper:

    def __init__(self, gse_id, platform_id):

        self.gse = gse_id
        self.platform = platform_id

        self.base_dir = Path(__file__).resolve().parents[3]

        self.expression_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "expression"
        )

        self.annotation_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "clean_annotations"
        )

        self.output_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "mapped_expression"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_expression(self):

        expression_file = (
            self.expression_dir /
            f"{self.gse}_expression.csv"
        )

        self.expression = pd.read_csv(
            expression_file,
            low_memory=False
        )

        print("\nExpression Shape")
        print(self.expression.shape)

    def load_annotation(self):

        annotation_file = (
            self.annotation_dir /
            f"{self.platform}_clean_annotation.csv"
        )

        self.annotation = pd.read_csv(
            annotation_file,
            low_memory=False
        )

        print("\nAnnotation Shape")
        print(self.annotation.shape)

    def merge(self):

        probe_column = self.expression.columns[0]

        merged = self.expression.merge(
            self.annotation,
            left_on=probe_column,
            right_on="probe_id",
            how="left"
        )

        self.merged = merged

        print("\nMerged Shape")
        print(self.merged.shape)

    def clean(self):

        before = len(self.merged)

        self.merged = self.merged.dropna(
            subset=["gene_symbol"]
        )

        removed = before - len(self.merged)

        print(f"\nUnmapped Probes Removed : {removed}")

    def create_gene_matrix(self):

        expression_columns = self.expression.columns[1:]

        gene_matrix = self.merged[
            ["gene_symbol"] +
            list(expression_columns)
        ]

        self.gene_matrix = gene_matrix

    def save(self):

        output = (
            self.output_dir /
            f"{self.gse}_gene_expression.csv"
        )

        self.gene_matrix.to_csv(
            output,
            index=False
        )

        print("\nSaved")

        print(output)

    def summary(self):

        print("\n")

        print("="*70)

        print(self.gene_matrix.head())

        print("="*70)

        print(f"Genes : {len(self.gene_matrix)}")


if __name__ == "__main__":

    datasets = [

        ("GSE89749", "GPL10558"),

        ("GSE26566", "GPL6104"),

        ("GSE32225", "GPL8432")

    ]

    for gse, gpl in datasets:

        print("\n" + "="*80)

        mapper = ProbeMapper(
            gse,
            gpl
        )

        mapper.load_expression()

        mapper.load_annotation()

        mapper.merge()

        mapper.clean()

        mapper.create_gene_matrix()

        mapper.summary()

        mapper.save()