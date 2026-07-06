from pathlib import Path
import pandas as pd
import numpy as np


class DuplicateProbeResolver:

    def __init__(self, gse_id):

        self.gse = gse_id

        self.base_dir = Path(__file__).resolve().parents[3]

        self.input_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "mapped_expression"
        )

        self.output_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "resolved_expression"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load(self):

        file = (
            self.input_dir /
            f"{self.gse}_gene_expression.csv"
        )

        print(f"\nLoading {file.name}")

        self.df = pd.read_csv(
            file,
            low_memory=False
        )

        print("Input Shape:", self.df.shape)

    def calculate_variance(self):

        expression_cols = self.df.columns[1:]

        self.df["variance"] = (
            self.df[expression_cols]
            .astype(float)
            .var(axis=1)
        )

    def resolve_duplicates(self):

        duplicate_genes = self.df[
            self.df["gene_symbol"].duplicated(keep=False)
        ]["gene_symbol"].nunique()

        print(f"Duplicated Genes Found : {duplicate_genes}")

        self.df = (
            self.df
            .sort_values(
                "variance",
                ascending=False
            )
            .drop_duplicates(
                subset="gene_symbol",
                keep="first"
            )
        )

        print("Resolved Shape:", self.df.shape)

    def clean(self):

        self.df = self.df.drop(
            columns=["variance"]
        )

    def save(self):

        output = (
            self.output_dir /
            f"{self.gse}_resolved_expression.csv"
        )

        self.df.to_csv(
            output,
            index=False
        )

        print("\nSaved")

        print(output)

    def summary(self):

        print("\n")

        print("="*70)

        print("FINAL SUMMARY")

        print("="*70)

        print("Genes :", len(self.df))

        print()

        print(self.df.head())

        print("="*70)


if __name__ == "__main__":

    datasets = [

        "GSE89749",

        "GSE26566",

        "GSE32225"

    ]

    for dataset in datasets:

        print("\n" + "="*80)

        resolver = DuplicateProbeResolver(
            dataset
        )

        resolver.load()

        resolver.calculate_variance()

        resolver.resolve_duplicates()

        resolver.clean()

        resolver.summary()

        resolver.save()