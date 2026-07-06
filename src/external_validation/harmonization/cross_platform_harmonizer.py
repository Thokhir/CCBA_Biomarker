from pathlib import Path
import pandas as pd
import numpy as np


class CrossPlatformHarmonizer:

    def __init__(self, gse):

        self.gse = gse

        self.base_dir = Path(__file__).resolve().parents[3]

        self.tcga_file = (
            self.base_dir /
            "data" /
            "processed" /
            "expression_logCPM.csv"
        )

        self.geo_file = (
            self.base_dir /
            "data" /
            "geo" /
            "resolved_expression" /
            f"{gse}_resolved_expression.csv"
        )

        self.output_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "harmonized_expression"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load(self):

        self.tcga = pd.read_csv(
            self.tcga_file
        )

        self.geo = pd.read_csv(
            self.geo_file
        )

        print("\nTCGA Shape")
        print(self.tcga.shape)

        print("\nGEO Shape")
        print(self.geo.shape)

    def common_genes(self):

        tcga = set(
            self.tcga["gene_name"]
        )

        geo = set(
            self.geo["gene_symbol"]
        )

        common = sorted(
            tcga.intersection(geo)
        )

        print()

        print("Common Genes:", len(common))

        self.tcga = self.tcga[
            self.tcga["gene_name"].isin(common)
        ].copy()

        self.geo = self.geo[
            self.geo["gene_symbol"].isin(common)
        ].copy()

        self.tcga = self.tcga.sort_values(
            "gene_name"
        ).reset_index(drop=True)

        self.geo = self.geo.sort_values(
            "gene_symbol"
        ).reset_index(drop=True)

    def compute_tcga_statistics(self):

        tcga_values = self.tcga.iloc[:,2:]

        self.mean = tcga_values.mean(axis=1)

        self.std = tcga_values.std(axis=1)

        self.std.replace(
            0,
            1,
            inplace=True
        )

        print()

        print("Training statistics computed.")

    def harmonize(self):

        geo_values = self.geo.iloc[:,1:]

        harmonized = (
            geo_values.sub(
                self.mean,
                axis=0
            )
            .div(
                self.std,
                axis=0
            )
        )

        harmonized.insert(
            0,
            "gene_symbol",
            self.geo["gene_symbol"]
        )

        self.harmonized = harmonized

    def qc(self):

        print()

        print("="*60)

        print("QC")

        print("="*60)

        print("Rows:", len(self.harmonized))

        print("Columns:", len(self.harmonized.columns))

        print()

        print("Missing Values")

        print(
            self.harmonized.isnull().sum().sum()
        )

        print()

        print("Duplicated Genes")

        print(
            self.harmonized["gene_symbol"]
            .duplicated()
            .sum()
        )

    def save(self):

        output = (
            self.output_dir /
            f"{self.gse}_harmonized.csv"
        )

        self.harmonized.to_csv(
            output,
            index=False
        )

        print()

        print("Saved")

        print(output)


if __name__ == "__main__":

    datasets = [

        "GSE89749",

        "GSE26566",

        "GSE32225"

    ]

    for gse in datasets:

        print("\n")

        print("="*80)

        harmonizer = CrossPlatformHarmonizer(
            gse
        )

        harmonizer.load()

        harmonizer.common_genes()

        harmonizer.compute_tcga_statistics()

        harmonizer.harmonize()

        harmonizer.qc()

        harmonizer.save()