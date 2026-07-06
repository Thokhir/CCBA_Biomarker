from pathlib import Path
import pandas as pd


class AnnotationCleaner:

    def __init__(self, platform):

        self.platform = platform

        self.base_dir = Path(__file__).resolve().parents[3]

        self.input_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "processed_annotations"
        )

        self.output_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "clean_annotations"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load(self):

        file = (
            self.input_dir /
            f"{self.platform}_standard_annotation.csv"
        )

        self.df = pd.read_csv(
            file,
            low_memory=False
        )

        self.original_rows = len(self.df)

        print(f"\nLoaded {file}")
        print(self.df.shape)

    def clean_gene_symbols(self):

        self.df["gene_symbol"] = (
            self.df["gene_symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        self.df = self.df[
            self.df["gene_symbol"] != ""
        ]

        self.df = self.df[
            self.df["gene_symbol"] != "-"
        ]

        self.df = self.df[
            self.df["gene_symbol"] != "NA"
        ]

    def remove_control_probes(self):

        control_keywords = [

            "CONTROL",
            "PHAGE",
            "NEGATIVE",
            "SPIKE",
            "ERCC",
            "AFFX",
            "HOUSEKEEPING",
            "ANTI"

        ]

        mask = pd.Series(False, index=self.df.index)

        for keyword in control_keywords:

            mask |= self.df[
                "gene_symbol"
            ].str.contains(
                keyword,
                case=False,
                na=False
            )

        removed = mask.sum()

        self.df = self.df[
            ~mask
        ]

        print(f"Removed Control Probes : {removed}")

    def remove_duplicates(self):

        before = len(self.df)

        self.df = self.df.drop_duplicates(
            subset=["probe_id"]
        )

        probe_removed = before - len(self.df)

        before = len(self.df)

        self.df = self.df.drop_duplicates(
            subset=["probe_id", "gene_symbol"]
        )

        mapping_removed = before - len(self.df)

        print(f"Duplicate Probes Removed : {probe_removed}")
        print(f"Duplicate Mappings Removed : {mapping_removed}")

    def save(self):

        output = (
            self.output_dir /
            f"{self.platform}_clean_annotation.csv"
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

        print(f"Platform : {self.platform}")

        print(f"Original Rows : {self.original_rows}")

        print(f"Final Rows : {len(self.df)}")

        print("="*70)

        print(self.df.head())


if __name__ == "__main__":

    platforms = [

        "GPL10558",

        "GPL6104",

        "GPL8432"

    ]

    for platform in platforms:

        print("\n" + "="*80)

        cleaner = AnnotationCleaner(
            platform
        )

        cleaner.load()

        cleaner.clean_gene_symbols()

        cleaner.remove_control_probes()

        cleaner.remove_duplicates()

        cleaner.summary()

        cleaner.save()