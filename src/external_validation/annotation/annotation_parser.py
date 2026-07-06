from pathlib import Path
import pandas as pd


class AnnotationParser:

    def __init__(self, platform_id):

        self.platform_id = platform_id

        self.base_dir = Path(__file__).resolve().parents[3]

        self.annotation_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "annotations"
        )

        self.output_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "processed_annotations"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load(self):

        file = (
            self.annotation_dir /
            f"{self.platform_id}_annotation.csv"
        )

        self.df = pd.read_csv(
            file,
            low_memory=False
        )

        print(f"\nLoaded {file}")
        print(self.df.shape)

    def detect_columns(self):

        columns = list(self.df.columns)

        print("\nDetected Columns\n")
        print(columns)

        lookup = {

            "probe_id": [
                "ID",
                "Probe_Id",
                "PROBE_ID"
            ],

            "gene_symbol": [
                "Symbol",
                "Gene Symbol",
                "GENE_SYMBOL",
                "Gene symbol"
            ],

            "gene_name": [
                "Definition",
                "Gene Title",
                "Description",
                "Gene Name"
            ],

            "entrez_id": [
                "ENTREZ_GENE_ID",
                "Entrez Gene",
                "Entrez_ID"
            ],

            "ensembl_id": [
                "Ensembl",
                "ENSEMBL",
                "Ensembl_ID"
            ]
        }

        self.mapping = {}

        for field, candidates in lookup.items():

            found = None

            for candidate in candidates:

                if candidate in columns:

                    found = candidate
                    break

            self.mapping[field] = found

        print("\nDetected Mapping\n")

        for k, v in self.mapping.items():

            print(f"{k:15} -> {v}")

    def standardize(self):

        output = pd.DataFrame()

        for std_col, original in self.mapping.items():

            if original is None:

                output[std_col] = None

            else:

                output[std_col] = self.df[original]

        self.standard = output

    def save(self):

        file = (
            self.output_dir /
            f"{self.platform_id}_standard_annotation.csv"
        )

        self.standard.to_csv(
            file,
            index=False
        )

        print("\nSaved")

        print(file)

        print("\nPreview")

        print(self.standard.head())


if __name__ == "__main__":

    platforms = [

        "GPL10558",

        "GPL6104",

        "GPL8432"

    ]

    for platform in platforms:

        print("\n" + "="*80)

        parser = AnnotationParser(
            platform
        )

        parser.load()

        parser.detect_columns()

        parser.standardize()

        parser.save()