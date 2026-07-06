from pathlib import Path
import GEOparse
import pandas as pd


class GEOExpressionExtractor:

    def __init__(self, gse_id):

        self.gse_id = gse_id

        self.base_dir = Path(__file__).resolve().parents[3]

        self.raw_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "raw"
        )

        self.output_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "expression"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_dataset(self):

        print("="*80)
        print(f"Loading {self.gse_id}")
        print("="*80)

        self.gse = GEOparse.get_GEO(
            geo=self.gse_id,
            destdir=str(self.raw_dir)
        )

    def extract(self):

        expression = None

        total = len(self.gse.gsms)

        print(f"\nSamples Found : {total}")

        for i, (gsm_id, gsm) in enumerate(self.gse.gsms.items(), start=1):

            print(f"Processing {i}/{total} : {gsm_id}")

            table = gsm.table.copy()

            columns = list(table.columns)

            probe_column = columns[0]

            value_column = None

            candidates = [

                "VALUE",
                "value",
                "Signal",
                "signal",
                "AVG_Signal"

            ]

            for c in candidates:

                if c in columns:
                    value_column = c
                    break

            if value_column is None:

                raise ValueError(
                    f"No expression column detected for {gsm_id}\n"
                    f"Columns are:\n{columns}"
                )

            sample = table[
                [probe_column, value_column]
            ].copy()

            sample.columns = [

                "probe_id",
                gsm_id

            ]

            if expression is None:

                expression = sample

            else:

                expression = expression.merge(

                    sample,

                    on="probe_id",

                    how="outer"

                )

        self.expression = expression

    def quality_control(self):

        print("\n")

        print("="*80)

        print("QUALITY CONTROL")

        print("="*80)

        print("Shape")

        print(self.expression.shape)

        print()

        print("Missing Values")

        print(
            self.expression.isnull().sum().sum()
        )

        print()

        print("Duplicated Probe IDs")

        print(
            self.expression["probe_id"].duplicated().sum()
        )

    def save(self):

        output = (

            self.output_dir /

            f"{self.gse_id}_expression.csv"

        )

        self.expression.to_csv(

            output,

            index=False

        )

        print("\nSaved")

        print(output)


if __name__ == "__main__":

    datasets = [

        "GSE89749",

        "GSE26566",

        "GSE32225"

    ]

    for dataset in datasets:

        extractor = GEOExpressionExtractor(

            dataset

        )

        extractor.load_dataset()

        extractor.extract()

        extractor.quality_control()

        extractor.save()