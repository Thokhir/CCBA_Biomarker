from pathlib import Path
import pandas as pd


class MetadataContentInspector:

    def __init__(self):

        self.base_dir = Path(__file__).resolve().parents[3]

        self.metadata_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "metadata"
        )

        self.output_dir = (
            self.base_dir /
            "results" /
            "metadata_inspection"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.priority_columns = [

            "title",
            "source_name_ch1",
            "characteristics_ch1",
            "description"

        ]

    def discover(self):

        self.files = sorted(
            self.metadata_dir.glob("*_sample_metadata.csv")
        )

        print("="*90)
        print("METADATA FILES")
        print("="*90)

        for f in self.files:
            print(f.name)

        print()

    def inspect_dataset(self, file):

        gse = file.stem.replace("_sample_metadata", "")

        print("="*90)
        print(gse)
        print("="*90)

        df = pd.read_csv(file)

        report = []

        for col in self.priority_columns:

            if col not in df.columns:
                continue

            print(f"\nCOLUMN : {col}")

            values = (
                df[col]
                .fillna("NA")
                .astype(str)
                .value_counts()
            )

            print(values.head(50))

            print()

            for value, count in values.items():

                report.append({

                    "column": col,

                    "value": value,

                    "count": count

                })

        report = pd.DataFrame(report)

        outfile = (
            self.output_dir /
            f"{gse}_content_report.csv"
        )

        report.to_csv(outfile, index=False)

        print("Saved :", outfile)

        print()

    def run(self):

        self.discover()

        for file in self.files:

            self.inspect_dataset(file)

        print("="*90)
        print("Inspection Completed")
        print("="*90)


if __name__ == "__main__":

    MetadataContentInspector().run()