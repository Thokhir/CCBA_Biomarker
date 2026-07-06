from pathlib import Path
import pandas as pd


class MetadataQualityAssessment:

    def __init__(self):

        self.base_dir = Path(__file__).resolve().parents[3]

        self.metadata_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "metadata"
        )

        self.report_dir = (
            self.base_dir /
            "results" /
            "metadata_reports"
        )

        self.report_dir.mkdir(parents=True, exist_ok=True)

    def discover_files(self):

        self.files = sorted(
            self.metadata_dir.glob("*_sample_metadata.csv")
        )

        print("=" * 90)
        print("METADATA FILES FOUND")
        print("=" * 90)

        for f in self.files:
            print(f.name)

        print()

    def inspect_dataset(self, file):

        gse = file.stem.replace("_sample_metadata", "")

        print("=" * 90)
        print(gse)

        df = pd.read_csv(file)

        report = []

        for col in df.columns:

            unique_values = df[col].dropna().astype(str).unique()

            report.append({

                "column": col,

                "dtype": str(df[col].dtype),

                "missing_values": int(df[col].isna().sum()),

                "unique_values": int(df[col].nunique()),

                "example_values":
                " | ".join(unique_values[:5])

            })

        report_df = pd.DataFrame(report)

        outfile = (
            self.report_dir /
            f"{gse}_metadata_report.csv"
        )

        report_df.to_csv(outfile, index=False)

        print(f"Samples : {len(df)}")
        print(f"Columns : {len(df.columns)}")
        print("Saved :", outfile)

        print("\nMost Informative Columns")

        informative = report_df[
            report_df["unique_values"] > 1
        ].sort_values(
            "unique_values",
            ascending=False
        )

        print(
            informative[
                [
                    "column",
                    "unique_values",
                    "missing_values"
                ]
            ].head(10)
        )

        print()

    def run(self):

        self.discover_files()

        for file in self.files:

            self.inspect_dataset(file)

        print("=" * 90)
        print("Metadata Quality Assessment Completed")
        print("=" * 90)


if __name__ == "__main__":

    MetadataQualityAssessment().run()