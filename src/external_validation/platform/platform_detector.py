from pathlib import Path
import GEOparse
import pandas as pd
import json


class PlatformDetector:

    def __init__(self, gse_id):

        self.gse_id = gse_id

        self.base_dir = Path(__file__).resolve().parents[3]

        self.raw_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "raw"
        )

        self.report_dir = (
            self.base_dir /
            "results" /
            "platform_reports"
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_dataset(self):

        print("=" * 70)
        print(f"Loading {self.gse_id}")
        print("=" * 70)

        self.gse = GEOparse.get_GEO(
            geo=self.gse_id,
            destdir=str(self.raw_dir)
        )

    def detect_platform(self):

        self.platforms = []

        for gpl_id, gpl in self.gse.gpls.items():

            platform = {}

            platform["gse"] = self.gse_id
            platform["platform_id"] = gpl_id

            platform["title"] = (
                gpl.metadata.get(
                    "title",
                    ["Unknown"]
                )[0]
            )

            platform["technology"] = (
                gpl.metadata.get(
                    "technology",
                    ["Unknown"]
                )[0]
            )

            platform["organism"] = (
                gpl.metadata.get(
                    "organism",
                    ["Unknown"]
                )[0]
            )

            platform["manufacturer"] = (
                gpl.metadata.get(
                    "manufacturer",
                    ["Unknown"]
                )[0]
            )

            platform["sample_count"] = len(
                self.gse.gsms
            )

            first_sample = next(
                iter(self.gse.gsms.values())
            )

            table = first_sample.table

            platform["columns"] = list(
                table.columns
            )

            probe_column = table.columns[0]

            expression_column = None

            candidate_columns = [

                "VALUE",

                "value",

                "Signal",

                "signal",

                "AVG_Signal"

            ]

            for col in candidate_columns:

                if col in table.columns:

                    expression_column = col

                    break

            platform["probe_column"] = probe_column

            platform["expression_column"] = expression_column

            platform["ready_for_mapping"] = (
                expression_column is not None
            )

            self.platforms.append(
                platform
            )

    def save_report(self):

        df = pd.DataFrame(
            self.platforms
        )

        output = (
            self.report_dir /
            f"{self.gse_id}_platform_report.csv"
        )

        df.to_csv(
            output,
            index=False
        )

        print("\nPlatform Report Saved")

        print(output)

        print()

        print(df)

    def summary(self):

        print("\n")

        print("=" * 70)

        print("PLATFORM SUMMARY")

        print("=" * 70)

        for p in self.platforms:

            print(json.dumps(
                p,
                indent=4
            ))

        print("=" * 70)


if __name__ == "__main__":

    datasets = [

        "GSE89749",

        "GSE26566",

        "GSE32225"

    ]

    for dataset in datasets:

        detector = PlatformDetector(
            dataset
        )

        detector.load_dataset()

        detector.detect_platform()

        detector.summary()

        detector.save_report()