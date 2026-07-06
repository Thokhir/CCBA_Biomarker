from pathlib import Path
import pandas as pd


class SampleAnnotationEngine:

    def __init__(self):

        self.base_dir = Path(__file__).resolve().parents[3]

        self.input_dir = (
            self.base_dir /
            "results" /
            "parsed_metadata"
        )

        self.output_dir = (
            self.base_dir /
            "results" /
            "sample_annotation"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def process_dataset(self, file):

        gse = file.stem.replace("_parsed_metadata", "")

        print("=" * 90)
        print(gse)

        df = pd.read_csv(file)

        df["final_label"] = df["phenotype"].map({

            "Tumor": 1,
            "Normal": 0

        })

        df["annotation_status"] = df["phenotype"].apply(

            lambda x:
            "Resolved"
            if x in ["Tumor", "Normal"]
            else "Needs Review"

        )

        cols = [

            "sample_id",

            "phenotype",

            "final_label",

            "disease",

            "tissue",

            "subtype",

            "confidence",

            "matched_column",

            "matched_keyword",

            "annotation_status"

        ]

        df = df[cols]

        outfile = (

            self.output_dir /
            f"{gse}_annotation.csv"

        )

        df.to_csv(outfile, index=False)

        print()

        print("Annotation Summary")

        print(df["annotation_status"].value_counts())

        print()

        print("Phenotype Summary")

        print(df["phenotype"].value_counts())

        print()

        print("Saved :", outfile)

        print()

    def run(self):

        files = sorted(

            self.input_dir.glob(
                "*_parsed_metadata.csv"
            )

        )

        for file in files:

            self.process_dataset(file)

        print("=" * 90)
        print("Sample Annotation Completed")
        print("=" * 90)


if __name__ == "__main__":

    SampleAnnotationEngine().run()