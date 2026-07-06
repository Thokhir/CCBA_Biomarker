import pandas as pd
from pathlib import Path


class PhenotypeDetector:

    def __init__(self, gse):

        self.gse = gse

        self.base_dir = Path(__file__).resolve().parents[3]

        self.metadata_file = (
            self.base_dir
            / "data"
            / "geo"
            / "metadata"
            / f"{gse}_sample_metadata.csv"
        )

        self.output_file = (
            self.base_dir
            / "data"
            / "geo"
            / "metadata"
            / f"{gse}_phenotype.csv"
        )

    def load_metadata(self):

        self.df = pd.read_csv(self.metadata_file)

        print("=" * 80)
        print(self.gse)
        print("Samples :", len(self.df))

    def detect_column(self):

        candidate_columns = []

        for col in self.df.columns:

            text = " ".join(
                self.df[col].astype(str).fillna("").tolist()
            ).lower()

            keywords = [
                "tumor",
                "normal",
                "cancer",
                "cholangiocarcinoma",
                "control",
                "adjacent",
                "non-tumor",
                "healthy",
                "cca"
            ]

            score = sum(
                word in text
                for word in keywords
            )

            if score > 0:
                candidate_columns.append((col, score))

        candidate_columns.sort(
            key=lambda x: x[1],
            reverse=True
        )

        if len(candidate_columns) == 0:

            raise Exception(
                "No phenotype column detected."
            )

        self.phenotype_column = candidate_columns[0][0]

        print()
        print("Detected phenotype column:")
        print(self.phenotype_column)

    def assign_labels(self):

        labels = []

        phenotype = []

        for value in self.df[self.phenotype_column]:

            text = str(value).lower()

            if any(
                x in text
                for x in [
                    "normal",
                    "control",
                    "healthy",
                    "adjacent",
                    "non-tumor",
                    "nontumor"
                ]
            ):

                labels.append(0)
                phenotype.append("Normal")

            elif any(
                x in text
                for x in [
                    "tumor",
                    "cancer",
                    "cholangiocarcinoma",
                    "cca"
                ]
            ):

                labels.append(1)
                phenotype.append("Tumor")

            else:

                labels.append(-1)
                phenotype.append("Unknown")

        self.df["phenotype"] = phenotype
        self.df["label"] = labels

    def summary(self):

        print()
        print("=" * 80)
        print("Phenotype Summary")

        print(self.df["phenotype"].value_counts())

        print()

        print("Label Summary")

        print(self.df["label"].value_counts())

    def save(self):

        self.df.to_csv(
            self.output_file,
            index=False
        )

        print()
        print("Saved:")
        print(self.output_file)


if __name__ == "__main__":

    datasets = [

        "GSE89749",
        "GSE26566",
        "GSE32225"

    ]

    for gse in datasets:

        detector = PhenotypeDetector(gse)

        detector.load_metadata()

        detector.detect_column()

        detector.assign_labels()

        detector.summary()

        detector.save()