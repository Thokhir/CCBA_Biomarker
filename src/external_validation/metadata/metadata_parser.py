from pathlib import Path
import pandas as pd
import re

from keyword_library import (
    PHENOTYPE_KEYWORDS,
    DISEASE_KEYWORDS,
    TISSUE_KEYWORDS,
    SUBTYPE_KEYWORDS,
    COLUMN_PRIORITY
)


class MetadataParser:

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
            "parsed_metadata"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def normalize(self, text):

        if pd.isna(text):
            return ""

        text = str(text).lower()

        text = re.sub(r"[^a-z0-9 ]", " ", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def search_dictionary(self, text, dictionary):

        text = self.normalize(text)

        for label, keywords in dictionary.items():

            for keyword in keywords:

                if keyword.lower() in text:

                    return label, keyword

        return None, None

    def parse_dataset(self, file):

        gse = file.stem.replace("_sample_metadata", "")

        print("=" * 90)
        print(gse)

        df = pd.read_csv(file)

        parsed = []

        for _, row in df.iterrows():

            phenotype = "Unknown"
            disease = "Unknown"
            tissue = "Unknown"
            subtype = "NA"

            matched_column = "NA"
            matched_keyword = "NA"

            confidence = 0.0

            for column in COLUMN_PRIORITY:

                if column not in row.index:
                    continue

                value = str(row[column])

                p, pk = self.search_dictionary(
                    value,
                    PHENOTYPE_KEYWORDS
                )

                d, dk = self.search_dictionary(
                    value,
                    DISEASE_KEYWORDS
                )

                t, tk = self.search_dictionary(
                    value,
                    TISSUE_KEYWORDS
                )

                s, sk = self.search_dictionary(
                    value,
                    SUBTYPE_KEYWORDS
                )

                if p:

                    phenotype = p

                    matched_column = column

                    matched_keyword = pk

                    confidence = 0.95

                if d:
                    disease = d

                if t:
                    tissue = t

                if s:
                    subtype = s

            parsed.append({

                "sample_id":
                    row["sample_id"],

                "phenotype":
                    phenotype,

                "disease":
                    disease,

                "tissue":
                    tissue,

                "subtype":
                    subtype,

                "matched_column":
                    matched_column,

                "matched_keyword":
                    matched_keyword,

                "confidence":
                    confidence

            })

        parsed = pd.DataFrame(parsed)

        outfile = (
            self.output_dir /
            f"{gse}_parsed_metadata.csv"
        )

        parsed.to_csv(outfile, index=False)

        print(parsed.head())

        print()

        print(parsed["phenotype"].value_counts())

        print()

        print("Saved :", outfile)

    def run(self):

        files = sorted(

            self.metadata_dir.glob(
                "*_sample_metadata.csv"
            )

        )

        for file in files:

            self.parse_dataset(file)


if __name__ == "__main__":

    MetadataParser().run()