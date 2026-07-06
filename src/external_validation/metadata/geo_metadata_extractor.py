from pathlib import Path
import GEOparse
import pandas as pd


class GEOMetadataExtractor:

    def __init__(self):

        self.base_dir = Path(__file__).resolve().parents[3]

        self.raw_dir = self.base_dir / "data" / "geo" / "raw"

        self.output_dir = self.base_dir / "data" / "geo" / "metadata"

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_datasets(self):

        self.soft_files = sorted(self.raw_dir.glob("*_family.soft.gz"))

        print("=" * 80)
        print("Datasets Found")

        for f in self.soft_files:
            print(f.name)

        print()

    def process_dataset(self, soft_file):

        gse = soft_file.name.replace("_family.soft.gz", "")

        print("=" * 80)
        print(gse)

        geo = GEOparse.get_GEO(filepath=str(soft_file))

        metadata = []

        for gsm_name, gsm in geo.gsms.items():

            row = {}

            row["sample_id"] = gsm_name

            for key, value in gsm.metadata.items():

                if isinstance(value, list):

                    row[key] = " | ".join(value)

                else:

                    row[key] = value

            metadata.append(row)

        df = pd.DataFrame(metadata)

        outfile = self.output_dir / f"{gse}_sample_metadata.csv"

        df.to_csv(outfile, index=False)

        print(f"Samples : {len(df)}")
        print(f"Columns : {len(df.columns)}")
        print("Saved :", outfile)

    def run(self):

        self.discover_datasets()

        for soft in self.soft_files:

            self.process_dataset(soft)

        print()
        print("=" * 80)
        print("Metadata extraction completed.")
        print("=" * 80)


if __name__ == "__main__":

    extractor = GEOMetadataExtractor()

    extractor.run()