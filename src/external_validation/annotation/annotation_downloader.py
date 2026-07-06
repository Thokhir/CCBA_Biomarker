from pathlib import Path
import GEOparse
import pandas as pd


class AnnotationDownloader:

    def __init__(self, platform_id):

        self.platform_id = platform_id

        self.base_dir = Path(__file__).resolve().parents[3]

        self.annotation_dir = (
            self.base_dir /
            "data" /
            "geo" /
            "annotations"
        )

        self.annotation_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def download(self):

        print("="*70)
        print(f"Downloading {self.platform_id}")
        print("="*70)

        self.gpl = GEOparse.get_GEO(
            geo=self.platform_id,
            destdir=str(self.annotation_dir)
        )

    def inspect(self):

        print("\nPlatform Metadata")

        print("="*70)

        for key, value in self.gpl.metadata.items():

            print(f"{key}: {value}")

        print("\n")

        print("Annotation Columns")

        print("="*70)

        print(list(self.gpl.table.columns))

    def save_annotation(self):

        output = (
            self.annotation_dir /
            f"{self.platform_id}_annotation.csv"
        )

        self.gpl.table.to_csv(
            output,
            index=False
        )

        print("\nAnnotation Saved")

        print(output)

    def summary(self):

        print("\n")

        print("="*70)

        print("Annotation Shape")

        print(self.gpl.table.shape)

        print("="*70)

        print(self.gpl.table.head())



if __name__ == "__main__":

    platforms = [

        "GPL10558",

        "GPL6104",

        "GPL8432"

    ]

    for platform in platforms:

        downloader = AnnotationDownloader(
            platform
        )

        downloader.download()

        downloader.inspect()

        downloader.summary()

        downloader.save_annotation()