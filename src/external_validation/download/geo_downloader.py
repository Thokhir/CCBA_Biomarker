import GEOparse
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

GSE_ID = "GSE89749"

BASE_DIR = Path(__file__).resolve().parents[2]

GEO_RAW_DIR = (
    BASE_DIR /
    "data" /
    "geo" /
    "raw"
)

GEO_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# DOWNLOAD GEO DATASET
# =====================================================

print(f"\nDownloading {GSE_ID} ...")

gse = GEOparse.get_GEO(
    geo=GSE_ID,
    destdir=str(GEO_RAW_DIR)
)

print("\nDownload Complete")

print("\nDataset Information")
print("=" * 60)

print("GSE ID:", gse.get_accession())

print("Title:")
print(gse.metadata.get("title"))

print("\nPlatform(s):")
print(gse.gpls.keys())

print("\nSamples:")
print(len(gse.gsms))

# =====================================================
# SAVE SAMPLE LIST
# =====================================================

sample_file = (
    BASE_DIR /
    "data" /
    "geo" /
    "metadata" /
    f"{GSE_ID}_samples.txt"
)

sample_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(sample_file, "w") as f:

    for gsm in gse.gsms:

        f.write(f"{gsm}\n")

print("\nSaved sample list:")
print(sample_file)