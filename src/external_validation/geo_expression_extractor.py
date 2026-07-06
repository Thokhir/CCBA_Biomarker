import GEOparse
import pandas as pd
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

GSE_IDS = [
    "GSE89749",
    "GSE26566",
    "GSE32225"
]

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = (
    BASE_DIR /
    "data" /
    "geo" /
    "raw"
)

PROCESSED_DIR = (
    BASE_DIR /
    "data" /
    "geo" /
    "processed"
)

METADATA_DIR = (
    BASE_DIR /
    "data" /
    "geo" /
    "metadata"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD GEO DATASET
# =====================================================

for gse_id in GSE_IDS:

    print(f"\nDownloading {gse_id}")

    gse = GEOparse.get_GEO(
        geo=gse_id,
        destdir=str(RAW_DIR)
    )

    print(f"{gse_id} Download Complete")

# =====================================================
# SAMPLE INFORMATION
# =====================================================

sample_metadata = []

expression_matrix = None

# =====================================================
# LOOP THROUGH ALL GSM SAMPLES
# =====================================================

for gsm_name, gsm in gse.gsms.items():

    print(f"Processing {gsm_name}")

    metadata = {
        "sample_id": gsm_name
    }

    for key, value in gsm.metadata.items():

        metadata[key] = "; ".join(value)

    sample_metadata.append(metadata)

    table = gsm.table.copy()

    value_column = None

    possible_columns = [
        "VALUE",
        "value",
        "Signal",
        "signal"
    ]

    for col in possible_columns:

        if col in table.columns:
            value_column = col
            break

    if value_column is None:
        print(f"No expression column in {gsm_name}")
        continue

    probe_column = table.columns[0]

    sample_df = table[
        [probe_column, value_column]
    ].copy()

    sample_df.columns = [
        "Probe_ID",
        gsm_name
    ]

    if expression_matrix is None:

        expression_matrix = sample_df

    else:

        expression_matrix = expression_matrix.merge(
            sample_df,
            on="Probe_ID",
            how="outer"
        )

# =====================================================
# SAVE SAMPLE METADATA
# =====================================================

metadata_df = pd.DataFrame(
    sample_metadata
)

metadata_file = (
    METADATA_DIR /
    f"{GSE_ID}_sample_metadata.csv"
)

metadata_df.to_csv(
    metadata_file,
    index=False
)

# =====================================================
# SAVE EXPRESSION MATRIX
# =====================================================

expression_file = (
    PROCESSED_DIR /
    f"{GSE_ID}_expression_matrix.csv"
)

expression_matrix.to_csv(
    expression_file,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n==============================")
print("GEO Extraction Completed")
print("==============================")

print("Samples :", len(metadata_df))

print("Probes  :", len(expression_matrix))

print("\nMetadata Saved:")
print(metadata_file)

print("\nExpression Matrix Saved:")
print(expression_file)

print("\nMatrix Shape:")
print(expression_matrix.shape)