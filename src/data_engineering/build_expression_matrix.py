"""
=========================================================
TCGA-CHOL RNA-Seq Expression Matrix Builder
Project: CCA Biomarker Discovery Platform
Version: Day 4 Industrial Pipeline
=========================================================
"""

import pandas as pd
from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

COUNT_DIR = BASE_DIR / "data" / "raw" / "counts"
INTERMEDIATE_DIR = BASE_DIR / "data" / "intermediate"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# FIND ALL COUNT FILES
# =========================================================

count_files = sorted(COUNT_DIR.glob("*.tsv"))

print("=" * 70)
print("TCGA CHOL Expression Matrix Builder")
print("=" * 70)

print(f"Files found: {len(count_files)}")

if len(count_files) == 0:
    raise SystemExit(
        "ERROR: No count files found in data/raw/counts/"
    )

# =========================================================
# TCGA STAR COUNT COLUMNS
# =========================================================

COUNT_COLUMNS = [
    "unstranded",
    "stranded_first",
    "stranded_second"
]

# =========================================================
# MASTER MATRIX
# =========================================================

master_df = None

# =========================================================
# PROCESS EACH FILE
# =========================================================

for i, file in enumerate(count_files, start=1):

    print(f"\n[{i}/{len(count_files)}] Processing: {file.name}")

    # -----------------------------------------------------
    # Read file
    # -----------------------------------------------------

    df = pd.read_csv(
        file,
        sep="\t",
        comment="#"
    )

    # -----------------------------------------------------
    # Verify required columns
    # -----------------------------------------------------

    required_columns = [
        "gene_id",
        "gene_name"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"{file.name} missing required column: {col}"
            )

    # -----------------------------------------------------
    # Identify count column
    # -----------------------------------------------------

    count_column = next(
        (
            c for c in COUNT_COLUMNS
            if c in df.columns
        ),
        None
    )

    if count_column is None:

        raise ValueError(
            f"\nNo supported count column found in:\n"
            f"{file.name}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # -----------------------------------------------------
    # Remove QC rows
    # -----------------------------------------------------

    df = df[
        ~df["gene_id"].astype(str).str.startswith("N_")
    ].copy()

    # -----------------------------------------------------
    # Keep only required columns
    # -----------------------------------------------------

    sample_name = file.stem

    df = df[
        [
            "gene_id",
            "gene_name",
            count_column
        ]
    ]

    # -----------------------------------------------------
    # Rename count column to sample name
    # -----------------------------------------------------

    df.columns = [
        "gene_id",
        "gene_name",
        sample_name
    ]

    # -----------------------------------------------------
    # Build master matrix
    # -----------------------------------------------------

    if master_df is None:

        master_df = df

    else:

        master_df = master_df.merge(
            df[
                [
                    "gene_id",
                    sample_name
                ]
            ],
            on="gene_id",
            how="inner"
        )

# =========================================================
# QC REPORT
# =========================================================

print("\n")
print("=" * 70)
print("MATRIX QC")
print("=" * 70)

print("Shape:")
print(master_df.shape)

print("\nColumns:")
print(master_df.columns[:10])

print("\nFirst 5 rows:")
print(master_df.head())

print("\nMissing values:")
print(master_df.isnull().sum().sum())

# =========================================================
# SAVE MATRIX
# =========================================================

OUTPUT_FILE = (
    INTERMEDIATE_DIR /
    "expression_matrix_raw_counts.csv"
)

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved Successfully")

print(f"\nOutput File:\n{OUTPUT_FILE}")

# =========================================================
# BASIC STATISTICS
# =========================================================

expression_only = master_df.iloc[:, 2:]

print("\nExpression Statistics:")
print(expression_only.describe())

print("\nPipeline Completed Successfully")


print(master_df.shape)

print(master_df.head())

print(master_df.iloc[:5,:10])