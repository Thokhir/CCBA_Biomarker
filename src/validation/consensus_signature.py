import pandas as pd
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

stability_file = (
    BASE_DIR /
    "results" /
    "nested_cv_feature_stability.csv"
)

# =====================================================
# LOAD DATA
# =====================================================

stability_df = pd.read_csv(
    stability_file
)

print(
    "Total Biomarkers:",
    len(stability_df)
)

# =====================================================
# CLASSIFY STABILITY
# =====================================================

def classify_stability(count):

    if count == 5:
        return "Core Biomarker"

    elif count == 4:
        return "Highly Stable"

    elif count == 3:
        return "Stable"

    else:
        return "Exploratory"

stability_df["category"] = (
    stability_df["selection_count"]
    .apply(classify_stability)
)

# =====================================================
# SORT
# =====================================================

stability_df = stability_df.sort_values(
    "selection_count",
    ascending=False
)

# =====================================================
# CORE SIGNATURE
# =====================================================

core_signature = stability_df[
    stability_df["selection_count"] >= 4
]

# =====================================================
# SAVE
# =====================================================

RESULT_DIR = BASE_DIR / "results"

core_signature.to_csv(
    RESULT_DIR /
    "consensus_biomarker_signature.csv",
    index=False
)

stability_df.to_csv(
    RESULT_DIR /
    "feature_stability_report.csv",
    index=False
)

# =====================================================
# REPORT
# =====================================================

print("\n")
print("="*60)
print("FEATURE STABILITY REPORT")
print("="*60)

print(
    stability_df["category"]
    .value_counts()
)

print("\n")

print(
    "Consensus Signature Size:",
    len(core_signature)
)

print("\nTop Consensus Biomarkers")

print(
    core_signature.head(20)
)

print("\nSaved Files")

print(
    RESULT_DIR /
    "consensus_biomarker_signature.csv"
)

print(
    RESULT_DIR /
    "feature_stability_report.csv"
)