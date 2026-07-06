import os
import sys
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
# Define constants for directories and API endpoints
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
METADATA_DIR = BASE_DIR / "data" / "metadata"
INTERMEDIATE_DIR = BASE_DIR / "data" / "intermediate"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
# GDC API endpoints
FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
# Define filters for TCGA-CHOL gene expression quantification data
filters = {
    "op": "and",
    "content": [
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-CHOL"]
            }
        },
        {
            "op": "in",
            "content": {
                "field": "data_type",
                "value": ["Gene Expression Quantification"]
            }
        },
        {
            "op": "in",
            "content": {
                "field": "analysis.workflow_type",
                "value": ["STAR - Counts"]
            }
        }
    ]
}
# Define fields to retrieve from the GDC API
fields = [
    "file_id",
    "file_name",
    "cases.submitter_id",
    "cases.samples.sample_type"
]
# Construct parameters for the API request
params = {
    "filters": json.dumps(filters),
    "fields": ",".join(fields),
    "format": "JSON",
    "size": "100"
}
# Make the API request to retrieve file metadata
response = requests.get(FILES_ENDPOINT, params=params)
print(response.url)
print(response.status_code)
print(response.text[:1000])
# Check if the request was successful
print(response.status_code)
# Parse the JSON response
data = response.json()
# Extract the list of files (hits) from the response
hits = data["data"]["hits"]
# Print the total number of files found and the first hit to verify the structure of the data
print(f"Total files found: {len(hits)}")
if len(hits) == 0:
    print("No files returned from API. Exiting without writing metadata.")
    sys.exit(0)

# Print the first hit to verify the structure of the data
print(json.dumps(hits[0], indent=4))

# Extract relevant metadata from the hits and store it in a list of records
records = []
for hit in hits:
    file_id = hit["file_id"]
    file_name = hit["file_name"]
    case_id = hit["cases"][0]["submitter_id"]
    sample_type = hit["cases"][0]["samples"][0]["sample_type"]
    records.append({
        "file_id": file_id,
        "file_name": file_name,
        "case_id": case_id,
        "sample_type": sample_type,
    })

metadata_df = pd.DataFrame(records)
# Print the first few rows of the metadata DataFrame to verify the extracted information
print(metadata_df.head())
# Save the metadata DataFrame to a CSV file in the metadata directory
metadata_df.to_csv(
    METADATA_DIR / "tcga_chol_metadata.csv",
    index=False
)

# Create a binary label column where 'Primary Tumor' is labeled as 1 and all other sample types are labeled as 0
metadata_df["label"] = metadata_df["sample_type"].apply(
    lambda x: 1 if x == "Primary Tumor" else 0
)

print("\nColumns in metadata_df:")
print(metadata_df.columns.tolist())

print("\nDataFrame Shape:")
print(metadata_df.shape)

print("\nFirst Rows:")
print(metadata_df.head())
# Print the distribution of sample types in the metadata DataFrame to verify the labels
print(metadata_df["sample_type"].value_counts())