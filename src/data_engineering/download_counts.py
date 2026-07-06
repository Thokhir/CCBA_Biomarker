import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
METADATA_DIR = BASE_DIR / "data" / "metadata"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# LOAD METADATA
metadata_file = METADATA_DIR / "tcga_chol_metadata.csv"

metadata_df = pd.read_csv(metadata_file)

print(metadata_df.head())

# FUNCTION TO DOWNLOAD FILES
def download_file(file_id, output_path):

    url = f"https://api.gdc.cancer.gov/data/{file_id}"

    response = requests.get(url, stream=True)

    if response.status_code == 200:

        with open(output_path, "wb") as file:

            for chunk in response.iter_content(chunk_size=8192):

                file.write(chunk)

        return True

    return False

# DOWNLOAD COUNT FILES
COUNT_DIR = RAW_DATA_DIR / "counts"

COUNT_DIR.mkdir(parents=True, exist_ok=True)

# Iterate through metadata and download count files
for _, row in tqdm(metadata_df.iterrows(),
                   total=len(metadata_df)):

    file_id = row["file_id"]

    filename = f"{file_id}.tsv"

    output_path = COUNT_DIR / filename

    if output_path.exists():
        continue

    success = download_file(
        file_id,
        output_path
    )

    if success:
        print(f"Downloaded {filename}")