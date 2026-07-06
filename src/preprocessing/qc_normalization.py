import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]

matrix_file = (
    BASE_DIR /
    "data" /
    "intermediate" /
    "expression_matrix_raw_counts.csv"
)

df = pd.read_csv(matrix_file)

print(df.shape)

gene_info = df[["gene_id", "gene_name"]]

expression = df.iloc[:, 2:]

missing = expression.isnull().sum().sum()

print("Missing Values:", missing)

library_sizes = expression.sum(axis=0)

print(library_sizes.head())

plt.figure(figsize=(12,6))

library_sizes.sort_values().plot(
    kind="bar"
)

plt.title(
    "Library Size Per Sample"
)

plt.ylabel("Total Counts")

plt.tight_layout()

plt.show()

min_count = 10

min_samples = int(
    expression.shape[1] * 0.20
)

mask = (
    expression >= min_count
).sum(axis=1) >= min_samples

filtered_expression = expression[mask]
filtered_gene_info = gene_info[mask]

print("Before:", expression.shape)

print("After:", filtered_expression.shape)

cpm = (
    filtered_expression
    .div(
        filtered_expression.sum(axis=0),
        axis=1
    )
    * 1_000_000
)

log_cpm = np.log2(
    cpm + 1
)

plt.figure(figsize=(10,6))

sns.histplot(
    filtered_expression.values.flatten(),
    bins=100
)

plt.title(
    "Raw Counts Distribution"
)

plt.show()

plt.figure(figsize=(10,6))

sns.histplot(
    log_cpm.values.flatten(),
    bins=100
)

plt.title(
    "Log2 CPM Distribution"
)

plt.show()

normalized_df = pd.concat(
    [
        filtered_gene_info,
        log_cpm
    ],
    axis=1
)

output_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "expression_logCPM.csv"
)

output_file.parent.mkdir(
    exist_ok=True
)

normalized_df.to_csv(
    output_file,
    index=False
)

print("\nQC REPORT")
print("="*50)

print(
    "Genes After Filtering:",
    filtered_expression.shape[0]
)

print(
    "Samples:",
    filtered_expression.shape[1]
)

print(
    "Missing Values:",
    normalized_df.isnull().sum().sum()
)