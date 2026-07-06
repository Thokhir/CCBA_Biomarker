import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)
import sys


BASE_DIR = Path(__file__).resolve().parents[2]

# ensure src is on sys.path so utils can be imported when running as a script
sys.path.insert(0, str(BASE_DIR / "src"))
from utils.plot_utils import save_fig

matrix_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "expression_logCPM.csv"
)

df = pd.read_csv(matrix_file)

print(df.shape)

gene_info = df[
    ["gene_id", "gene_name"]
]

expression = df.iloc[:, 2:]

X = expression.T

print(X.shape)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(
    X_scaled
)


print(
    pca.explained_variance_ratio_
)


pca_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Sample": X.index
})


plt.figure(
    figsize=(10,8)
)

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2"
)

plt.title(
    "PCA of TCGA-CHOL Samples"
)

plt.tight_layout()

# save plot (name only needs to change per plot)
save_fig("pca_scatter")
plt.show()



pca_df["distance"] = np.sqrt(
    pca_df["PC1"]**2 +
    pca_df["PC2"]**2
)


threshold = (
    pca_df["distance"].mean() +
    3 * pca_df["distance"].std()
)

outliers = pca_df[
    pca_df["distance"] > threshold
]

print(outliers)


correlation_matrix = X.T.corr()



plt.figure(
    figsize=(12,10)
)

sns.heatmap(
    correlation_matrix,
    cmap="viridis"
)

plt.title(
    "Sample Correlation"
)

save_fig("sample_correlation")
plt.show()


linked = linkage(
    X_scaled,
    method="ward"
)


plt.figure(
    figsize=(12,8)
)

dendrogram(
    linked,
    labels=X.index,
    leaf_rotation=90
)

plt.title(
    "Sample Clustering"
)

plt.tight_layout()

save_fig("sample_clustering")
plt.show()


OUTPUT_DIR = (
    BASE_DIR /
    "results"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

pca_df.to_csv(
    OUTPUT_DIR /
    "pca_results.csv",
    index=False
)


outliers.to_csv(
    OUTPUT_DIR /
    "potential_outliers.csv",
    index=False
)



print("\nQC SUMMARY")
print("="*50)

print(
    "Samples:",
    X.shape[0]
)

print(
    "Genes:",
    X.shape[1]
)

print(
    "Potential Outliers:",
    len(outliers)
)