import pandas as pd

df = pd.read_csv(
    r"D:\BIO_IT_PROJECTS\CCA_BIOMARKER_PLATFORM\results\aligned_external_cohorts\GSE26566_aligned.csv"
)

print(df.columns.tolist())