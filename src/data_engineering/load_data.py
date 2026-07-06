import pandas as pd

sample_data = {
    "Gene": ["TP53", "KRAS", "EGFR"],
    "Sample_1": [1000, 500, 8000],
    "Sample_2": [900, 450, 7500]
}

df = pd.DataFrame(sample_data)

# print(df)

print(df.shape)

df.to_csv("data/intermediate/sample_expression.csv", index=False)