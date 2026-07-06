import pandas as pd
import mygene

from pathlib import Path

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

signature_file = (
    BASE_DIR /
    "results" /
    "consensus_biomarker_signature.csv"
)

# =====================================================
# LOAD SIGNATURE
# =====================================================

signature_df = pd.read_csv(
    signature_file
)

genes = signature_df[
    "gene_name"
].tolist()

print("Genes Found:")
print(genes)

# =====================================================
# MYGENE CLIENT
# =====================================================

mg = mygene.MyGeneInfo()

records = []

# =====================================================
# QUERY EACH GENE
# =====================================================

for gene in genes:

    try:

        result = mg.query(
            gene,
            species="human",
            size=1
        )

        if len(result["hits"]) == 0:

            print(
                f"No match found: {gene}"
            )

            continue

        hit = result["hits"][0]

        approved_symbol = hit.get(
            "symbol",
            ""
        )

        entrez = hit.get(
            "entrezgene",
            ""
        )

        alias = hit.get(
            "alias",
            ""
        )

        ensembl = ""

        if "ensembl" in hit:

            if isinstance(
                hit["ensembl"],
                list
            ):

                ensembl = hit[
                    "ensembl"
                ][0].get(
                    "gene",
                    ""
                )

            else:

                ensembl = hit[
                    "ensembl"
                ].get(
                    "gene",
                    ""
                )

        records.append({

            "input_gene":
                gene,

            "approved_symbol":
                approved_symbol,

            "entrez_id":
                entrez,

            "ensembl_id":
                ensembl,

            "aliases":
                alias

        })

    except Exception as e:

        print(
            f"Error for {gene}: {e}"
        )

annotation_df = pd.DataFrame(
    records
)

# =====================================================
# SAVE
# =====================================================

RESULT_DIR = (
    BASE_DIR /
    "results"
)

annotation_df.to_csv(
    RESULT_DIR /
    "biomarker_annotation_table.csv",
    index=False
)

print("\nAnnotation Table")

print(annotation_df)

print(
    "\nSaved:"
)

print(
    RESULT_DIR /
    "biomarker_annotation_table.csv"
)