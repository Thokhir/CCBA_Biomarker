"""
Metadata Keyword Library
------------------------
Centralized dictionaries used for automatic sample annotation.
"""

# -------------------------
# Phenotype Keywords
# -------------------------

PHENOTYPE_KEYWORDS = {

    "Tumor": [

        "tumor",
        "tumour",
        "cancer",
        "cholangiocarcinoma",
        "carcinoma",
        "cca",
        "malignant",
        "primary tumor",
        "primary tumour"

    ],

    "Normal": [

        "normal",
        "control",
        "healthy",
        "non tumor",
        "non-tumor",
        "adjacent",
        "surrounding liver",
        "bile duct",
        "benign"

    ]
}

# -------------------------
# Tissue Keywords
# -------------------------

TISSUE_KEYWORDS = {

    "Liver": [

        "liver",
        "hepatic",
        "surrounding liver"

    ],

    "Bile Duct": [

        "bile duct",
        "biliary"

    ],

    "Tumor Tissue": [

        "tumor",
        "tumour"

    ]
}

# -------------------------
# Disease Keywords
# -------------------------

DISEASE_KEYWORDS = {

    "Cholangiocarcinoma": [

        "cholangiocarcinoma",
        "cca",
        "intrahepatic cholangiocarcinoma"

    ],

    "Healthy": [

        "healthy",
        "normal"

    ]
}

# -------------------------
# Molecular Subtypes
# -------------------------

SUBTYPE_KEYWORDS = {

    "Inflammation": [

        "inflammation"

    ],

    "Proliferation": [

        "proliferation"

    ]
}

# -------------------------
# Priority Columns
# -------------------------

COLUMN_PRIORITY = [

    "characteristics_ch1",
    "source_name_ch1",
    "description",
    "title"

]