"""
==============================================================================
External Prediction Engine
==============================================================================

Project
-------
Cholangiocarcinoma Biomarker Discovery Platform

Purpose
-------
Apply the trained clinical Random Forest model to independent GEO cohorts
and generate prediction probabilities for external validation.

Inputs
------
results/
    trained_model/
        rf_model.pkl
        feature_order.csv
        selected_features.csv
        model_metadata.json

results/
    aligned_external_cohorts/
        *_aligned.csv

Outputs
-------
results/
    external_predictions/
        GSE26566_predictions.csv
        GSE32225_predictions.csv
        GSE89749_predictions.csv

results/
    prediction_reports/
        prediction_summary.csv
        prediction_log.txt

Author
------
Shaik Basha

==============================================================================
"""

from pathlib import Path
from datetime import datetime
import platform
import json
import joblib
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

ANNOTATION_COLUMNS = [

    "sample_id",

    "phenotype",

    "annotation_status"

]

# =============================================================================
# CLASS
# =============================================================================

class ExternalPredictionEngine:

    """
    Production-grade prediction engine
    for external validation cohorts.
    """

    def __init__(self):

        self.base = Path(__file__).resolve().parents[3]

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.model_dir = (

            self.base /

            "results" /

            "trained_model"

        )

        self.model_file = (

            self.model_dir /

            "rf_model.pkl"

        )

        self.feature_order_file = (

            self.model_dir /

            "feature_order.csv"

        )

        self.metadata_file = (

            self.model_dir /

            "model_metadata.json"

        )

        # --------------------------------------------------
        # External cohorts
        # --------------------------------------------------

        self.cohort_dir = (

            self.base /

            "results" /

            "aligned_external_cohorts"

        )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        self.output_dir = (

            self.base /

            "results" /

            "external_predictions"

        )

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        self.report_dir = (

            self.base /

            "results" /

            "prediction_reports"

        )

        self.report_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        # --------------------------------------------------
        # Objects
        # --------------------------------------------------

        self.model = None

        self.feature_order = None

        self.model_metadata = None

        self.cohort_files = []

        self.summary = []

# =============================================================================
# FILE CHECK
# =============================================================================

    def check_files(self):

        print("=" * 90)
        print("CHECKING INPUT FILES")
        print("=" * 90)

        required = [

            self.model_file,

            self.feature_order_file,

            self.metadata_file

        ]

        for file in required:

            if not file.exists():

                raise FileNotFoundError(

                    f"\nMissing file\n{file}"

                )

            print("FOUND :", file.name)

        if not self.cohort_dir.exists():

            raise FileNotFoundError(

                self.cohort_dir

            )

        self.cohort_files = sorted(

            self.cohort_dir.glob(

                "*_aligned.csv"

            )

        )

        if len(self.cohort_files) == 0:

            raise FileNotFoundError(

                "No aligned cohorts found."

            )

        print()

        print("=" * 90)
        print("EXTERNAL COHORTS FOUND")
        print("=" * 90)

        for f in self.cohort_files:

            print(f.name)

        print()

# =============================================================================
# LOAD MODEL
# =============================================================================

    def load_model(self):

        print("=" * 90)
        print("LOADING TRAINED MODEL")
        print("=" * 90)

        self.model = joblib.load(

            self.model_file

        )

        self.feature_order = pd.read_csv(

            self.feature_order_file

        )["gene_name"].tolist()

        with open(

            self.metadata_file,

            "r"

        ) as f:

            self.model_metadata = json.load(f)

        print(

            "Model Loaded Successfully"

        )

        print(

            "Expected Biomarkers :",

            len(self.feature_order)

        )

        print()

# =============================================================================
# DISCOVER COHORTS
# =============================================================================

    def discover_cohorts(self):

        print("=" * 90)
        print("COHORT DISCOVERY")
        print("=" * 90)

        print(

            "Number of Cohorts :",

            len(self.cohort_files)

        )

        total_samples = 0

        for file in self.cohort_files:

            df = pd.read_csv(file)

            total_samples += len(df)

            print(

                f"{file.stem} : {len(df)} samples"

            )

        print()

        print(

            "Total Samples :",

            total_samples

        )

        print()

# =============================================================================
# DATASET QC
# =============================================================================

    def validate_dataset(self, df, cohort):

        print("-" * 70)
        print(cohort)
        print("-" * 70)

        print(

            "Dataset Shape :",

            df.shape

        )

        required = [

            "sample_id",

            "phenotype",

            "annotation_status"

        ]

        missing = [

            c

            for c in required

            if c not in df.columns

        ]

        if len(missing):

            raise ValueError(

                f"{cohort}\n"

                f"Missing required columns\n"

                f"{missing}"

            )

        print("Required Columns : OK")

        if "final_label" in df.columns:

            print("Using final_label")

        elif "label" in df.columns:

            print("Using label")

        else:

            print("Creating labels from phenotype")

            df["label"] = df["phenotype"].map({ 
                "Tumor": 1,

                "Normal": 0

            })

        duplicate_samples = df["sample_id"].duplicated().sum()

        if duplicate_samples > 0:

            raise ValueError(

                f"{cohort}\n"

                f"Duplicate sample IDs detected."

            )

        print(

            "Annotation Columns : OK"

        )

        print(

            "Duplicate Samples :",

            duplicate_samples

        )

        print()

# =============================================================================
# PREPARE DATASET
# =============================================================================

    def prepare_dataset(self, cohort_file):

        cohort = cohort_file.stem.replace(

            "_aligned",

            ""

        )

        print("=" * 90)
        print("PREPARING :", cohort)
        print("=" * 90)

        df = pd.read_csv(

            cohort_file

        )

        self.validate_dataset(

            df,

            cohort

        )

        # --------------------------------------------------
        # Annotation
        # --------------------------------------------------

        annotation = df.copy()

        # --------------------------------------------------
        # Expression Matrix
        # --------------------------------------------------

        expression = df.drop(

            columns=ANNOTATION_COLUMNS

        )

        print(

            "Expression Shape :",

            expression.shape

        )

        # --------------------------------------------------
        # Feature availability
        # --------------------------------------------------

        available = []

        missing = []

        for gene in self.feature_order:

            if gene in expression.columns:

                available.append(gene)

            else:

                missing.append(gene)

        print(

            "Expected Features :",

            len(self.feature_order)

        )

        print(

            "Available Features :",

            len(available)

        )

        print(

            "Missing Features :",

            len(missing)

        )

        if len(missing) > 0:

            print()

            print("Missing Biomarkers")

            for gene in missing:

                print(" -", gene)

        # --------------------------------------------------
        # Add missing biomarkers
        # --------------------------------------------------

        for gene in missing:

            expression[gene] = 0.0

        # --------------------------------------------------
        # Exact feature ordering
        # --------------------------------------------------

        expression = expression[

            self.feature_order

        ]

        print()

        print(

            "Prediction Matrix :",

            expression.shape

        )

        print()

        return (

            cohort,

            annotation,

            expression,

            missing

        )


# =============================================================================
# QUALITY CHECK
# =============================================================================

    def verify_prediction_matrix(

        self,

        expression,

        cohort

    ):

        print(

            "Verifying Feature Matrix..."

        )

        if list(expression.columns) != self.feature_order:

            raise ValueError(

                f"{cohort}\n"

                "Feature ordering mismatch."

            )

        if expression.isna().sum().sum() > 0:

            raise ValueError(

                f"{cohort}\n"

                "Missing values detected."

            )

        duplicate_columns = (

            expression.columns

            .duplicated()

            .sum()

        )

        if duplicate_columns > 0:

            raise ValueError(

                f"{cohort}\n"

                "Duplicate biomarkers detected."

            )

        print(

            "Feature Matrix : OK"

        )

        print()


# =============================================================================
# COHORT INFORMATION
# =============================================================================

    def summarize_dataset(

        self,

        annotation,

        cohort

    ):

        print("=" * 90)
        print("COHORT SUMMARY")
        print("=" * 90)

        print(

            "Cohort :", cohort

        )

        print(

            "Samples :",

            len(annotation)

        )

        if "phenotype" in annotation.columns:

            print()

            print(

                annotation["phenotype"]

                .value_counts()

            )

        print()

        if "label" in annotation.columns:

            print(

                "Label Distribution"

            )

            print(

                annotation["label"]

                .value_counts()

            )

        print()


# =============================================================================
# PREPROCESS ALL COHORTS
# =============================================================================

    def preprocess_all(self):

        self.datasets = []

        for cohort_file in self.cohort_files:

            (

                cohort,

                annotation,

                expression,

                missing

            ) = self.prepare_dataset(

                cohort_file

            )

            self.verify_prediction_matrix(

                expression,

                cohort

            )

            self.summarize_dataset(

                annotation,

                cohort

            )

            self.datasets.append({

                "cohort": cohort,

                "annotation": annotation,

                "expression": expression,

                "missing": missing

            })

        print("=" * 90)
        print("ALL EXTERNAL COHORTS PREPARED")
        print("=" * 90)

        print(

            "Datasets Ready :",

            len(self.datasets)

        )

        print()

# =============================================================================
# PREDICT SINGLE COHORT
# =============================================================================

    def predict_single_cohort(self, dataset):

        cohort = dataset["cohort"]

        annotation = dataset["annotation"]

        X = dataset["expression"]

        missing = dataset["missing"]

        print("=" * 90)
        print(f"PREDICTING : {cohort}")
        print("=" * 90)

        # --------------------------------------------------
        # Random Forest Prediction
        # --------------------------------------------------

        predicted_labels = self.model.predict(X)

        prediction_probabilities = self.model.predict_proba(X)[:, 1]

        # --------------------------------------------------
# Determine True Labels
# --------------------------------------------------

        if "final_label" in annotation.columns:

            true_labels = annotation["final_label"].copy()

        elif "label" in annotation.columns:

            true_labels = annotation["label"].copy()

        else:

            print("Creating labels from phenotype...")

            true_labels = (
                annotation["phenotype"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "tumor": 1,
                    "normal": 0
                })
            )

            if true_labels.isna().any():

                raise ValueError(
                    f"{cohort}\nUnable to create labels from phenotype column."
                )

        # --------------------------------------------------
        # Confidence Category
        # --------------------------------------------------

        confidence = []

        for probability in prediction_probabilities:

            if probability >= 0.90:

                confidence.append("Very High")

            elif probability >= 0.75:

                confidence.append("High")

            elif probability >= 0.60:

                confidence.append("Moderate")

            else:

                confidence.append("Low")

        # --------------------------------------------------
        # Correct / Incorrect
        # --------------------------------------------------

        # --------------------------------------------------
# Correct / Incorrect
# --------------------------------------------------

        prediction_status = []

        for true_label, predicted_label in zip(
                true_labels,
                predicted_labels):

            if int(true_label) == int(predicted_label):

                prediction_status.append("Correct")

            else:

                prediction_status.append("Incorrect")

                
        # --------------------------------------------------
        # Build Prediction Table
        # --------------------------------------------------

        prediction_table = annotation.copy()

        prediction_table["predicted_label"] = predicted_labels

        prediction_table["tumor_probability"] = np.round(

            prediction_probabilities,

            6

        )
        
        prediction_table["prediction_status"] = prediction_status

        prediction_table["confidence"] = confidence

        prediction_table["cohort"] = cohort

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        tumor_predictions = int(

            np.sum(predicted_labels == 1)

        )

        normal_predictions = int(

            np.sum(predicted_labels == 0)

        )

        correct_predictions = prediction_status.count(

            "Correct"

        )

        incorrect_predictions = prediction_status.count(

            "Incorrect"

        )

        print(

            "Samples               :",

            len(prediction_table)

        )

        print(

            "Tumor Predictions     :",

            tumor_predictions

        )

        print(

            "Normal Predictions    :",

            normal_predictions

        )

        print(

            "Correct Predictions   :",

            correct_predictions

        )

        print(

            "Incorrect Predictions :",

            incorrect_predictions

        )

        print(

            "Average Probability   :",

            round(

                prediction_probabilities.mean(),

                4

            )

        )

        print()

        # --------------------------------------------------
        # Save internally
        # --------------------------------------------------

        dataset["prediction_table"] = prediction_table

        dataset["probabilities"] = prediction_probabilities

        dataset["predictions"] = predicted_labels

        dataset["summary"] = {

            "Cohort": cohort,

            "Samples": len(prediction_table),

            "Tumor_Predictions": tumor_predictions,

            "Normal_Predictions": normal_predictions,

            "Correct": correct_predictions,

            "Incorrect": incorrect_predictions,

            "Missing_Biomarkers": len(missing),

            "Average_Probability":

                prediction_probabilities.mean()

        }

        self.summary.append(

            dataset["summary"]

        )


# =============================================================================
# RUN PREDICTIONS
# =============================================================================

    def predict_all(self):

        print("=" * 90)
        print("RUNNING EXTERNAL PREDICTIONS")
        print("=" * 90)

        for dataset in self.datasets:

            self.predict_single_cohort(

                dataset

            )

        print("=" * 90)
        print("PREDICTIONS COMPLETED")
        print("=" * 90)

        print(

            "External Cohorts Processed :",

            len(self.datasets)

        )

        print()

# =============================================================================
# EXPORT PREDICTIONS
# =============================================================================

    def export_predictions(self):

        print("=" * 90)
        print("EXPORTING PREDICTIONS")
        print("=" * 90)

        for dataset in self.datasets:

            cohort = dataset["cohort"]

            prediction_table = dataset["prediction_table"]

            outfile = (

                self.output_dir /

                f"{cohort}_predictions.csv"

            )

            prediction_table.to_csv(

                outfile,

                index=False

            )

            print("Saved :", outfile.name)

        print()


# =============================================================================
# EXPORT SUMMARY
# =============================================================================

    def export_summary(self):

        print("=" * 90)
        print("EXPORTING SUMMARY")
        print("=" * 90)

        summary_df = pd.DataFrame(

            self.summary

        )

        outfile = (

            self.report_dir /

            "prediction_summary.csv"

        )

        summary_df.to_csv(

            outfile,

            index=False

        )

        print("Saved :", outfile.name)

        print()
# =============================================================================
# WRITE EXECUTION LOG
# =============================================================================

    def write_log(self):

        logfile = (

            self.report_dir /

            "prediction_log.txt"

        )

        with open(

            logfile,

            "w",

            encoding="utf-8"

        ) as f:

            f.write("=" * 80 + "\n")

            f.write("External Prediction Engine\n")

            f.write("=" * 80 + "\n\n")

            f.write(

                f"Execution Time : {datetime.now()}\n"

            )

            f.write(

                f"Python Version : {platform.python_version()}\n"

            )

            f.write(

                f"Operating System : {platform.system()}\n"

            )

            f.write(

                f"Machine : {platform.machine()}\n\n"

            )

            f.write(

                f"Model : {self.model_file.name}\n"

            )

            f.write(

                f"Expected Biomarkers : {len(self.feature_order)}\n"

            )

            f.write(

                f"Cohorts Processed : {len(self.datasets)}\n\n"

            )

            for dataset in self.datasets:

                summary = dataset["summary"]

                f.write("-" * 60 + "\n")

                f.write(

                    f"Cohort : {summary['Cohort']}\n"

                )

                f.write(

                    f"Samples : {summary['Samples']}\n"

                )

                f.write(

                    f"Tumor Predictions : {summary['Tumor_Predictions']}\n"

                )

                f.write(

                    f"Normal Predictions : {summary['Normal_Predictions']}\n"

                )

                f.write(

                    f"Correct Predictions : {summary['Correct']}\n"

                )

                f.write(

                    f"Incorrect Predictions : {summary['Incorrect']}\n"

                )

                f.write(

                    f"Missing Biomarkers : {summary['Missing_Biomarkers']}\n"

                )

                f.write(

                    f"Average Probability : {summary['Average_Probability']:.4f}\n\n"

                )

        print("=" * 90)
        print("LOG EXPORTED")
        print("=" * 90)

        print("Saved :", logfile.name)

        print()

# =============================================================================
# RUN
# =============================================================================

    def run(self):

        self.check_files()

        self.load_model()

        self.discover_cohorts()

        self.preprocess_all()

        self.predict_all()

        self.export_predictions()

        self.export_summary()

        self.write_log()

        print("=" * 90)
        print("EXTERNAL PREDICTION ENGINE COMPLETED")
        print("=" * 90)

        print("\nGenerated Files\n")

        for file in sorted(self.output_dir.glob("*")):

            print(file.name)

        print()

        for file in sorted(self.report_dir.glob("*")):

            print(file.name)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    engine = ExternalPredictionEngine()

    engine.run()

