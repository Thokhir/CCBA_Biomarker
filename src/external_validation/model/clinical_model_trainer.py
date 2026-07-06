"""
==============================================================================
Clinical Model Trainer
==============================================================================

Project
-------
Cholangiocarcinoma Biomarker Discovery Platform

Purpose
-------
Train the final clinical Random Forest classifier using the stable biomarker
panel identified from nested cross-validation.

Outputs
-------
results/trained_model/

    rf_model.pkl

    selected_features.csv

    feature_order.csv

    feature_importance.csv

    cross_validation_results.csv

    training_metrics.csv

    training_metrics.json

    training_summary.txt

Author
------
Shaik Basha
==============================================================================

"""

import json
import platform

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn import __version__ as sklearn_version

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (

    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix

)

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

N_SPLITS = 5

MIN_SELECTION_COUNT = 3

RF_PARAMETERS = {

    "n_estimators":500,

    "max_depth":None,

    "min_samples_split":2,

    "min_samples_leaf":1,

    "bootstrap":True,

    "class_weight":"balanced",

    "random_state":RANDOM_STATE,

    "n_jobs":-1

}

# =============================================================================
# CLASS
# =============================================================================

class ClinicalModelTrainer:

    def __init__(self):

        self.base = Path(__file__).resolve().parents[3]

        self.expression_file = (

            self.base /
            "data" /
            "processed" /
            "expression_logCPM.csv"

        )

        self.metadata_file = (

            self.base /
            "data" /
            "metadata" /
            "tcga_chol_metadata.csv"

        )

        self.feature_file = (

            self.base /
            "results" /
            "clinical_biomarker_panel.csv"

        )

        self.output_dir = (

            self.base /
            "results" /
            "trained_model"

        )

        self.output_dir.mkdir(

            parents=True,
            exist_ok=True

        )

        self.expression = None
        self.metadata = None
        self.features = None

        self.model = None

        self.X = None
        self.y = None

        self.feature_names = None

        self.metrics = None

        self.feature_importance = None
    def check_files(self):

        print("="*90)
        print("CHECKING INPUT FILES")
        print("="*90)

        for file in [

            self.expression_file,
            self.metadata_file,
            self.feature_file

        ]:

            if not file.exists():

                raise FileNotFoundError(file)

            print("FOUND :",file.name)

        print()

    def load_data(self):

        print("="*90)
        print("LOADING DATA")
        print("="*90)

        self.expression = pd.read_csv(
            self.expression_file
        )

        self.metadata = pd.read_csv(
            self.metadata_file
        )

        self.features = pd.read_csv(
            self.feature_file
        )

        print("Expression Shape :",self.expression.shape)

        print("Metadata Shape   :",self.metadata.shape)

        print("Stable Features  :",len(self.features))

        print()

    def collapse_duplicate_genes(self):

        duplicated = self.expression["gene_name"].duplicated().sum()

        if duplicated==0:

            print("No duplicated genes detected.")

            return

        print()

        print("Duplicate genes detected :",duplicated)

        print("Collapsing duplicated genes...")

        before=len(self.expression)

        numeric_columns=self.expression.columns[2:]

        self.expression=(

            self.expression

            .groupby(

                "gene_name",

                as_index=False

            )[numeric_columns]

            .mean()

        )

        after=len(self.expression)

        print("Genes before :",before)

        print("Genes after  :",after)

        print("Removed      :",before-after)

        print()

    def quality_control(self):

        print("="*90)

        print("QUALITY CONTROL")

        print("="*90)

        if self.expression.empty:

            raise ValueError("Expression matrix empty.")

        if self.metadata.empty:

            raise ValueError("Metadata empty.")

        if self.features.empty:

            raise ValueError("Feature table empty.")

        self.collapse_duplicate_genes()

        if self.metadata["file_id"].duplicated().any():

            raise ValueError("Duplicate sample IDs detected.")

        if self.expression.isna().sum().sum()>0:

            raise ValueError("Expression matrix contains NA values.")

        # ----------------------------------------------------------
# Support both feature stability file and clinical panel
# ----------------------------------------------------------

        if "selection_count" in self.features.columns:

            self.features = self.features[
                self.features["selection_count"] >= MIN_SELECTION_COUNT
            ].copy()

        else:

            print("Clinical biomarker panel detected.")

            self.features = self.features.copy()

        self.features["gene_name"]=(
            self.features["gene_name"]
            .astype(str)
            .str.upper()
        )

        self.expression["gene_name"]=(
            self.expression["gene_name"]
            .astype(str)
            .str.upper()
        )

        print("Stable biomarkers :",len(self.features))

        print()

    def build_matrix(self):

        print("="*90)

        print("BUILDING MACHINE LEARNING MATRIX")

        print("="*90)

        available=set(self.expression["gene_name"])

        selected=[]

        missing=[]

        for gene in self.features["gene_name"]:

            if gene in available:

                selected.append(gene)

            else:

                missing.append(gene)

        print("Available biomarkers :",len(selected))

        print("Missing biomarkers   :",len(missing))

        if len(selected)==0:

            raise ValueError("No biomarker matched.")

        expr=self.expression[

            self.expression["gene_name"].isin(selected)

        ].copy()

        expr=expr.set_index("gene_name")

        self.feature_names=expr.index.tolist()

        self.X=expr.T

        if "label" not in self.metadata.columns:

            self.metadata["label"]=self.metadata["sample_type"].apply(

                lambda x:1 if x=="Primary Tumor" else 0

            )

        label_map=dict(

            zip(

                self.metadata["file_id"],

                self.metadata["label"]

            )

        )

        self.y=np.array(

            [

                label_map[s]

                for s in self.X.index

            ]

        )

        print()

        print("Machine Learning Matrix")

        print("-----------------------")

        print("Samples :",self.X.shape[0])

        print("Genes   :",self.X.shape[1])

        print()

        print("Tumor Samples :",sum(self.y==1))

        print("Normal Samples:",sum(self.y==0))

        print()

# =============================================================================
# STRATIFIED CROSS VALIDATION
# =============================================================================

    def cross_validation(self):

        print("=" * 90)
        print("RUNNING STRATIFIED CROSS VALIDATION")
        print("=" * 90)

        cv = StratifiedKFold(

            n_splits=N_SPLITS,

            shuffle=True,

            random_state=RANDOM_STATE

        )

        fold_results = []

        fold = 1

        for train_index, test_index in cv.split(self.X, self.y):

            print(f"\nFold {fold}")

            X_train = self.X.iloc[train_index]

            X_test = self.X.iloc[test_index]

            y_train = self.y[train_index]

            y_test = self.y[test_index]

            model = RandomForestClassifier(

                **RF_PARAMETERS

            )

            model.fit(

                X_train,

                y_train

            )

            predictions = model.predict(

                X_test

            )

            probabilities = model.predict_proba(

                X_test

            )[:, 1]

            accuracy = accuracy_score(

                y_test,

                predictions

            )

            auc = roc_auc_score(

                y_test,

                probabilities

            )

            precision = precision_score(

                y_test,

                predictions,

                zero_division=0

            )

            recall = recall_score(

                y_test,

                predictions,

                zero_division=0

            )

            f1 = f1_score(

                y_test,

                predictions,

                zero_division=0

            )

            balanced = balanced_accuracy_score(

                y_test,

                predictions

            )

            cm = confusion_matrix(

                y_test,

                predictions

            )

            tn, fp, fn, tp = cm.ravel()

            specificity = tn / (tn + fp)

            sensitivity = tp / (tp + fn)

            print("-" * 50)

            print(f"Accuracy           : {accuracy:.4f}")

            print(f"AUC                : {auc:.4f}")

            print(f"Precision          : {precision:.4f}")

            print(f"Recall             : {recall:.4f}")

            print(f"F1 Score           : {f1:.4f}")

            print(f"Sensitivity        : {sensitivity:.4f}")

            print(f"Specificity        : {specificity:.4f}")

            print(f"Balanced Accuracy  : {balanced:.4f}")

            print("Confusion Matrix")

            print(cm)

            fold_results.append({

                "Fold": fold,

                "Accuracy": accuracy,

                "AUC": auc,

                "Precision": precision,

                "Recall": recall,

                "Sensitivity": sensitivity,

                "Specificity": specificity,

                "BalancedAccuracy": balanced,

                "F1": f1,

                "TN": tn,

                "FP": fp,

                "FN": fn,

                "TP": tp

            })

            fold += 1

        self.metrics = pd.DataFrame(

            fold_results

        )

        outfile = (

            self.output_dir /

            "cross_validation_results.csv"

        )

        self.metrics.to_csv(

            outfile,

            index=False

        )

        print()

        print("=" * 90)

        print("CROSS VALIDATION SUMMARY")

        print("=" * 90)

        summary = pd.DataFrame({

            "Metric": [

                "Accuracy",

                "AUC",

                "Precision",

                "Recall",

                "Sensitivity",

                "Specificity",

                "BalancedAccuracy",

                "F1"

            ],

            "Mean": [

                self.metrics["Accuracy"].mean(),

                self.metrics["AUC"].mean(),

                self.metrics["Precision"].mean(),

                self.metrics["Recall"].mean(),

                self.metrics["Sensitivity"].mean(),

                self.metrics["Specificity"].mean(),

                self.metrics["BalancedAccuracy"].mean(),

                self.metrics["F1"].mean()

            ],

            "Std": [

                self.metrics["Accuracy"].std(),

                self.metrics["AUC"].std(),

                self.metrics["Precision"].std(),

                self.metrics["Recall"].std(),

                self.metrics["Sensitivity"].std(),

                self.metrics["Specificity"].std(),

                self.metrics["BalancedAccuracy"].std(),

                self.metrics["F1"].std()

            ]

        })

        print(summary)

        print()

        print("Cross Validation Results Saved")

        print(outfile)

        print()

# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

    def train_final_model(self):

        print("=" * 90)
        print("TRAINING FINAL CLINICAL MODEL")
        print("=" * 90)

        self.model = RandomForestClassifier(

            **RF_PARAMETERS

        )

        self.model.fit(

            self.X,

            self.y

        )

        print("Final Random Forest model trained successfully.")

        print("Number of Samples :", len(self.X))

        print("Number of Features:", len(self.feature_names))

        print()

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

    def create_feature_importance(self):

        print("=" * 90)
        print("CREATING FEATURE IMPORTANCE")
        print("=" * 90)

        importance = pd.DataFrame({

            "gene_name": self.feature_names,

            "importance": self.model.feature_importances_

        })

        importance = importance.sort_values(

            by="importance",

            ascending=False

        ).reset_index(drop=True)

        importance["rank"] = np.arange(

            1,

            len(importance) + 1

        )

        importance = importance[

            [

                "rank",

                "gene_name",

                "importance"

            ]

        ]

        # ---------------------------------------------------------
        # Save Feature Importance
        # ---------------------------------------------------------

        feature_file = (

            self.output_dir /

            "feature_importance.csv"

        )

        importance.to_csv(

            feature_file,

            index=False

        )

        # ---------------------------------------------------------
        # Selected Features
        # ---------------------------------------------------------

        selected = importance[

            [

                "gene_name"

            ]

        ]

        selected_file = (

            self.output_dir /

            "selected_features.csv"

        )

        selected.to_csv(

            selected_file,

            index=False

        )

        # ---------------------------------------------------------
        # Feature Order
        # ---------------------------------------------------------

        feature_order = pd.DataFrame({

            "feature_order": np.arange(

                len(self.feature_names)

            ),

            "gene_name": self.feature_names

        })

        order_file = (

            self.output_dir /

            "feature_order.csv"

        )

        feature_order.to_csv(

            order_file,

            index=False

        )

        print("Top 20 Important Biomarkers")

        print(

            importance.head(20)

        )

        print()

        print("Saved")

        print(feature_file)

        print(selected_file)

        print(order_file)

        print()

# =============================================================================
# SAVE TRAINED MODEL
# =============================================================================

    def save_model(self):

        print("=" * 90)
        print("SAVING MODEL")
        print("=" * 90)

        model_file = (

            self.output_dir /

            "rf_model.pkl"

        )

        joblib.dump(

            self.model,

            model_file

        )

        print("Model Saved")

        print(model_file)

        print()

# =============================================================================
# SAVE FEATURE METADATA
# =============================================================================

    def save_feature_metadata(self):

        metadata = {

            "training_samples": int(

                len(self.X)

            ),

            "training_features": int(

                len(self.feature_names)

            ),

            "random_state": RANDOM_STATE,

            "python_version": platform.python_version(),

            "sklearn_version": sklearn_version,

            "training_date": datetime.now().strftime(

                "%Y-%m-%d %H:%M:%S"

            ),

            "parameters": RF_PARAMETERS

        }

        with open(

            self.output_dir /

            "model_metadata.json",

            "w"

        ) as f:

            json.dump(

                metadata,

                f,

                indent=4

            )

        print("Model Metadata Saved")

        print()

# =============================================================================
# SAVE METRICS
# =============================================================================

    def save_metrics(self):

        print("=" * 90)
        print("SAVING TRAINING METRICS")
        print("=" * 90)

        metrics_file = (

            self.output_dir /

            "training_metrics.csv"

        )

        self.metrics.to_csv(

            metrics_file,

            index=False

        )

        summary = {

            "Accuracy_mean":

                float(self.metrics["Accuracy"].mean()),

            "Accuracy_std":

                float(self.metrics["Accuracy"].std()),

            "AUC_mean":

                float(self.metrics["AUC"].mean()),

            "AUC_std":

                float(self.metrics["AUC"].std()),

            "Precision_mean":

                float(self.metrics["Precision"].mean()),

            "Recall_mean":

                float(self.metrics["Recall"].mean()),

            "Sensitivity_mean":

                float(self.metrics["Sensitivity"].mean()),

            "Specificity_mean":

                float(self.metrics["Specificity"].mean()),

            "BalancedAccuracy_mean":

                float(self.metrics["BalancedAccuracy"].mean()),

            "F1_mean":

                float(self.metrics["F1"].mean())

        }

        json_file = (

            self.output_dir /

            "training_metrics.json"

        )

        with open(

            json_file,

            "w"

        ) as f:

            json.dump(

                summary,

                f,

                indent=4

            )

        print("Saved")

        print(metrics_file)

        print(json_file)

        print()


# =============================================================================
# SAVE TRAINING SUMMARY
# =============================================================================

    def save_training_summary(self):

        print("=" * 90)
        print("WRITING TRAINING SUMMARY")
        print("=" * 90)

        report = []

        report.append("=" * 70)
        report.append("CLINICAL RANDOM FOREST MODEL")
        report.append("=" * 70)

        report.append("")

        report.append(
            f"Training Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        report.append(
            f"Python Version : {platform.python_version()}"
        )

        report.append(
            f"Scikit-learn   : {sklearn_version}"
        )

        report.append("")

        report.append(
            f"Samples : {len(self.X)}"
        )

        report.append(
            f"Biomarkers : {len(self.feature_names)}"
        )

        report.append("")

        report.append("Random Forest Parameters")

        for k, v in RF_PARAMETERS.items():

            report.append(f"{k} : {v}")

        report.append("")
        report.append("=" * 70)
        report.append("CROSS VALIDATION PERFORMANCE")
        report.append("=" * 70)

        report.append(
            f"Accuracy            : {self.metrics['Accuracy'].mean():.4f}"
        )

        report.append(
            f"AUC                 : {self.metrics['AUC'].mean():.4f}"
        )

        report.append(
            f"Precision           : {self.metrics['Precision'].mean():.4f}"
        )

        report.append(
            f"Recall              : {self.metrics['Recall'].mean():.4f}"
        )

        report.append(
            f"Sensitivity         : {self.metrics['Sensitivity'].mean():.4f}"
        )

        report.append(
            f"Specificity         : {self.metrics['Specificity'].mean():.4f}"
        )

        report.append(
            f"Balanced Accuracy   : {self.metrics['BalancedAccuracy'].mean():.4f}"
        )

        report.append(
            f"F1 Score            : {self.metrics['F1'].mean():.4f}"
        )

        report.append("")
        report.append("=" * 70)

        outfile = (

            self.output_dir /

            "training_summary.txt"

        )

        with open(

            outfile,

            "w",

            encoding="utf-8"

        ) as f:

            f.write(

                "\n".join(report)

            )

        print("Saved")

        print(outfile)

        print()


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

    def run(self):

        self.check_files()

        self.load_data()

        self.quality_control()

        self.build_matrix()

        self.cross_validation()

        self.train_final_model()

        self.create_feature_importance()

        self.save_model()

        self.save_feature_metadata()

        self.save_metrics()

        self.save_training_summary()

        print("=" * 90)
        print("CLINICAL MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 90)

        print()

        print("Outputs")

        for f in sorted(

            self.output_dir.glob("*")

        ):

            print(" -", f.name)

        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    trainer = ClinicalModelTrainer()

    trainer.run()