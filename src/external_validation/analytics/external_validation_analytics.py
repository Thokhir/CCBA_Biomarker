"""
==============================================================================
External Validation Analytics Engine
==============================================================================

Project
-------
Cholangiocarcinoma Biomarker Discovery Platform

Module
------
External Validation Analytics

Purpose
-------
Evaluate the diagnostic performance of the trained clinical biomarker model
using independent external GEO cohorts.

Features
--------
• Cohort-wise diagnostic evaluation
• Overall pooled evaluation
• ROC analysis
• Precision-Recall analysis
• Calibration analysis
• Bootstrap confidence intervals
• Publication-ready figures
• Publication-ready tables
• JSON/TXT reports
• Execution logging

Inputs
------
results/
    external_predictions/
        *_predictions.csv

results/
    trained_model/
        model_metadata.json

Outputs
-------
results/
    external_validation/

Author
------
Shaik Basha

Version
-------
2.0 Production

==============================================================================
"""

# =============================================================================
# STANDARD LIBRARY
# =============================================================================

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# SCIENTIFIC COMPUTING
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# VISUALIZATION
# =============================================================================

import matplotlib.pyplot as plt

# =============================================================================
# STATISTICS
# =============================================================================

from scipy import stats

# =============================================================================
# MACHINE LEARNING METRICS
# =============================================================================

from sklearn.metrics import (

    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    auc

)

from sklearn.calibration import calibration_curve

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

BOOTSTRAP_ITERATIONS = 2000

CONFIDENCE_LEVEL = 0.95

FIGURE_DPI = 300

POSITIVE_LABEL = 1

NEGATIVE_LABEL = 0

LOG_FILE = "execution_log.txt"

REPORT_NAME = "validation_report"

# =============================================================================
# CLASS
# =============================================================================


class ExternalValidationAnalytics:

    """
    Production External Validation Analytics Engine
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self):

        self.base = Path(__file__).resolve().parents[3]

        # ---------------------------------------------------------------------
        # INPUT DIRECTORIES
        # ---------------------------------------------------------------------

        self.prediction_dir = (

            self.base /
            "results" /
            "external_predictions"

        )

        self.model_dir = (

            self.base /
            "results" /
            "trained_model"

        )

        # ---------------------------------------------------------------------
        # OUTPUT DIRECTORIES
        # ---------------------------------------------------------------------

        self.output_dir = (

            self.base /
            "results" /
            "external_validation"

        )

        self.figure_dir = (

            self.output_dir /
            "figures"

        )

        self.table_dir = (

            self.output_dir /
            "tables"

        )

        self.report_dir = (

            self.output_dir /
            "reports"

        )

        self.log_dir = (

            self.output_dir /
            "logs"

        )

        # ---------------------------------------------------------------------
        # CREATE OUTPUT DIRECTORIES
        # ---------------------------------------------------------------------

        for directory in [

            self.output_dir,
            self.figure_dir,
            self.table_dir,
            self.report_dir,
            self.log_dir

        ]:

            directory.mkdir(

                parents=True,
                exist_ok=True

            )

        # ---------------------------------------------------------------------
        # LOGGER
        # ---------------------------------------------------------------------

        self.logger = None

        # ---------------------------------------------------------------------
        # MODEL METADATA
        # ---------------------------------------------------------------------

        self.metadata = {}

        # ---------------------------------------------------------------------
        # INPUT FILES
        # ---------------------------------------------------------------------

        self.prediction_files = []

        # ---------------------------------------------------------------------
        # DATASET REGISTRY
        # ---------------------------------------------------------------------

        self.datasets = {}

        # ---------------------------------------------------------------------
        # RESULTS CONTAINERS
        # ---------------------------------------------------------------------

        self.performance = []

        self.overall_metrics = {}

        self.roc_results = {}

        self.pr_results = {}

        self.calibration_results = {}

        self.bootstrap_results = {}

        self.summary = {}

        self.execution_start = datetime.now()

        self.execution_end = None
        # =========================================================================
    # LOGGER
    # =========================================================================

    def setup_logger(self):

        """
        Configure console and file logger.
        """

        self.logger = logging.getLogger(
            "ExternalValidationAnalytics"
        )

        self.logger.setLevel(
            logging.INFO
        )

        self.logger.handlers.clear()

        formatter = logging.Formatter(

            "%(asctime)s | %(levelname)s | %(message)s"

        )

        file_handler = logging.FileHandler(

            self.log_dir / LOG_FILE,

            mode="w"

        )

        file_handler.setFormatter(
            formatter
        )

        console_handler = logging.StreamHandler(
            sys.stdout
        )

        console_handler.setFormatter(
            formatter
        )

        self.logger.addHandler(
            file_handler
        )

        self.logger.addHandler(
            console_handler
        )

        self.logger.info(
            "=" * 80
        )

        self.logger.info(
            "External Validation Analytics Started"
        )

        self.logger.info(
            "=" * 80
        )

    # =========================================================================
    # CHECK INPUT FILES
    # =========================================================================

    def check_files(self):

        self.logger.info("Checking input files...")

        if not self.prediction_dir.exists():

            raise FileNotFoundError(

                f"Prediction directory not found:\n"
                f"{self.prediction_dir}"

            )

        if not self.model_dir.exists():

            raise FileNotFoundError(

                f"Model directory not found:\n"
                f"{self.model_dir}"

            )

        metadata_file = (

            self.model_dir /

            "model_metadata.json"

        )

        if not metadata_file.exists():

            raise FileNotFoundError(

                metadata_file

            )

        self.prediction_files = sorted(

            self.prediction_dir.glob(

                "*_predictions.csv"

            )

        )

        if len(self.prediction_files) == 0:

            raise FileNotFoundError(

                "No prediction files found."

            )

        self.logger.info(

            "Prediction Files Found : %d",

            len(self.prediction_files)

        )

        for file in self.prediction_files:

            self.logger.info(

                file.name

            )

    # =========================================================================
    # LOAD MODEL METADATA
    # =========================================================================

    def load_metadata(self):

        self.logger.info(
            "Loading model metadata..."
        )

        metadata_file = (

            self.model_dir /

            "model_metadata.json"

        )

        with open(

            metadata_file,

            "r",

            encoding="utf-8"

        ) as f:

            self.metadata = json.load(f)

        self.logger.info(

            "Model Name : %s",

            self.metadata.get(

                "model_name",

                "Unknown"

            )

        )

        self.logger.info(

            "Algorithm : %s",

            self.metadata.get(

                "algorithm",

                "Random Forest"

            )

        )

        self.logger.info(

            "Training Samples : %s",

            self.metadata.get(

                "training_samples",

                "Unknown"

            )

        )

        self.logger.info(

            "Number of Biomarkers : %s",

            self.metadata.get(

                "number_of_features",

                "Unknown"

            )

        )

    # =========================================================================
    # LOAD PREDICTIONS
    # =========================================================================

    def load_predictions(self):

        self.logger.info(

            "Loading prediction files..."

        )

        self.datasets = {}

        for file in self.prediction_files:

            cohort = (

                file.stem

                .replace(

                    "_predictions",

                    ""

                )

            )

            df = pd.read_csv(file)

            self.datasets[cohort] = {

                "data": df,

                "samples": len(df),

                "source_file": file.name

            }

            self.logger.info(

                "%s : %d samples",

                cohort,

                len(df)

            )

        self.logger.info(

            "Datasets Loaded : %d",

            len(self.datasets)

        )

    # =========================================================================
    # REGISTER DATASETS
    # =========================================================================

    def register_datasets(self):

        self.logger.info(

            "Registering datasets..."

        )

        for cohort in self.datasets:

            df = self.datasets[cohort]["data"]

            tumor = 0

            normal = 0

            if "phenotype" in df.columns:

                phenotype = (

                    df["phenotype"]

                    .astype(str)

                    .str.lower()

                )

                tumor = (

                    phenotype == "tumor"

                ).sum()

                normal = (

                    phenotype == "normal"

                ).sum()

            self.datasets[cohort]["tumor"] = int(tumor)

            self.datasets[cohort]["normal"] = int(normal)

            self.logger.info(

                "%s -> Tumor:%d Normal:%d",

                cohort,

                tumor,

                normal

            )

    # =========================================================================
    # QUALITY CONTROL
    # =========================================================================

    def quality_control(self):

        """
        Validate all loaded prediction datasets.
        """

        self.logger.info("Running quality control...")

        for cohort_name, cohort_info in self.datasets.items():

            df = cohort_info["data"]

            # ---------------------------------------
            # Basic required columns
            # ---------------------------------------

            required = [

                "sample_id",

                "phenotype"

            ]

            missing = [

                c

                for c in required

                if c not in df.columns

            ]

            if missing:

                raise ValueError(

                    f"{cohort_name}\nMissing columns:\n{missing}"

                )

            # ---------------------------------------
            # Prediction column
            # ---------------------------------------

            if (

                "prediction" not in df.columns

                and

                "predicted_label" not in df.columns

            ):

                raise ValueError(

                    f"{cohort_name}\nPrediction column missing."

                )

        # ---------------------------------------
        # Probability column
        # ---------------------------------------

        if (

            "probability" not in df.columns

            and

            "tumor_probability" not in df.columns

        ):

            raise ValueError(

                f"{cohort_name}\nProbability column missing."

            )

        self.logger.info(

            f"{cohort_name}: QC passed."

        )

        self.logger.info("Quality control completed.\n")

        # =========================================================================
    # STANDARDIZE LABELS
    # =========================================================================

    def standardize_labels(self, df):

        """
        Convert available label columns into binary labels.

        Returns
        -------
        numpy.ndarray
            Binary labels
            Tumor = 1
            Normal = 0
        """

        if "label" in df.columns:

            labels = df["label"]

        elif "final_label" in df.columns:

            labels = df["final_label"]

        elif "phenotype" in df.columns:

            labels = df["phenotype"]

        else:

            raise ValueError(
                "No label column found."
            )

        labels = (

            labels
            .astype(str)
            .str.strip()
            .str.lower()

        )

        mapping = {

            "tumor":1,
            "primary tumor":1,
            "cancer":1,
            "cholangiocarcinoma":1,
            "cca":1,
            "1":1,
            "true":1,

            "normal":0,
            "healthy":0,
            "control":0,
            "adjacent":0,
            "0":0,
            "false":0

        }

        y_true = labels.map(mapping)

        if y_true.isna().any():

            unknown = sorted(

                labels[
                    y_true.isna()
                ].unique()

            )

            raise ValueError(

                f"Unknown labels detected:\n{unknown}"

            )

        return y_true.astype(int).values

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    def compute_confusion_matrix(

        self,
        y_true,
        y_pred

    ):

        """
        Compute confusion matrix.

        Returns
        -------
        TN FP FN TP
        """

        tn, fp, fn, tp = confusion_matrix(

            y_true,
            y_pred

        ).ravel()

        return (

            tn,
            fp,
            fn,
            tp

        )

    # =========================================================================
    # DIAGNOSTIC METRICS
    # =========================================================================

    def diagnostic_metrics(

        self,
        y_true,
        y_pred,
        probabilities

    ):

        """
        Calculate complete diagnostic statistics.
        """

        tn, fp, fn, tp = self.compute_confusion_matrix(

            y_true,
            y_pred

        )

        accuracy = accuracy_score(

            y_true,
            y_pred

        )

        balanced_accuracy = balanced_accuracy_score(

            y_true,
            y_pred

        )

        precision = precision_score(

            y_true,
            y_pred,
            zero_division=0

        )

        recall = recall_score(

            y_true,
            y_pred,
            zero_division=0

        )

        sensitivity = recall

        specificity = (

            tn /

            (tn + fp)

            if (tn + fp) > 0

            else np.nan

        )

        f1 = f1_score(

            y_true,
            y_pred,
            zero_division=0

        )

        roc_auc = roc_auc_score(

            y_true,
            probabilities

        )

        mcc = matthews_corrcoef(

            y_true,
            y_pred

        )

        kappa = cohen_kappa_score(

            y_true,
            y_pred

        )

        brier = brier_score_loss(

            y_true,
            probabilities

        )

        ppv = precision

        npv = (

            tn /

            (tn + fn)

            if (tn + fn) > 0

            else np.nan

        )

        fpr = (

            fp /

            (fp + tn)

            if (fp + tn) > 0

            else np.nan

        )

        fnr = (

            fn /

            (fn + tp)

            if (fn + tp) > 0

            else np.nan

        )
        fnr = (

            fn /

            (fn + tp)

            if (fn + tp) > 0

            else np.nan

        )

                # -------------------------------------------------------------
        # Likelihood Ratios
        # -------------------------------------------------------------

        if specificity == 1:

            lr_positive = np.inf

        else:

            lr_positive = (

                sensitivity /

                (1 - specificity)

            )

        if specificity == 0:

            lr_negative = np.inf

        else:

            lr_negative = (

                (1 - sensitivity) /

                specificity

            )

        # -------------------------------------------------------------
        # Diagnostic Odds Ratio
        # -------------------------------------------------------------

        if np.isinf(lr_positive):

            diagnostic_odds_ratio = np.inf

        elif lr_negative == 0:

            diagnostic_odds_ratio = np.inf

        else:

            diagnostic_odds_ratio = (

                lr_positive /

                lr_negative

            )

        # -------------------------------------------------------------
        # Return Metrics
        # -------------------------------------------------------------

        return {

            "Accuracy": accuracy,

            "Balanced_Accuracy": balanced_accuracy,

            "Sensitivity": sensitivity,

            "Specificity": specificity,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "ROC_AUC": roc_auc,

            "MCC": mcc,

            "Kappa": kappa,

            "PPV": ppv,

            "NPV": npv,

            "FPR": fpr,

            "FNR": fnr,

            "LR_Positive": lr_positive,

            "LR_Negative": lr_negative,

            "Diagnostic_Odds_Ratio": diagnostic_odds_ratio,

            "Brier_Score": brier,

            "TP": tp,

            "TN": tn,

            "FP": fp,

            "FN": fn

        }

    # =========================================================================
    # PRINT METRICS
    # =========================================================================

    def print_metrics(

        self,
        metrics

    ):

        """
        Print publication-quality diagnostic statistics.
        """

        self.logger.info("-" * 60)

        self.logger.info("Diagnostic Performance")

        self.logger.info("-" * 60)

        ordered_metrics = [

            "Accuracy",

            "Balanced_Accuracy",

            "Sensitivity",

            "Specificity",

            "Precision",

            "Recall",

            "F1",

            "ROC_AUC",

            "MCC",

            "Kappa",

            "PPV",

            "NPV",

            "FPR",

            "FNR",

            "LR_Positive",

            "LR_Negative",

            "Diagnostic_Odds_Ratio",

            "Brier_Score"

        ]

        for metric in ordered_metrics:

            value = metrics.get(metric, np.nan)

            if np.isinf(value):

                display = "Inf"

            elif pd.isna(value):

                display = "NA"

            else:

                display = f"{value:.4f}"

            self.logger.info(

                f"{metric:<28}: {display}"

            )

    # =========================================================================
    # PRINT CONFUSION MATRIX
    # =========================================================================

    def print_confusion_matrix(

        self,
        metrics

    ):

        """
        Print confusion matrix.
        """

        matrix = np.array([

            [

                metrics["TN"],

                metrics["FP"]

            ],

            [

                metrics["FN"],

                metrics["TP"]

            ]

        ])

        self.logger.info("")

        self.logger.info("Confusion Matrix")

        self.logger.info(matrix)

        self.logger.info("")

    # =============================================================================
    # EVALUATE SINGLE COHORT
    # =============================================================================

    def evaluate_single_cohort(
        self,
        cohort_name,
        cohort_info
    ):

        self.logger.info("=" * 80)
        self.logger.info(f"Evaluating {cohort_name}")
        self.logger.info("=" * 80)

        # ----------------------------------------------------------
        # Dataset
        # ----------------------------------------------------------

        if isinstance(cohort_info, dict):

            df = cohort_info["data"]

        else:

            df = cohort_info

        # ----------------------------------------------------------
        # True labels
        # ----------------------------------------------------------

        y_true = self.standardize_labels(df)

        # ----------------------------------------------------------
        # Prediction column
        # ----------------------------------------------------------

        if "prediction" in df.columns:

            y_pred = (
                df["prediction"]
                .astype(int)
                .values
            )

        elif "predicted_label" in df.columns:

            y_pred = (
                df["predicted_label"]
                .astype(int)
                .values
            )

        else:

            raise ValueError(
                f"{cohort_name}\nPrediction column not found."
            )

        # ----------------------------------------------------------
        # Probability column
        # ----------------------------------------------------------

        if "probability" in df.columns:

            y_prob = (
                df["probability"]
                .astype(float)
                .values
            )

        elif "tumor_probability" in df.columns:

            y_prob = (
                df["tumor_probability"]
                .astype(float)
                .values
            )

        else:

            raise ValueError(
                f"{cohort_name}\nProbability column not found."
            )

        # ----------------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        # ----------------------------------------------------------
        # Diagnostic Metrics
        # ----------------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        balanced_accuracy = balanced_accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        sensitivity = recall

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

        mcc = matthews_corrcoef(
            y_true,
            y_pred
        )

        kappa = cohen_kappa_score(
            y_true,
            y_pred
        )

        ppv = precision

        npv = (
            tn / (tn + fn)
            if (tn + fn) > 0
            else np.nan
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        )

        fnr = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else np.nan
        )

        # ----------------------------------------------------------
        # Likelihood Ratios
        # ----------------------------------------------------------

        if specificity == 1:

            lr_positive = np.inf

        else:

            lr_positive = sensitivity / (1 - specificity)

        if specificity == 0:

            lr_negative = np.inf

        else:

            lr_negative = (
                1 - sensitivity
            ) / specificity

        if (
            lr_negative == 0
            or
            np.isinf(lr_positive)
        ):

            dor = np.inf

        else:

            dor = lr_positive / lr_negative

        brier = brier_score_loss(
            y_true,
            y_prob
        )

        # ----------------------------------------------------------
        # Store Results
        # ----------------------------------------------------------

        metrics = {

            "Cohort": cohort_name,

            "Accuracy": accuracy,

            "Balanced_Accuracy": balanced_accuracy,

            "Sensitivity": sensitivity,

            "Specificity": specificity,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "ROC_AUC": roc_auc,

            "MCC": mcc,

            "Kappa": kappa,

            "PPV": ppv,

            "NPV": npv,

            "FPR": fpr,

            "FNR": fnr,

            "LR_Positive": lr_positive,

            "LR_Negative": lr_negative,

            "Diagnostic_Odds_Ratio": dor,

            "Brier_Score": brier,

            "TP": tp,

            "TN": tn,

            "FP": fp,

            "FN": fn

        }

        self.performance.append(metrics)

        self.logger.info(
            f"Accuracy : {accuracy:.4f}"
        )

        self.logger.info(
            f"ROC AUC  : {roc_auc:.4f}"
        )

        self.logger.info(
            f"Sensitivity : {sensitivity:.4f}"
        )

        self.logger.info(
            f"Specificity : {specificity:.4f}"
        )

        self.logger.info(
            f"F1 Score : {f1:.4f}"
        )

        self.logger.info(
            f"MCC : {mcc:.4f}"
        )

        self.logger.info(
            f"Kappa : {kappa:.4f}"
        )

        self.logger.info("Evaluation completed.\n")

        # ----------------------------------------------------------
        # ROC
        # ----------------------------------------------------------

        fpr, tpr, roc_thresholds = roc_curve(

            y_true,
            y_prob

        )

        self.roc_results[cohort_name] = pd.DataFrame({

            "False_Positive_Rate": fpr,

            "True_Positive_Rate": tpr,

            "Threshold": roc_thresholds

        })

        # ----------------------------------------------------------
        # Precision-Recall Curve
        # ----------------------------------------------------------

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(

            y_true,

            y_prob

        )

        pr_df = pd.DataFrame({

            "Recall": recall_curve[:-1],

            "Precision": precision_curve[:-1],

            "Threshold": pr_thresholds

        })

        self.pr_results[cohort_name] = pr_df

        # ----------------------------------------------------------
        # Calibration
        # ----------------------------------------------------------

        fraction_positive, mean_probability = calibration_curve(

            y_true,

            y_prob,

            n_bins=10,

            strategy="uniform"

        )

        calibration_df = pd.DataFrame({

            "Mean_Predicted_Probability": mean_probability,

            "Fraction_of_Positives": fraction_positive

        })

        self.calibration_results[cohort_name] = calibration_df

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------

        self.print_metrics(metrics)

        self.print_confusion_matrix(metrics)

        self.logger.info(

            "%s evaluation completed.",

            cohort_name

        )

        self.logger.info("")

    # =========================================================================
    # EVALUATE ALL COHORTS
    # =========================================================================

    def evaluate_all_cohorts(self):

        """
        Evaluate all independent validation cohorts.
        """

        self.logger.info("=" * 80)
        self.logger.info("External Validation")
        self.logger.info("=" * 80)

        self.performance = []

        self.roc_results = {}

        self.pr_results = {}

        self.calibration_results = {}

        for cohort_name, cohort_info in self.datasets.items():

            self.evaluate_single_cohort(

                cohort_name,

                cohort_info

            )

        # ----------------------------------------------------------
        # Export Cohort Metrics
        # ----------------------------------------------------------

        self.performance = pd.DataFrame(

            self.performance

        )

        output_file = (

            self.output_dir /

            "cohort_metrics.csv"

        )

        self.performance.to_csv(

            output_file,

            index=False

        )

        self.logger.info(

            "Cohort metrics exported."

        )

        self.logger.info(output_file)

        self.logger.info("")

        return self.performance
        # =========================================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # =========================================================================

    def bootstrap_confidence_intervals(self):

        """
        Estimate 95% bootstrap confidence intervals for each cohort.
        """

        self.logger.info("=" * 80)
        self.logger.info("BOOTSTRAP CONFIDENCE INTERVALS")
        self.logger.info("=" * 80)

        self.bootstrap_results = {}

        bootstrap_summary = []

        rng = np.random.RandomState(RANDOM_STATE)

        for cohort_name, cohort_info in self.datasets.items():

            self.logger.info(f"Bootstrapping {cohort_name}")

            df = cohort_info["data"].copy()

            y_true = self.standardize_labels(df)

            if "prediction" in df.columns:

                y_pred = df["prediction"].astype(int).values

            elif "predicted_label" in df.columns:

                y_pred = df["predicted_label"].astype(int).values

            else:

                raise ValueError(

                    f"{cohort_name}\nPrediction column missing."

                )

            if "probability" in df.columns:

                probabilities = df["probability"].astype(float).values

            elif "tumor_probability" in df.columns:

                probabilities = df["tumor_probability"].astype(float).values

            else:

                raise ValueError(

                    f"{cohort_name}\nProbability column missing."

                )

            n = len(df)

            accuracy_values = []

            auc_values = []

            f1_values = []

            sensitivity_values = []

            specificity_values = []

            for _ in range(BOOTSTRAP_ITERATIONS):

                index = rng.choice(

                    np.arange(n),

                    size=n,

                    replace=True

                )

                yt = y_true[index]

                yp = y_pred[index]

                pr = probabilities[index]

                if len(np.unique(yt)) < 2:

                    continue

                m = self.diagnostic_metrics(

                    yt,

                    yp,

                    pr

                )

                accuracy_values.append(m["Accuracy"])

                auc_values.append(m["ROC_AUC"])

                f1_values.append(m["F1"])

                sensitivity_values.append(m["Sensitivity"])

                specificity_values.append(m["Specificity"])

            def ci(values):

                return (

                    np.percentile(values,2.5),

                    np.percentile(values,97.5)

                )

            acc_low,acc_high = ci(accuracy_values)

            auc_low,auc_high = ci(auc_values)

            f1_low,f1_high = ci(f1_values)

            sen_low,sen_high = ci(sensitivity_values)

            spe_low,spe_high = ci(specificity_values)

            bootstrap_summary.append({

                "Cohort":cohort_name,

                "Accuracy_Lower":acc_low,
                "Accuracy_Upper":acc_high,

                "ROC_AUC_Lower":auc_low,
                "ROC_AUC_Upper":auc_high,

                "F1_Lower":f1_low,
                "F1_Upper":f1_high,

                "Sensitivity_Lower":sen_low,
                "Sensitivity_Upper":sen_high,

                "Specificity_Lower":spe_low,
                "Specificity_Upper":spe_high

            })

            self.bootstrap_results[cohort_name] = {

                "Accuracy":accuracy_values,

                "ROC_AUC":auc_values,

                "F1":f1_values,

                "Sensitivity":sensitivity_values,

                "Specificity":specificity_values

            }

        bootstrap_summary = pd.DataFrame(

            bootstrap_summary

        )

        bootstrap_summary.to_csv(

            self.output_dir /

            "bootstrap_confidence_intervals.csv",

            index=False

        )

        self.logger.info("Bootstrap completed.")
        self.logger.info("")

    # =========================================================================
    # OVERALL VALIDATION
    # =========================================================================

    def overall_validation(self):

        """
        Pool all external cohorts together and compute
        overall diagnostic performance.
        """

        self.logger.info("=" * 80)
        self.logger.info("OVERALL EXTERNAL VALIDATION")
        self.logger.info("=" * 80)

        all_frames = []

        for cohort in self.datasets.values():

            all_frames.append(

                cohort["data"]

            )

        overall = pd.concat(

            all_frames,

            ignore_index=True

        )

        y_true = self.standardize_labels(

            overall

        )

        if "prediction" in overall.columns:

            y_pred = overall["predicted_label"].astype(int).values

        else:

            y_pred = overall["predicted_label"].astype(int).values

        if "probability" in overall.columns:

            y_prob = overall["tumor_probability"].astype(float).values

        else:

            y_prob = overall["tumor_probability"].astype(float).values

        metrics = self.diagnostic_metrics(

            y_true,

            y_pred,

            y_prob

        )

        metrics["Samples"] = len(overall)

        metrics["Tumor"] = int(

            np.sum(y_true==1)

        )

        metrics["Normal"] = int(

            np.sum(y_true==0)

        )

        overall_df = pd.DataFrame(

            [metrics]

        )

        overall_df.to_csv(

            self.output_dir /

            "overall_metrics.csv",

            index=False

        )

        self.logger.info("Overall validation completed.")
        self.logger.info("")

        return overall_df
        # =========================================================================
    # ROC CURVES
    # =========================================================================

    def create_roc_figures(self):

        """
        Create publication-quality ROC curves.
        """

        self.logger.info("=" * 80)
        self.logger.info("CREATING ROC FIGURES")
        self.logger.info("=" * 80)

        plt.figure(figsize=(8,8))

        for cohort, roc_df in self.roc_results.items():

            auc_value = self.performance.loc[
                self.performance["Cohort"]==cohort,
                "ROC_AUC"
            ].values[0]

            plt.plot(

                roc_df["False_Positive_Rate"],
                roc_df["True_Positive_Rate"],
                linewidth=2,
                label=f"{cohort} (AUC={auc_value:.3f})"

            )

        plt.plot(
            [0,1],
            [0,1],
            "--",
            linewidth=1
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("External Validation ROC Curves")

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(

            self.figure_dir /
            "roc_curves.png",

            dpi=FIGURE_DPI

        )

        plt.close()

    # =========================================================================
    # PRECISION RECALL CURVES
    # =========================================================================

    def create_pr_figures(self):

        plt.figure(figsize=(8,8))

        for cohort, pr_df in self.pr_results.items():

            plt.plot(

                pr_df["Recall"],
                pr_df["Precision"],
                linewidth=2,
                label=cohort

            )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision-Recall Curves")

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(

            self.figure_dir /
            "precision_recall_curves.png",

            dpi=FIGURE_DPI

        )

        plt.close()

    # =========================================================================
    # CALIBRATION CURVES
    # =========================================================================

    def create_calibration_figures(self):

        plt.figure(figsize=(8,8))

        for cohort, cal_df in self.calibration_results.items():

            plt.plot(

                cal_df["Mean_Predicted_Probability"],

                cal_df["Fraction_of_Positives"],

                marker="o",

                linewidth=2,

                label=cohort

            )

        plt.plot(

            [0,1],

            [0,1],

            "--"

        )

        plt.xlabel(

            "Mean Predicted Probability"

        )

        plt.ylabel(

            "Observed Frequency"

        )

        plt.title(

            "Calibration Curves"

        )

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(

            self.figure_dir /

            "calibration_curves.png",

            dpi=FIGURE_DPI

        )

        plt.close()

    # =========================================================================
    # CONFUSION MATRICES
    # =========================================================================

    def create_confusion_matrix_figures(self):

        """
        Publication-quality confusion matrices.
        """

        self.logger.info(

            "Generating confusion matrices..."

        )

        for _, row in self.performance.iterrows():

            cohort = row["Cohort"]

            matrix = np.array([

                [row["TN"],row["FP"]],

                [row["FN"],row["TP"]]

            ])

            plt.figure(figsize=(5,5))

            plt.imshow(

                matrix,

                interpolation="nearest"

            )

            plt.colorbar()

            plt.xticks(

                [0,1],

                ["Normal","Tumor"]

            )

            plt.yticks(

                [0,1],

                ["Normal","Tumor"]

            )

            plt.xlabel(

                "Predicted"

            )

            plt.ylabel(

                "Actual"

            )

            plt.title(

                cohort

            )

            for i in range(2):

                for j in range(2):

                    plt.text(

                        j,

                        i,

                        int(matrix[i,j]),

                        ha="center",

                        va="center",

                        fontsize=12,

                        fontweight="bold"

                    )

            plt.tight_layout()

            plt.savefig(

                self.figure_dir /

                f"{cohort}_confusion_matrix.png",

                dpi=FIGURE_DPI

            )

            plt.close()

    # =========================================================================
    # BOOTSTRAP DISTRIBUTIONS
    # =========================================================================

    def create_bootstrap_figures(self):

        """
        Distribution of bootstrap ROC AUC.
        """

        for cohort, values in self.bootstrap_results.items():

            plt.figure(figsize=(7,5))

            plt.hist(

                values["ROC_AUC"],

                bins=40

            )

            plt.xlabel(

                "ROC AUC"

            )

            plt.ylabel(

                "Frequency"

            )

            plt.title(

                f"{cohort} Bootstrap ROC AUC"

            )

            plt.tight_layout()

            plt.savefig(

                self.figure_dir /

                f"{cohort}_bootstrap_auc.png",

                dpi=FIGURE_DPI

            )

            plt.close()

    # =========================================================================
    # GENERATE ALL FIGURES
    # =========================================================================

    def create_all_figures(self):

        self.logger.info("="*80)

        self.logger.info(

            "GENERATING PUBLICATION FIGURES"

        )

        self.logger.info("="*80)

        self.create_roc_figures()

        self.create_pr_figures()

        self.create_calibration_figures()

        self.create_confusion_matrix_figures()

        self.create_bootstrap_figures()

        self.logger.info(

            "All figures generated successfully."

        )

        self.logger.info("")

        # =========================================================================
    # EXPORT ANALYSIS DATA
    # =========================================================================

    def export_analysis_data(self):

        """
        Export ROC, PR, and calibration data for each cohort.
        """

        self.logger.info("=" * 80)
        self.logger.info("EXPORTING ANALYSIS DATA")
        self.logger.info("=" * 80)

        roc_dir = self.output_dir / "roc_data"
        pr_dir = self.output_dir / "pr_data"
        calibration_dir = self.output_dir / "calibration_data"

        roc_dir.mkdir(exist_ok=True)
        pr_dir.mkdir(exist_ok=True)
        calibration_dir.mkdir(exist_ok=True)

        for cohort, df in self.roc_results.items():

            df.to_csv(

                roc_dir / f"{cohort}_roc.csv",

                index=False

            )

        for cohort, df in self.pr_results.items():

            df.to_csv(

                pr_dir / f"{cohort}_precision_recall.csv",

                index=False

            )

        for cohort, df in self.calibration_results.items():

            df.to_csv(

                calibration_dir / f"{cohort}_calibration.csv",

                index=False

            )

        self.logger.info("Analysis data exported.")
        self.logger.info("")

    # =========================================================================
    # JSON SUMMARY
    # =========================================================================

    def export_json_summary(self):

        """
        Export JSON summary.
        """

        if self.execution_end is None:

            self.execution_end = datetime.now()

        summary = {

            "model": self.metadata,

            "number_of_external_cohorts": len(self.datasets),

            "total_samples": int(

                sum(

                    d["samples"]

                    for d in self.datasets.values()

                )

            ),

            "execution_time": str(

                self.execution_end -

                self.execution_start

            )

        }

        with open(

            self.report_dir /

            "validation_summary.json",

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                summary,

                f,

                indent=4

            )

    # =========================================================================
    # TEXT REPORT
    # =========================================================================

    def export_text_report(self):

        """
        Publication-ready report.
        """

        report = []

        report.append(

            "=" * 70

        )

        report.append(

            "EXTERNAL VALIDATION REPORT"

        )

        report.append(

            "=" * 70

        )

        report.append("")

        report.append(

            f"Generated : {datetime.now()}"

        )

        report.append(

            f"External Cohorts : {len(self.datasets)}"

        )

        report.append("")

        report.append(

            "Performance Summary"

        )

        report.append("-" * 70)

        report.append("")

        for _, row in self.performance.iterrows():

            report.append(

                f"Cohort : {row['Cohort']}"

            )

            report.append(

                f"Accuracy : {row['Accuracy']:.4f}"

            )

            report.append(

                f"ROC AUC : {row['ROC_AUC']:.4f}"

            )

            report.append(

                f"Sensitivity : {row['Sensitivity']:.4f}"

            )

            report.append(

                f"Specificity : {row['Specificity']:.4f}"

            )

            report.append("")

        with open(

            self.report_dir /

            "validation_report.txt",

            "w",

            encoding="utf-8"

        ) as f:

            f.write(

                "\n".join(report)

            )

    # =========================================================================
    # FINALIZE
    # =========================================================================

    def finalize(self):

        self.execution_end = datetime.now()

        self.logger.info("=" * 80)

        self.logger.info("ANALYSIS COMPLETED")

        self.logger.info("=" * 80)

        self.logger.info(

            "Execution Time : %s",

            self.execution_end -

            self.execution_start

        )

        self.logger.info("")

    # =========================================================================
    # RUN PIPELINE
    # =========================================================================

    def run(self):

        """
        Execute complete external validation analytics.
        """

        self.setup_logger()

        self.check_files()

        self.load_metadata()

        self.load_predictions()

        self.register_datasets()

        self.quality_control()

        self.evaluate_all_cohorts()

        self.bootstrap_confidence_intervals()

        self.overall_validation()

        self.create_all_figures()

        self.export_analysis_data()

        self.finalize()

        self.export_json_summary()

        self.export_text_report()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    engine = ExternalValidationAnalytics()

    engine.run()