import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
    balanced_accuracy_score,
    jaccard_score,
    cohen_kappa_score,
    classification_report,
)
import seaborn as sns


def compute_binary_metrics(y_true, y_pred, y_prob):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall (Sensitivity)": recall_score(y_true, y_pred),
        "Specificity": recall_score(y_true, y_pred, pos_label=0),
        "F1 Score": f1_score(y_true, y_pred),
        "Dice Coefficient": f1_score(y_true, y_pred),
        "Jaccard Index (IoU)": jaccard_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Matthews Correlation Coefficient (MCC)": matthews_corrcoef(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "False Positive Rate (FPR)": np.sum((y_pred == 1) & (y_true == 0))
        / max(np.sum(y_true == 0), 1),
        "False Negative Rate (FNR)": np.sum((y_pred == 0) & (y_true == 1))
        / max(np.sum(y_true == 1), 1),
    }
    return metrics


def compute_multiclass_metrics(y_true, y_pred, num_classes):
    metrics = {
        "Overall Accuracy": accuracy_score(y_true, y_pred),
        "Per-class Accuracy": classification_report(y_true, y_pred, output_dict=True),
        "Mean Precision": precision_score(y_true, y_pred, average="macro"),
        "Mean Recall": recall_score(y_true, y_pred, average="macro"),
        "Mean F1 Score": f1_score(y_true, y_pred, average="macro"),
        "Mean IoU": jaccard_score(y_true, y_pred, average="macro"),
        "Weighted F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Weighted IoU": jaccard_score(y_true, y_pred, average="weighted"),
        "Cohen’s Kappa": cohen_kappa_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
        "Macro-Averaged Precision": precision_score(y_true, y_pred, average="macro"),
        "Macro-Averaged Recall": recall_score(y_true, y_pred, average="macro"),
        "Macro-Averaged F1": f1_score(y_true, y_pred, average="macro"),
        "Micro-Averaged Precision": precision_score(y_true, y_pred, average="micro"),
        "Micro-Averaged Recall": recall_score(y_true, y_pred, average="micro"),
        "Micro-Averaged F1": f1_score(y_true, y_pred, average="micro"),
        "Frequency Weighted IoU": jaccard_score(y_true, y_pred, average="weighted"),
    }
    return metrics


def plot_binary_classification_graphs(y_true, y_prob, y_pred, model_dir="."):
    os.makedirs(model_dir, exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(model_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(model_dir, "precision_recall_curve.png"))
    plt.close()

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    # Histogram of Prediction Probabilities
    plt.figure()
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Class 0")
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Class 1")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Probabilities")
    plt.legend()
    plt.savefig(os.path.join(model_dir, "prediction_histogram.png"))
    plt.close()
