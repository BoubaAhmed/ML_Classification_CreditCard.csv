import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
    log_loss, matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    cohen_kappa_score, fbeta_score
)
from config import X_test, y_test, metrics_dir_NearMiss, metrics_file_NearMiss, matrix_dir,  metrics_evaluation_dir


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "F1 Score": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Log Loss": log_loss(y_test, y_proba),
        "AUC": roc_auc_score(y_test, y_proba),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Kappa": cohen_kappa_score(y_test, y_pred),
        "Specificity": cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 1] > 0 else 0,
        "F2 Score": fbeta_score(y_test, y_pred, beta=2),
        "Confusion Matrix": cm
    }

def save_metrics(metrics_file, model_name, metrics_data):
    if not os.path.exists(metrics_dir_NearMiss):
        os.makedirs(metrics_dir_NearMiss)

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as file:
            try:
                all_metrics = json.load(file)
                if not isinstance(all_metrics, list):
                    all_metrics = []
            except json.JSONDecodeError:
                all_metrics = [] 
    else:
        all_metrics = []
    all_metrics.append({model_name: metrics_data})
    with open(metrics_file, "w") as file:
        json.dump(all_metrics, file, indent=4)

    print(f"✅ Metrics for {model_name} updated successfully in {metrics_file}")


def plot_confusion_matrix(cm, model_name, labels=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels or ["Class 0", "Class 1"],
        yticklabels=labels or ["Class 0", "Class 1"]
    )
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(matrix_dir, f"Confusion Matrix for {model_name}"))
    plt.show()

def plot_metrics(metrics_data, model_name):
    metrics_data = {k: v for k, v in metrics_data.items() if k != "Training Time (seconds)"}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics_data.keys()), y=list(metrics_data.values()), palette="viridis")
    plt.title(f"{model_name} Evaluation Metrics")
    plt.xticks(rotation=45)
    plt.ylabel("Scores")
    plt.savefig(os.path.join(metrics_evaluation_dir, f"{model_name} Evaluation Metrics"))
    plt.show()

def evaluate_model(model_name, model, training_time):
    metrics_data = calculate_metrics(model, X_test, y_test)
    metrics_data['Training Time (seconds)'] = training_time 
    cm = metrics_data.pop("Confusion Matrix")
    save_metrics(metrics_file_NearMiss, model_name, metrics_data)
    plot_confusion_matrix(cm, model_name)
    plot_metrics(metrics_data, model_name)
    print(f"Metrics saved and plots generated for {model_name}.")
    return metrics_data

