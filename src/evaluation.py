import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h2o
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, auc, confusion_matrix,
    precision_score, f1_score, precision_recall_curve, average_precision_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
from src.visualization import plot_roc_auc, plot_confusion_matrix, plot_precision_recall
import os
import yaml

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Determine the save path
save_path = os.path.join(config['output']['save_path'], config['output']['folder_name'])
approach = config["approach"]
print(approach)
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists


def evaluate_model(best_model, test_data, y_test, approach=approach):
    if approach == "h2o":
        preds_df = best_model.predict(test_data).as_data_frame()
        y_pred_prob = preds_df.drop(columns=["predict"]).values
        y_pred = preds_df["predict"].values
        y_true = y_test.values
    else:  # sklearn
        y_pred_prob = best_model.predict_proba(test_data)
        y_pred = best_model.predict(test_data)
        y_true = y_test

    classification_rep = classification_report(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    #model_name = best_model
    performance_metrics = (
        f"Classification Report:\n{classification_rep}\n\n"
        f"F1 Score: {f1:.2f}\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        #f"Precision: {model_name:.2f}\n"
    )
    print(performance_metrics)
    # Visualize Metrics as Text on a Plot
    plt.figure(figsize=(10, 6))
    plt.axis('off')  # Turn off axes
    plt.text(
        0.5, 0.5, performance_metrics,
        fontsize=12, ha='center', va='center', multialignment='left',
        bbox=dict(boxstyle="round", edgecolor="black", facecolor="lightgrey")
    )
    plt.title("Model Performance Metrics", fontsize=14)

    if save_path:
        metrics_text_path = f"{save_path}/performance_metrics_text.png"
        plt.savefig(metrics_text_path, bbox_inches='tight')
        print(f"Saved performance metrics text plot to: {metrics_text_path}")
    #plt.show()
    plt.close()

    return y_true, y_pred, y_pred_prob, performance_metrics


def plot_evaluation_metrics(y_true, y_pred, y_pred_prob, save_path):
    # Original classes, e.g. [0, 1, 2, 3]
    classes = np.unique(y_true)
    
    # Shift them to 1-based, e.g. [1, 2, 3, 4]
    classes_1_based = classes + 1
    
    # Pass classes_1_based to the plotting functions for display
    plot_roc_auc(y_true, y_pred_prob, classes, save_path)
    plot_confusion_matrix(y_true, y_pred, classes_1_based, save_path)
    plot_precision_recall(y_true, y_pred_prob, classes, save_path)

