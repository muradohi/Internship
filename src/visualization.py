import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import h2o
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, 
    f1_score, precision_recall_curve, average_precision_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import os
import yaml



def generate_varimp(best_model, save_path):
    # Check if the model supports variable importance
    if hasattr(best_model, 'varimp') and callable(best_model.varimp):
        # Get variable importance as a DataFrame
        varimp_df = pd.DataFrame(best_model.varimp(), columns=['variable', 'relative_importance', 'scaled_importance', 'percentage'])

        # Ensure 'percentage' column is numeric
        varimp_df['percentage'] = pd.to_numeric(varimp_df['percentage'], errors='coerce')
        varimp_df = varimp_df.dropna(subset=['percentage'])

        if varimp_df.empty:
            print("Variable importance data is empty after processing.")
            return

        # Select the top 10 important features
        top_10_features = varimp_df.nlargest(10, 'percentage')

        # Plot the top 10 features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='percentage', y='variable', data=top_10_features, palette='viridis')
        plt.title("Top 10 Important Variables")
        plt.xlabel("Importance (%)")
        plt.ylabel("Features")
        plt.tight_layout()

        file_path = os.path.join(save_path, "variable_importance.png")
        plt.savefig(file_path)
        print(f"Saved variable importance plot to: {file_path}")
        #plt.show()
        plt.close()
    else:
        print("The model does not support variable importance.")


def generate_model_correlation_heatmap(aml_leaderboard, test_data):
    # Ensure test_data is an H2OFrame
    if not isinstance(test_data, h2o.H2OFrame):
        test_data = h2o.H2OFrame(test_data)

    # Get the list of models in the AutoML leaderboard
    model_list = [h2o.get_model(model_id) for model_id in aml_leaderboard['model_id'].as_data_frame().model_id]
    
    print("Generating Model Correlation Heatmap...")
    h2o.model_correlation_heatmap(model_list, test_data)
    




def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                     xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Force the tick marks to be at the center of each cell and show only the integer class labels.
    ax.set_xticks(np.arange(len(classes)) + 1.0)
    ax.set_yticks(np.arange(len(classes)) + 1.0)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    plt.tight_layout()
    
    file_path = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(file_path)
    #plt.show()
    plt.close()

def plot_actual_vs_predicted(actual_class, predicted_class, save_path):
    plt.figure(figsize=(6, 6))
    plt.bar(["Actual", "Predicted"], [actual_class, predicted_class], color=["blue", "orange"])
    plt.title("Actual vs Predicted Class")
    
    # Set the y-axis limits and ticks to show integer stages only.
    max_val = max(actual_class, predicted_class)
    plt.ylim([1, max_val + 1])
    plt.yticks(range(1, max_val + 1))
    
    file_path = os.path.join(save_path, "accVSpred.png")
    plt.savefig(file_path)
    #plt.show()
    plt.close()


def plot_precision_recall(y_true, y_pred_prob, classes, save_path):
    # Convert classes to 1-based for display if they're numeric
    classes_display = [cls + 1 for cls in classes]

    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]
    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        average_precision = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])
        plt.plot(
            recall, precision, lw=2,
            label=f'Class {classes_display[i]} (AP = {average_precision:.2f})'
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    file_path = os.path.join(save_path, "precision_recall_curve.png")
    plt.savefig(file_path)
    plt.close()

def plot_roc_auc(y_true, y_pred_prob, classes, save_path):
    if isinstance(y_true, h2o.H2OFrame):
        y_true = y_true.as_data_frame().squeeze()

    y_true = np.array(y_true)

    # If classes is None, infer from y_true; otherwise use the provided classes
    classes = np.unique(y_true) if classes is None else classes

    # Convert classes to 1-based for display if they're numeric
    classes_display = [cls + 1 for cls in classes]

    print(f"y_true: {y_true[:5]}")
    print(f"classes (0-based): {classes}")
    print(f"classes (1-based, for display): {classes_display}")
    print(f"y_true shape: {y_true.shape}")

    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]

    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, lw=2,
            label=f"Class {classes_display[i]} (AUC = {roc_auc:.2f})"
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    file_path = os.path.join(save_path, "roc_auc_curve.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved ROC AUC plot to: {file_path}")




def plot_model_correlation_heatmap(aml, test_data, save_path):
    # Ensure test_data is an H2OFrame
    if not isinstance(test_data, h2o.H2OFrame):
        test_data = h2o.H2OFrame(test_data)

    # Get the leaderboard as a DataFrame
    leaderboard = aml.leaderboard.as_data_frame()
    models = leaderboard['model_id'].tolist()

    # Get predictions for each model on the test set
    model_predictions = []
    for model_id in models:
        model = h2o.get_model(model_id)
        preds = model.predict(test_data).as_data_frame()["predict"].astype(float)
        model_predictions.append(preds)

    # Concatenate predictions to create a DataFrame
    preds_df = pd.DataFrame(model_predictions).T
    preds_df.columns = models

    # Compute the correlation matrix
    correlation_matrix = preds_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Model Correlation Heatmap")
    plt.xlabel("Model")
    plt.ylabel("Model")
    plt.tight_layout()
    file_path = os.path.join(save_path, "modelCorr.png")
    plt.savefig(file_path)
    #plt.show()
    plt.close()
