import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV

from src.data_preprocessing import preprocess_data  # This now returns (train_df, test_df)
from src.evaluation import evaluate_model, plot_evaluation_metrics
from src.visualization import plot_model_correlation_heatmap, generate_varimp
from src.explanation import (
    explain_single_instance,
    create_feature_comparison_dataframe,
)
from src.pdf import generate_pdf_with_patient_data

class AutoMLClassifier:
    def __init__(self, approach="", csv_path="", target_column="", k_features=20, test_size=0.20, 
                 random_state=42, max_models=5, max_runtime_secs=600, nfolds=3, 
                 save_path="/Users/murad/Desktop/Masters_Internship_Project/exai_fig", 
                 folder_path= "/Users/murad/Desktop/Masters_Internship_Project/"):
        self.approach = approach
        self.csv_path = csv_path
        self.target_column = target_column
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.nfolds = int(nfolds) 
        self.save_path = save_path
        self.folder_path = folder_path
        self.train = None      # For H2O training frame
        self.test = None       # For H2O test frame
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.aml = None

    def initialize_h2o(self):
        h2o.init(max_mem_size="16G")

    def preprocess_data(self):
        """
        Uses the updated preprocess_data function to obtain training and test DataFrames.
        It keeps the 'Generated_From' column in a copy of the raw data for later use
        (e.g. matching instances) while removing it from the feature matrices used for training.
        """
        # Get training and test DataFrames from the updated pipeline.
        # (Assume that the external preprocess_data() returns a tuple (train_df, test_df)
        # that contains the 'Generated_From' column.)
        raw_train_df, raw_test_df = preprocess_data(self.csv_path, self.target_column, self.k_features)
        
        # Remove any extra whitespace in column names.
        raw_train_df.columns = raw_train_df.columns.str.strip()
        raw_test_df.columns = raw_test_df.columns.str.strip()
        
        # Save the raw versions (with all columns) for later use.
        self.raw_train_df = raw_train_df.copy()
        self.raw_test_df = raw_test_df.copy()
        
        # Now, drop the "Generated_From" column from the training and test data that will be used for modeling.
        train_df = raw_train_df.copy()
        test_df = raw_test_df.copy()
        if "Generated_From" in train_df.columns:
            train_df = train_df.drop(columns=["Generated_From"])
        if "Generated_From" in test_df.columns:
            test_df = test_df.drop(columns=["Generated_From"])
        
        # Separate features and target for training data.
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]
        
        # Separate features and target for test data.
        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]
        
        # Optionally, ensure labels are zero-indexed (if needed).
        y_train = y_train - y_train.min()
        y_test = y_test - y_test.min()
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # If using h2o, convert the train and test sets to H2OFrames.
        if self.approach == "h2o":
            self.train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
            self.test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
            # Convert target column to a factor in h2o.
            self.train[self.target_column] = self.train[self.target_column].asfactor()


    def train_h2o_model(self):
        print(f"nfolds: {self.nfolds}, save_path: {self.save_path}")
        aml = H2OAutoML(
            nfolds=self.nfolds,
            balance_classes=True,
            max_models=self.max_models,
            max_runtime_secs=self.max_runtime_secs,
            seed=self.random_state,
        )
        aml.train(x=self.X_train.columns.tolist(), y=self.target_column, training_frame=self.train)
        self.best_model = aml.leader
        self.aml = aml
        if self.best_model is None:
            raise ValueError("No model was selected as the best model in H2O AutoML.")

    def stratified_cv_evaluation(self, model):
        skf = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        print(f"CV Accuracy: {mean_cv:.2f} ± {std_cv:.2f}")
        return mean_cv, std_cv
    
    def train_sklearn_model(self):

        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

        # Define models to compare.
        models = {
            "RandomForestClassifier": RandomForestClassifier(random_state=self.random_state),
            "XRF": ExtraTreesClassifier(random_state=self.random_state),
        }

        # Expanded parameter grids for better regularization and control.
        param_grids = {
            "RandomForestClassifier": {
                'n_estimators': [50, 100,150],
                'max_depth': [5, 10, None],          # Limiting tree depth can reduce overfitting.
                'min_samples_split': [2, 5, 10],       # Increasing the minimum split size reduces overfitting.
                'min_samples_leaf': [1, 2, 4]          # A higher number for min samples per leaf may improve generalization.
            },
            "XRF": {
                'n_estimators': [50, 100,150],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
        }

        best_models = {}

        # Loop over the models and perform hyperparameter search.
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[model_name],
                n_iter=50,  # Reduced n_iter; our parameter space is now larger than before, but 50 iterations should be enough.
                cv=3,
                n_jobs=-1,
                scoring='accuracy',
                random_state=self.random_state
            )
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_

            # Ensure X_test has the same feature names as X_train.
            X_test_df = pd.DataFrame(self.X_test, columns=self.X_train.columns)
            y_pred = best_model.predict(X_test_df)

            print(f"{model_name} Best Parameters: {random_search.best_params_}")
            print(f"{model_name} Classification Report:\n", classification_report(self.y_test, y_pred))
            test_accuracy = accuracy_score(self.y_test, y_pred)
            print(f"{model_name} Accuracy Score: {test_accuracy:.2f}")

            print("Evaluating with Stratified Cross-Validation...")
            train_accuracy = accuracy_score(self.y_train, best_model.predict(self.X_train))
            cv_mean, cv_std = self.stratified_cv_evaluation(best_model)
            print(f"Training Accuracy: {train_accuracy:.2f}")
            print(f"Testing Accuracy: {test_accuracy:.2f}")
            if train_accuracy - test_accuracy > 0.1:
                print("The model might be overfitting.")

            # Store metrics and the best model.
            best_models[model_name] = {
                "best_model": best_model,
                "train_accuracy": train_accuracy,
                "accuracy": test_accuracy,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
            }

        # Select the best model based solely on cross-validation mean accuracy.
        best_model_name = max(best_models, key=lambda name: best_models[name]['cv_mean'])
        self.best_model = best_models[best_model_name]["best_model"]
        print(f"Selected Best Model: {best_model_name}")

        # Save the best model.
        joblib.dump(self.best_model, f"{self.save_path}/best_model.pkl")
        print(f"Saved the best model to {self.save_path}/best_model.pkl")
        print(f"Best Model: {best_model_name} with CV Accuracy: {best_models[best_model_name]['cv_mean']:.2f}")

    def train_model(self):
        if self.approach == "h2o":
            self.train_h2o_model()
        elif self.approach == "sklearn":
            self.train_sklearn_model()

    def full_pipeline(self):
        if self.approach == "h2o":
            self.initialize_h2o()

        self.preprocess_data()
        self.train_model()

        # Evaluate and visualize the model's performance
        if self.approach == "h2o":
            y_true, y_pred, y_pred_prob, performance_metrics = evaluate_model(self.best_model, self.test, self.y_test)
        else:
            y_true, y_pred, y_pred_prob, performance_metrics = evaluate_model(self.best_model, self.X_test, self.y_test)

        plot_evaluation_metrics(y_true, y_pred, y_pred_prob, self.save_path)

        if self.approach == "h2o":
            plot_model_correlation_heatmap(self.aml, self.test, self.save_path)

        generate_varimp(self.best_model, self.save_path)

        # Explain and analyze features
        create_feature_comparison_dataframe(
            self.best_model, self.X_train, self.y_train, self.X_test, self.y_test, instance_index=0
        )
        explain_single_instance(self.best_model, self.X_train, self.X_test, self.y_test, self.save_path)

        # Generate PDF report
        save_path = self.folder_path
        exai_fig_path = os.path.join(save_path, "exai_fig")
        report_path = os.path.join(save_path, "report")
        print("Generating PDF report...")
        print(exai_fig_path)
        print(report_path)
        summary_file_path = os.path.join(exai_fig_path, "summary.txt")
        lime_anchor_path = os.path.join(exai_fig_path, "lime_anchor_comparison.png")
        confusion_matrix_path = os.path.join(exai_fig_path, "confusion_matrix.png")
        precision_recall_path = os.path.join(exai_fig_path, "precision_recall_curve.png")
        roc_auc_path = os.path.join(exai_fig_path, "roc_auc_curve.png")
        performance_metrics_text_path = os.path.join(exai_fig_path, "performance_metrics_text.png")
        output_pdf_path = os.path.join(report_path, "exai_report.pdf")
        print(summary_file_path)
        print(output_pdf_path)

        if os.path.exists(summary_file_path):
            generate_pdf_with_patient_data(
                lime_anchor_path=lime_anchor_path,
                confusion_matrix_path=confusion_matrix_path,
                precision_recall_path= precision_recall_path,
                roc_auc_path= roc_auc_path,
                performance_metrics_text_path=performance_metrics_text_path,
                summary_file_path=summary_file_path,
                output_pdf_path=output_pdf_path,
                csv_path=self.csv_path,
                target_column=self.target_column
            )
            print(f"PDF report generated at: {output_pdf_path}")
        else:
            print("Summary file not found. Skipping PDF generation.")
