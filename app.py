import streamlit as st
import os
import pandas as pd
import shutil
import yaml
import subprocess
import sys
import joblib
import h2o

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.evaluation import evaluate_model, plot_evaluation_metrics
from src.visualization import plot_model_correlation_heatmap, generate_varimp
from src.explanation import explain_single_instance
from src.pdf import generate_pdf_with_patient_data
from ruamel.yaml import YAML

def update_config_file(config_path, updates):
    try:
        yaml_obj = YAML()
        yaml_obj.preserve_quotes = True
        with open(config_path, "r") as file:
            config = yaml_obj.load(file)

        def recursive_update(orig, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in orig:
                    recursive_update(orig[key], value)
                else:
                    orig[key] = value
        recursive_update(config, updates)

        with open(config_path, "w") as file:
            yaml_obj.dump(config, file)

        st.sidebar.success(f"Configuration updated and saved at {config_path}")
    except Exception as e:
        st.sidebar.error(f"Error updating the configuration file: {e}")

def load_best_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    return joblib.load(model_path)

# ------------------- Streamlit Page Layout -------------------
st.title("Automated Machine Learning Pipeline with PDF Output")

st.sidebar.header("Pipeline Mode")
pipeline_mode = st.sidebar.radio("Select Mode", ["Run Whole Pipeline", "Single Instance Prediction"])

# ------------------- Load Config ----------------------------
config_path = "/Users/murad/Desktop/Masters_Internship_Project/config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/best_model.pkl"
dataset_path = config['dataset']['csv_path']

# ------------------- Config Editing in Sidebar --------------
st.sidebar.header("Configuration Settings")
with st.sidebar.expander("Update Configuration"):
    approach = st.selectbox("Select Approach", ["h2o", "sklearn"],
                           index=["h2o", "sklearn"].index(config.get("approach", "sklearn")))
    test_size = st.slider("Test Size", 0.1, 0.5,
                          config.get("dataset", {}).get("test_size", 0.2), step=0.05)
    target_column = st.text_input("Target Column",
                                  value=config.get("dataset", {}).get("target_column", "Stage"))
    k_features = st.slider("Number of Features to Select (k)", 1, 50,
                           config.get("dataset", {}).get("k_features", 20))

    if st.button("Save Configuration"):
        updates = {
            "approach": approach,
            "dataset": {
                "test_size": test_size,
                "target_column": target_column,
                "k_features": k_features
            }
        }
        update_config_file(config_path, updates)

# ------------------- Whole Pipeline Runner -------------------
def run_whole_pipeline():
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success("File uploaded successfully!")
        data_df = pd.read_csv(file_path)
        st.subheader("Uploaded Dataset")
        st.write(data_df.head())
        
        if st.sidebar.button("Run Pipeline"):
            st.subheader("Running Pipeline...")
            try:
                result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Pipeline completed successfully!")
                else:
                    st.error("Pipeline encountered errors:\n" + result.stderr)
            except Exception as e:
                st.error(f"Error running pipeline: {e}")
            
            report_path = "/Users/murad/Desktop/Masters_Internship_Project/report/exai_report.pdf"
            if os.path.exists(report_path):
                with open(report_path, "rb") as pdf_file:
                    st.download_button("Download PDF Report", pdf_file,
                                       file_name="exai_report.pdf",
                                       mime="application/pdf")
            else:
                st.error("PDF report not found.")

# ------------------- Single Instance Runner -------------------
def run_single_instance():
    st.sidebar.header("Upload Single Instance CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your single instance file (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        data_df = pd.read_csv(file_path)
        st.subheader("Uploaded Single Instance")
        st.write(data_df)
        
        # Must be exactly one row
        if len(data_df) != 1:
            st.error("Uploaded file must contain exactly one instance (one row).")
            return
        
        if st.sidebar.button("Run Single Instance Prediction"):
            st.subheader("Running EXAI Analysis...")
            
            # 1. Load your pre-trained model
            best_model = load_best_model(model_path)

            # (NEW) 2. Load the fitted preprocessing pipeline
            pipeline_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/preprocessing_pipeline.pkl"
            if not os.path.exists(pipeline_path):
                st.error("Preprocessing pipeline not found. Make sure it was saved during training.")
                return
            
            preprocessing_pipeline = joblib.load(pipeline_path)

            # 3. Remove target column if it exists
            tgt_col = config["dataset"]["target_column"]
            if tgt_col in data_df.columns:
                data_df.drop(columns=[tgt_col], inplace=True)

            # 4. Transform the single instance with the saved pipeline
            try:
                single_transformed = preprocessing_pipeline.transform(data_df)
            except Exception as e:
                st.error(f"Error applying loaded pipeline: {e}")
                return

            # 5. If h2o approach, init h2o; else skip
            if config["approach"] == "h2o":
                h2o.init()
                # The pipeline output is a numpy array. We need a DataFrame with correct column names
                # In the pipeline, some columns might be dropped or one-hot. For demonstration:
                col_count = single_transformed.shape[1]
                col_names = [f"col_{i}" for i in range(col_count)]
                instance_df = pd.DataFrame(single_transformed, columns=col_names)
                instance_h2o = h2o.H2OFrame(instance_df)
                predictions = best_model.predict(instance_h2o).as_data_frame()
                predicted_class = predictions["predict"][0]
            else:
                # scikit-learn approach
                predicted_class = best_model.predict(single_transformed)[0]

            st.write(f"Predicted Class: {predicted_class}")

            # Evaluate & Explain
            # (For demonstration, let's pass data_df (untransformed) to keep a minimal code change.
            #  If you want consistent SHAP, you might pass single_transformed or reconstruct a DataFrame.)
            evaluate_model(best_model, data_df, [predicted_class], approach=config["approach"])
            plot_evaluation_metrics([predicted_class], [predicted_class], [0.5], config["output"]["save_path"])
            explain_single_instance(
                best_model,
                data_df,
                data_df,
                [predicted_class],
                config["output"]["save_path"],
                config["approach"]
            )

            # Generate PDF
            generate_pdf_with_patient_data(
                lime_anchor_path=os.path.join(config["output"]["save_path"], "lime_anchor_comparison.png"),
                confusion_matrix_path=os.path.join(config["output"]["save_path"], "confusion_matrix.png"),
                precision_recall_path=os.path.join(config["output"]["save_path"], "precision_recall_curve.png"),
                roc_auc_path=os.path.join(config["output"]["save_path"], "roc_auc_curve.png"),
                performance_metrics_text_path=os.path.join(config["output"]["save_path"], "performance_metrics_text.png"),
                summary_file_path=os.path.join(config["output"]["save_path"], "summary.txt"),
                output_pdf_path="/Users/murad/Desktop/Masters_Internship_Project/report/exai_single_instance.pdf",
                csv_path=dataset_path,
                target_column=config["dataset"]["target_column"]
            )

            report_path = "/Users/murad/Desktop/Masters_Internship_Project/report/exai_single_instance.pdf"
            if os.path.exists(report_path):
                with open(report_path, "rb") as pdf_file:
                    st.download_button(
                        "Download Single-Instance PDF Report",
                        pdf_file,
                        file_name="exai_single_instance.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("PDF report not found.")

# ------------------- Main Streamlit Flow ---------------------
if pipeline_mode == "Run Whole Pipeline":
    run_whole_pipeline()
else:
    run_single_instance()
