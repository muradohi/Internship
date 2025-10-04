from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from huggingface_hub import InferenceClient
from reportlab.lib.units import inch
import pandas as pd
import numpy as np
import json
import os
import yaml

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Paths for the images and files
lime_anchor_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/lime_anchor_comparison.png"
confusion_matrix_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/confusion_matrix.png"
precision_recall_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/precision_recall_curve.png"
performance_metrics_text_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/performance_metrics_text.png"
roc_auc_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/roc_auc_curve.png"
summary_file_path = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/summary.txt"
output_path = "/Users/murad/Desktop/Masters_Internship_Project/report"
output_pdf_path = "/Users/murad/Desktop/Masters_Internship_Project/report/exai_report.pdf"
csv_path = config['dataset']['csv_path']
target_column = config['dataset']['target_column']



model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'

llm_client = InferenceClient(

    model=model_name,
    token = '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',
    timeout=120,

)


def call_llm(inference_client: InferenceClient, prompt: str):

    response = inference_client.post(

        json={

            "inputs": prompt,

            "parameters": {"max_new_tokens": 200},

            "task": "text-generation",

        }
    )

    return json.loads(response.decode())[0]["generated_text"]

def parse_summary_file(summary_file_path):
    """Parse the summary file to extract patient details and the summary."""
    patient_details = {}
    summary_text = ""
    current_section = None

    

    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("Patient Details"):
                    current_section = "patient_details"
                    continue
                elif line.__contains__("Prediction Summary:"):
                    current_section = "summary_text"
                    continue

                if current_section == "patient_details" and ":" in line:
                    key, value = line.split(":", 1)
                    patient_details[key.strip()] = float(value.strip()) if value.strip().replace('.', '', 1).isdigit() else value.strip()
                elif current_section == "summary_text":
                    summary_text += line + "\n"

    return patient_details, summary_text

def reverse_preprocessing(processed_instance, csv_path, target_column):
    original_data = pd.read_csv(csv_path, na_values=["NA"])
    numeric_cols = original_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    preprocessor.fit(original_data)

    # Reverse processing
    numeric_data = np.array([[processed_instance.get(col, np.nan) for col in numeric_cols]])
    original_numeric = preprocessor.named_transformers_['num']['scaler'].inverse_transform(numeric_data)

    # Reverse one-hot encoding
    onehot = preprocessor.named_transformers_['cat']['onehot']
    original_categorical = None
    if categorical_cols:
        categorical_data = [processed_instance.get(col, 0) for col in onehot.get_feature_names_out(categorical_cols)]
        original_categorical = onehot.inverse_transform([categorical_data])

    # Combine data
    reversed_instance = pd.DataFrame(original_numeric, columns=numeric_cols)
    if categorical_cols:
        for i, col in enumerate(categorical_cols):
            reversed_instance[col] = original_categorical[:, i] if original_categorical is not None else np.nan

    reversed_instance[target_column] = processed_instance.get(target_column, np.nan)

    # Fill NaNs with placeholders
    reversed_instance.fillna("Unknown", inplace=True)
    return reversed_instance


def format_patient_data_as_table(reversed_instance):
    """Format the reversed patient data into a multi-column table."""
    data = [[key, value] for key, value in reversed_instance.iloc[0].items()]
    columns = 2
    table_data = []
    for i in range(0, len(data), columns):
        row = data[i:i+columns]
        while len(row) < columns:
            row.append(["", ""])  # Fill missing cells
        table_data.append([item for pair in row for item in pair])

    header_row = ["Key", "Value"] * columns
    table_data.insert(0, header_row)

    table = Table(table_data, colWidths=[120] * (columns * 2))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    return table





def generate_pdf_with_patient_data(lime_anchor_path, confusion_matrix_path,precision_recall_path,
                                   roc_auc_path,performance_metrics_text_path, summary_file_path, 
                                   output_pdf_path, csv_path, target_column):
    # Parse the summary file to get processed instance and summary text
    processed_instance, summary_text = parse_summary_file(summary_file_path)
    reversed_instance = reverse_preprocessing(processed_instance, csv_path, target_column)
    reversed_instance_dict = reversed_instance.iloc[0].to_dict()
    print(reversed_instance_dict)
    print(reversed_instance)

    # Generate explanation using LLM
    llm_prompt = f"""
    You are a medical assistant. Analyze the following patient details and provide a structured explanation in plain English. Explain the patient ID, age, biochemical markers, clinical features, and treatment status in a clear, professional manner.

    Patient details:
    {json.dumps(reversed_instance_dict, indent=4)}

    Provide the response in this format:
    1. **Patient Identification**: ...
    2. **Demographics**: the age is either in days or years ...
    3. **Biochemical Markers**: ...
    4. **Clinical Features**: ...
    5. **Treatment Status**: ...
    """
    llm_response = call_llm(llm_client, llm_prompt)

    # Remove unnecessary preamble from LLM response
    formatted_response = llm_response.split("**Treatment Status**: ...")[-1].strip()

    # Create the document
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # First Page: Add Plots
    elements.append(Paragraph("Explainable AI–Based Evaluation Report", styles["Title"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Evaluation & ExAI Plots", styles["Heading2"]))

    # Define the dimensions for the images
    image_width, image_height = 2.5 * inch, 2.5 * inch  # Adjust for larger plots

    # Create a table for the images
    image_table_data = []
    if os.path.exists(lime_anchor_path) and os.path.exists(confusion_matrix_path):
        image_table_data.append([
            Image(lime_anchor_path, width=image_width, height=image_height),
            Image(confusion_matrix_path, width=image_width, height=image_height),
        ])

    if os.path.exists(precision_recall_path) and os.path.exists(roc_auc_path):
        image_table_data.append([
            Image(precision_recall_path, width=image_width, height=image_height),
            Image(roc_auc_path, width=image_width, height=image_height),
        ])

    if os.path.exists(performance_metrics_text_path):
        image_table_data.append([
            Image(performance_metrics_text_path, width=image_width, height=image_height),
        ])

    # Dynamically set row heights to match the number of rows in image_table_data
    row_heights = [image_height] * len(image_table_data)

    # Create a table from the image data
    image_table = Table(image_table_data, colWidths=[image_width, image_width], rowHeights=row_heights)
    image_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    ]))

    # Add the table to the document
    elements.append(image_table)
    #elements.append(Spacer(1, 20))

    elements.append(PageBreak())

    # Second Page: Add Reversed Patient Data
    elements.append(Paragraph("Original Patient Data", styles["Title"]))
    elements.append(Spacer(1, 20))
    patient_table = format_patient_data_as_table(reversed_instance)
    elements.append(patient_table)
    elements.append(Spacer(1, 20))

    # Add LLM Explanation
    elements.append(Paragraph("Patient Data Summary", styles["Title"]))
    elements.append(Spacer(1, 20))
    for line in formatted_response.split("\n"):
        elements.append(Paragraph(line.strip(), styles["BodyText"]))
        elements.append(Spacer(1, 10))

    # Third Page: Add Detailed Summary
    elements.append(Paragraph("LLM Detailed Summary", styles["Title"]))
    elements.append(Spacer(1, 20))
    for paragraph in summary_text.split("\n\n"):
        elements.append(Paragraph(paragraph.strip(), styles["BodyText"]))
        elements.append(Spacer(1, 15))

    elements.append(PageBreak())

    # Build the PDF
    doc.build(elements)

