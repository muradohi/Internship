import os
import random
import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE

# Import your synthetic pipeline
from src.dataGen import SyntheticDataPipeline

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

csv_path = config["dataset"]["csv_path"]
target_column = config["dataset"]["target_column"]
k_features = config["dataset"]["k_features"]
datagen_mode = config["datagen"]

def preprocess_data(csv_path, target_column, k_features=k_features,
                    test_size=0.1, random_state=42):
    """
    Depending on config['datagen']:
      - If 'synthetic', follows the synthetic data generation process:
          * Splits the data into train/test.
          * Runs SyntheticDataPipeline on the training data & subsets of test data.
          * Appends synthetic test samples to the balanced training data.
          * Applies a preprocessing pipeline (including one-hot encoding).
          * Saves the fitted pipeline.
          * Returns (train_df, test_df).
      - Otherwise, uses an alternative process (SMOTE, feature selection, t-SNE).
        * Also saves the fitted pipeline.
        * Returns (train_selected, test_selected).
    """

    if datagen_mode == "synthetic":
        # 1. Load dataset & drop missing target
        full_dataset = pd.read_csv(csv_path)
        full_dataset.dropna(subset=[target_column], inplace=True)

        # 2. Train/test split
        train_data, test_data = train_test_split(
            full_dataset,
            test_size=test_size,
            random_state=random_state,
            stratify=full_dataset[target_column],
        )

        # 3. Synthetic pipeline on training data
        temp_train_path = "/Users/murad/Desktop/Masters_Internship_Project/temp_train.csv"
        train_data.to_csv(temp_train_path, index=False)

        pipeline_synth = SyntheticDataPipeline(
            config_path="/Users/murad/Desktop/Masters_Internship_Project/config/config.yaml",
            file_path=temp_train_path,
            save_dir=os.path.dirname(temp_train_path),
            max_count=1000,
        )
        pipeline_synth.run_pipeline()
        balanced_train_data = pd.read_csv(pipeline_synth.get_save_path())

        # 5. Test data mark Source & Generated_From
        test_data["Source"] = "Original"
        test_data["Generated_From"] = np.nan

        # 6. Subset of test data for augmentation
        test_to_augment_list = []
        for grp, df_grp in test_data.groupby(target_column):
            n = len(df_grp)
            n_aug = int(0.8 * n)
            df_aug = df_grp.sample(n=n_aug, random_state=random_state)
            test_to_augment_list.append(df_aug)
        test_to_augment = pd.concat(test_to_augment_list)

        temp_test_path = "/Users/murad/Desktop/Masters_Internship_Project/temp_test.csv"
        test_to_augment.to_csv(temp_test_path, index=False)

        pipeline_test = SyntheticDataPipeline(
            config_path="/Users/murad/Desktop/Masters_Internship_Project/config/config.yaml",
            file_path=temp_test_path,
            save_dir=os.path.dirname(temp_test_path),
            max_count=300,
        )
        pipeline_test.run_pipeline()
        synthetic_test_data = pd.read_csv(pipeline_test.get_save_path())

        # 7. Append synthetic test data to balanced training data
        balanced_train_data = pd.concat([balanced_train_data, synthetic_test_data],
                                        ignore_index=True)

        # 7b. Optional: remove the temporary CSVs
        # os.remove(temp_train_path)
        # os.remove(temp_test_path)

        # 8. If the target is still object, label encode

        if balanced_train_data[target_column].dtype == 'object':
            # LabelEncoder automatically assigns labels starting at 0.
            le = LabelEncoder()
            balanced_train_data[target_column] = le.fit_transform(balanced_train_data[target_column])
            test_data[target_column] = le.transform(test_data[target_column])
        else:
            # For numeric labels that don't start at 0, subtract the minimum value.
            min_val = balanced_train_data[target_column].min()
            balanced_train_data[target_column] = balanced_train_data[target_column] - min_val
            test_data[target_column] = test_data[target_column] - min_val


        # 9. Separate features & target
        train_target = balanced_train_data[target_column]
        test_target = test_data[target_column]
        train_features = balanced_train_data.drop(columns=[target_column])
        test_features = test_data.drop(columns=[target_column])

        # 9b. if 'is_synthetic' columns exist, ensure they're categorical
        if 'is_synthetic' in train_features.columns:
            train_features['is_synthetic'] = train_features['is_synthetic'].astype('category')
        if 'is_synthetic' in test_features.columns:
            test_features['is_synthetic'] = test_features['is_synthetic'].astype('category')

        # 10. numeric / categorical columns
        numeric_cols = train_features.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()

        # Create numeric & cat transformers
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ])

        # (Optional) If you want feature selection in the pipeline
        # from sklearn.feature_selection import SelectKBest, mutual_info_classif
        # pipeline = Pipeline([
        #     ("preprocessor", preprocessor),
        #     ("selector", SelectKBest(mutual_info_classif, k=k_features))
        # ])
        # For demonstration, keep it simple:
        pipeline = preprocessor

        # Fit on train_features
        train_transformed = pipeline.fit_transform(train_features)
        test_transformed = pipeline.transform(test_features)

        # Get final feature names (for reference)
        cat_feature_names = pipeline.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols)
        feature_names = numeric_cols + list(cat_feature_names)

        # (NEW) Save the pipeline
        save_path_ppl = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/preprocessing_pipeline.pkl"
        joblib.dump(pipeline, save_path_ppl)
        print(f"Saved preprocessing pipeline to {save_path_ppl}")

        # Build final DataFrames
        train_df = pd.DataFrame(train_transformed, columns=feature_names)
        train_df[target_column] = train_target.reset_index(drop=True)

        test_df = pd.DataFrame(test_transformed, columns=feature_names)
        test_df[target_column] = test_target.reset_index(drop=True)

        return train_df, test_df

    else:
        # ---------------- Alternative Process using SMOTE, Feature Selection, t-SNE ---------------------
        # Load dataset and drop rows missing the target
        dataset = pd.read_csv(csv_path)
        dataset.dropna(subset=[target_column], inplace=True)

        # If target is object, label-encode it
        if dataset[target_column].dtype == 'object':
            le = LabelEncoder()
            dataset[target_column] = le.fit_transform(dataset[target_column])

        # -------------------- Split Full Dataset into train_data and test_data --------------------
        train_data, test_data = train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
            stratify=dataset[target_column]
        )
        num_extra_samples = 17
        extra_samples = test_data.tail(num_extra_samples)
        train_data = pd.concat([train_data, extra_samples], ignore_index=True)

        # --- For internal processing, extract features and target from train_data and test_data ---
        # (This is done internally, so you only need to work with train_data and test_data.)
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        # -------------------- Identify numeric and categorical columns (from training data) --------------------
        numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        # -------------------- Define the Preprocessing Pipelines --------------------
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ])

        # -------------------- Fit Preprocessor and Transform Both Train and Test Sets --------------------
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Get feature names from the preprocessor (combining numeric and one-hot encoded columns)
        if categorical_cols:
            cat_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols)
        else:
            cat_feature_names = []
        feature_names = numeric_cols + list(cat_feature_names)

        # -------------------- Apply SMOTE on Training Data Only --------------------
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

        # -------------------- Feature Selection --------------------
        selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]

        # Apply the same feature selection transformation to the test set
        X_test_selected = selector.transform(X_test_transformed)

        # -------------------- Rebuild Final DataFrames --------------------
        # For training data: SMOTE has been applied.
        final_train_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
        final_train_df[target_column] = y_train_resampled.reset_index(drop=True)

        # For test data: Only transformation and feature selection have been applied.
        final_test_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)
        final_test_df[target_column] = y_test.reset_index(drop=True)

        # Save the preprocessor for future use
        save_path_ppl = "/Users/murad/Desktop/Masters_Internship_Project/exai_fig/preprocessing_pipeline.pkl"
        joblib.dump(preprocessor, save_path_ppl)
        print(f"Saved partial preprocessing pipeline to {save_path_ppl}")

        # Optionally, save the final DataFrames to CSV files
        smote_train_path = "/Users/murad/Desktop/Masters_Internship_Project/smote_train.csv"
        smote_test_path = "/Users/murad/Desktop/Masters_Internship_Project/smote_test.csv"
        final_train_df.to_csv(smote_train_path, index=False)
        final_test_df.to_csv(smote_test_path, index=False)

        # Return the final training and test DataFrames
        train_selected, test_selected = final_train_df, final_test_df

        return train_selected, test_selected


# # Example Usage:
# train_df, test_df = preprocess_data(csv_path, target_column, k_features=k_features)
# print("Transformed Training DataFrame (first 5 rows):")
# print(train_df.head())
# if not test_df.empty:
#     print("\nTransformed Test DataFrame (first 5 rows):")
#     print(test_df.head())
# else:
#     print("\nNo separate test DataFrame generated in alternative process.")
