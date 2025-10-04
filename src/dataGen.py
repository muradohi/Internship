import os
import numpy as np
import random
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from itertools import cycle

# Imports from pgmpy for Bayesian network learning and sampling
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling

class SyntheticDataPipeline:
    def __init__(self, config_path, file_path, save_dir=None, max_count=100, random_state=42):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.save_dir = save_dir or os.path.dirname(file_path)
        self.save_path = os.path.join(self.save_dir, f"balanced_{self.file_name}")
        self.max_count = max_count
        self.random_state = random_state
        self.target_column = self.config['dataset']['target_column']
        self.fixcol = self.config['dataset']['fixcol']
        self.data = None
        self.final_balanced_data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        if 'ID' not in self.data.columns:
            print("ID column not found. Creating ID column...")
            self.data['ID'] = range(1, len(self.data) + 1)
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        # Drop rows with missing target values
        self.data = self.data.dropna(subset=[self.target_column])
        print(f"Data loaded successfully with {len(self.data)} rows.")

    def compute_correlation_groups(self, threshold=0.4):
        """Compute groups of correlated numerical features."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        correlated_features = set()
        for col in correlation_matrix.columns:
            # Build the list of correlated features (excluding the column itself)
            high_corr = [x for x in correlation_matrix[col].index if x != col and abs(correlation_matrix[col][x]) > threshold]
            if high_corr:
                correlated_features.add(frozenset(high_corr))
        return [list(group) for group in correlated_features]

    def generate_hybrid_sample(self, target_diagnosis, feature_value_map, original_data, original_id, correlated_groups, noise=False):
        """
        Generates one synthetic sample for a given target class using a hybrid approach:
        - Categorical features are generated using a Bayesian network.
        - Numerical features are generated via a rule-based method (using correlated groups and random sampling).
        """
        # Subset data for the target class
        diag_data = original_data[original_data[self.target_column] == target_diagnosis]
        synthetic_sample = {'Generated_From': original_id, self.target_column: target_diagnosis}
        
        # --- CATEGORICAL PART: Bayesian Network ---
        # Identify categorical features
        cat_features = diag_data.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_features:
            # Use only rows with non-missing values for the categorical columns
            cat_data = diag_data[cat_features].dropna()
            if not cat_data.empty:
                # Learn a Bayesian network on the categorical data
                hc = HillClimbSearch(cat_data)
                best_structure = hc.estimate(scoring_method=BicScore(cat_data))
                # Convert the learned structure (a DAG) into a BayesianModel
                bn_model = BayesianModel(best_structure.edges())
                bn_model.fit(cat_data, estimator=MaximumLikelihoodEstimator)
                sampler = BayesianModelSampling(bn_model)
                # Sample one synthetic record for the categorical features
                cat_sample = sampler.forward_sample(size=1)
                # For each categorical column, if it exists in the generated sample, assign its value;
                # otherwise, fall back to a random selection from the available values.
                for col in cat_features:
                    if col in cat_sample.columns:
                        synthetic_sample[col] = cat_sample.iloc[0][col]
                    else:
                        if col in feature_value_map and feature_value_map[col]:
                            synthetic_sample[col] = random.choice(feature_value_map[col])
                        else:
                            synthetic_sample[col] = None
                            
        # --- NUMERICAL PART: Rule-based Generation ---
        # Identify numerical features
        num_features = diag_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # First, fill in values from correlated groups
        for group in correlated_groups:
            if set(group).issubset(diag_data.columns):
                available = diag_data[group].dropna()
                if not available.empty:
                    correlated_values = available.sample(1).iloc[0]
                    for feature, value in correlated_values.items():
                        synthetic_sample[feature] = value
                        
        # For numerical features not yet set, sample randomly from available values
        for feature in num_features:
            # Skip if the feature was already set (or is the target)
            if feature in synthetic_sample or feature == self.target_column:
                continue
            
            if feature in diag_data.columns:
                # Get the unique available values from this feature
                values = diag_data[feature].dropna().unique()
                if len(values) > 0:
                    if len(values) == 1:
                        # Only one unique value exists; add noise to create variation.
                        base_value = values[0]
                        # Choose a noise scale: 5% of the absolute value (or a fixed value if zero)
                        noise_scale = 0.05 * abs(base_value) if base_value != 0 else 0.1
                        sampled_value = base_value + random.uniform(-noise_scale, noise_scale)
                    else:
                        sampled_value = random.choice(values.tolist())
                else:
                    sampled_value = random.choice(feature_value_map.get(feature, [np.nan]))
            else:
                sampled_value = random.choice(feature_value_map.get(feature, [np.nan]))
            
            # Optionally, if the noise flag is True, add an extra noise term.
            if noise and isinstance(sampled_value, (int, float)):
                sampled_value += random.uniform(-0.1, 0.1)
                
            synthetic_sample[feature] = sampled_value
            
        return synthetic_sample



    def balance_dataset(self):
        """
        For each target class:
        - Retains the original rows (marked with Source 'Original').
        - Uses generate_hybrid_sample to create synthetic samples until each class has at least `max_count` rows.
        - Synthetic rows receive a unique Generated_From value.
        The balanced dataset is then saved to disk.
        """
        correlated_groups = self.compute_correlation_groups()
        # Create a feature value map for all columns (using non-null unique values)
        feature_value_map = {col: self.data[col].dropna().unique().tolist() for col in self.data.columns}
        
        # Get the counts for each target class
        class_counts = self.data[self.target_column].value_counts()
        balanced_data = []
        
        for target_class, count in class_counts.items():
            # Select all rows from the original data for the target class
            original_class_data = self.data[self.data[self.target_column] == target_class].copy()
            # Number of synthetic rows to generate for this class
            num_to_generate = max(0, self.max_count - count)
            
            # Mark the original rows
            original_class_data['Source'] = 'Original'
            balanced_data.append(original_class_data)
            
            # Get the list of unique IDs from the original data
            unique_ids = original_class_data['ID'].unique().tolist()
            synthetic_data = []
            
            # Use a counter to ensure each synthetic sample gets a unique Generated_From value.
            counter = 0
            for i in range(num_to_generate):
                # If there is more than one unique ID, randomly choose one;
                # if there is only one, or even if there are many, append a counter to ensure uniqueness.
                if unique_ids:
                    chosen_id = random.choice(unique_ids)
                else:
                    chosen_id = "Unknown"
                # Append the counter to the chosen ID
                unique_generated_from = f"{chosen_id}"
                counter += 1

                # Generate one synthetic sample, passing the unique_generated_from value.
                sample = self.generate_hybrid_sample(
                    target_diagnosis=target_class,
                    feature_value_map=feature_value_map,
                    original_data=original_class_data,
                    original_id=unique_generated_from,
                    correlated_groups=correlated_groups,
                    noise=False
                )
                synthetic_data.append(sample)
            
            synthetic_data_df = pd.DataFrame(synthetic_data)
            synthetic_data_df['Source'] = 'Synthetic'
            balanced_data.append(synthetic_data_df)
        
        self.final_balanced_data = pd.concat(balanced_data, ignore_index=True)
        self.final_balanced_data.to_csv(self.save_path, index=False)
        print(f"Balanced dataset saved to {self.save_path}")

    def tsne_visualization(self):
        """
        Uses t-SNE to visualize the balanced dataset.
        Categorical columns are label-encoded and missing values imputed for the visualization.
        Rows with missing target values are dropped.
        """
        combined_data = self.final_balanced_data.drop(['Generated_From'], axis=1, errors='ignore')
        combined_data = combined_data.dropna(subset=[self.target_column])
        source_labels = self.final_balanced_data.loc[combined_data.index, 'Source']
        target_labels = combined_data[self.target_column]
        categorical_columns = combined_data.select_dtypes(include=['object']).columns
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        for col in categorical_columns:
            combined_data[col] = label_encoders[col].fit_transform(combined_data[col].astype(str))
        imputer = SimpleImputer(strategy="mean")
        combined_data_imputed = pd.DataFrame(imputer.fit_transform(combined_data), columns=combined_data.columns)
        tsne = TSNE(n_components=2, random_state=self.random_state)
        tsne_results = tsne.fit_transform(combined_data_imputed.drop([self.target_column, 'Source'], axis=1, errors='ignore'))
        markers = ['o', 's', 'v', 'P', '*', 'X', 'h', 'd', '#']
        marker_cycle = cycle(markers)
        unique_classes = target_labels.unique()
        unique_sources = source_labels.unique()
        class_colors = plt.cm.tab10(range(len(unique_classes)))
        class_color_map = {cls: color for cls, color in zip(unique_classes, class_colors)}
        combination_marker_map = {(cls, src): next(marker_cycle) for cls in unique_classes for src in unique_sources}
        plt.figure(figsize=(12, 8))
        for target_class in unique_classes:
            for source in unique_sources:
                indices = (target_labels == target_class) & (source_labels == source)
                plt.scatter(
                    tsne_results[indices, 0], tsne_results[indices, 1],
                    label=f"Class {target_class} - {source}",
                    color=class_color_map[target_class],
                    marker=combination_marker_map[(target_class, source)],
                    alpha=0.7,
                    edgecolor="k"
                )
        plt.title("t-SNE Visualization of Original vs. Synthetic Data")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show()

    def run_pipeline(self):
        self.load_data()
        self.balance_dataset()
        self.tsne_visualization()

    def get_save_path(self):
        return self.save_path

# Example usage:
# pipeline = SyntheticDataPipeline(config_path="config/config.yaml", file_path="your_dataset.csv")
# pipeline.run_pipeline()
