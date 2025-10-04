import hydra
from omegaconf import DictConfig
from src.automl_classifier import AutoMLClassifier
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Access nested configuration
    dataset_config = cfg.dataset
    randomness_config = cfg.randomness
    automl_config = cfg.automl
    output_config = cfg.output

    # Determine folder path for saving outputs
    folder_path = output_config.save_path
    save_path = os.path.join(output_config.save_path, output_config.folder_name)
    os.makedirs(folder_path, exist_ok=True)  
    print(f"Saving outputs in: {folder_path}")

    # Check selected approach
    approach = cfg.get("approach")  # Default to "h2o" if not specified
    print(f"Selected approach: {approach}")

    # Create AutoMLClassifier instance with the selected approach
    model = AutoMLClassifier(
        approach=approach,
        csv_path=dataset_config.csv_path,
        target_column=dataset_config.target_column,
        k_features=dataset_config.k_features,
        test_size=dataset_config.test_size,
        random_state=randomness_config.random_state,
        max_models=automl_config.max_models,
        max_runtime_secs=automl_config.max_runtime_secs,
        nfolds=automl_config.nfolds,
        save_path=save_path,
        folder_path=folder_path    # Pass folder path for saving outputs
    )

    # Run the full pipeline
    model.full_pipeline()

if __name__ == "__main__":
    main()
