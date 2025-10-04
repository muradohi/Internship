import os
import json
import yaml
import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load the configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access output parameters
save_path = os.path.join(config['output']['save_path'], config['output']['folder_name'])
approach = config['approach']


def dataloader(path):
    data_df = pd.read_csv(path)
    return data_df


def save_model_info_json(best_model, save_path, approach=approach):
    if approach == "h2o":
        model_info = {
            "model_id": best_model.model_id,
            "algorithm": best_model.algo,
            "parameters": {param: best_model.params[param]['actual'] for param in best_model.params.keys()},
            "description": "This model was selected for its performance in AutoML and compatibility with explainability tools."
        }
    else:  # sklearn
        model_info = {
            "model": str(best_model),
            "description": "This model was selected for its performance with hyperparameter tuning using sklearn."
        }

    os.makedirs(save_path, exist_ok=True)
    json_path = os.path.join(save_path, "model_info.json")
    with open(json_path, "w") as json_file:
        json.dump(model_info, json_file, indent=4)
    print(f"Model information saved to: {json_path}")


def save_explain_plots(model, data, save_path, approach=approach):
    save_dir = save_path
    os.makedirs(save_dir, exist_ok=True)
    if approach == "h2o":
        obj = model.explain(data, render=False)
        for key in obj.keys():
            if not obj.get(key).get("plots"):
                continue
            key_dir = os.path.join(save_dir, key)
            os.makedirs(key_dir, exist_ok=True)
            plots = obj.get(key).get("plots").keys()
            for plot in plots:
                fig = obj.get(key).get("plots").get(plot).figure()
                file_name = f"{plot}.png"
                file_path = os.path.join(key_dir, file_name)
                fig.savefig(file_path)
                plt.close(fig)
                print(f"Saved plot: {file_path}")


def h2o_train_test(config):
    csv_path = config['dataset']['csv_path']
    target_column = config['dataset']['target_column']
    ratios = config.get('ratios', [0.8])
    seed = config['randomness']['seed']

    h2o.init()
    data_h2o = h2o.H2OFrame(dataloader(csv_path))
    # Drop the "Generated_From" column if it exists so that it is not used for training
    if "Generated_From" in data_h2o.columns:
        data_h2o = data_h2o.drop("Generated_From")
        
    trainset, testset = data_h2o.split_frame(ratios=ratios, seed=seed)
    
    # Define predictor columns, excluding the target and (if still present) the Generated_From column
    x = [col for col in data_h2o.columns if col not in [target_column, "Generated_From"]]

    best_model = H2OAutoML(
        max_runtime_secs=config['automl']['max_runtime_secs'],
        nfolds=config['automl']['nfolds'],
        max_models=config['automl']['max_models'],
        include_algos=config['automl'].get('include_algos'),
        balance_classes=True,
        seed=seed
    )
    best_model.train(x=x, y=target_column, training_frame=trainset)

    test_df = testset[target_column].as_data_frame()
    predictions = best_model.predict(testset).as_data_frame()
    report = classification_report(test_df, predictions['predict'])
    print(report)

    save_model_info_json(best_model.leader, save_path, approach="h2o")
    save_explain_plots(best_model.leader, testset, save_path, approach="h2o")


def sklearn_train_test(config):
    data_df = dataloader(config['dataset']['csv_path'])
    target_column = config['dataset']['target_column']

    # Drop the "Generated_From" column if it exists so that the model is not trained on it
    if "Generated_From" in data_df.columns:
        data_df = data_df.drop(columns=["Generated_From"])
        
    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['dataset']['test_size'], random_state=config['randomness']['random_state']
    )

    # Path to the saved model
    model_path = "/Users/murad/Master_Internship_Project/exai_fig/best_model.pkl"
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The saved model was not found at {model_path}. Please ensure the model is trained and saved.")

    # Load the saved model
    best_model = joblib.load(model_path)
    print(f"Loaded model: {best_model}")

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    save_model_info_json(best_model, save_path, approach="sklearn")


# To use one of these functions, simply call it with the configuration.
# For example, to use H2O AutoML:
if approach == "h2o":
    h2o_train_test(config)
else:
    sklearn_train_test(config)
