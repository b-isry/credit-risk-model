import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    Load the processed customer features data.
    """
    logging.info(f"Loading processed data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise

def evaluate_model(y_true, y_pred, y_proba):
    """Calculates and returns a dictionary of evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plot_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def promote_best_model(experiment_name: str, registry_model_name: str):
    """
    Finds the best model in an experiment, registers it, and assigns it an
    alias if it's better than the current 'champion' model.
    """
    logging.info("--- Starting Model Promotion Process using Aliases ---")
    client = MlflowClient()

    try:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    except AttributeError:
        logging.error(f"Experiment '{experiment_name}' not found. Aborting promotion.")
        return

    # --- 1. Get all runs and filter by a performance threshold ---
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.roc_auc DESC"])
    MIN_ROC_AUC_THRESHOLD = 0.75
    candidate_runs = runs[runs['metrics.roc_auc'] > MIN_ROC_AUC_THRESHOLD]

    if candidate_runs.empty:
        logging.warning(f"No models met the ROC AUC threshold of {MIN_ROC_AUC_THRESHOLD}. No model will be promoted.")
        return

    best_overall_run = candidate_runs.iloc[0]
    
    # --- 2. Apply business rule: prefer simpler model if performance is similar ---
    best_roc_auc = best_overall_run["metrics.roc_auc"]
    PERFORMANCE_TOLERANCE = 0.01

    lr_runs = candidate_runs[candidate_runs['params.model_type'] == 'LogisticRegression']
    final_candidate_run = best_overall_run
    
    if not lr_runs.empty:
        best_lr_run = lr_runs.iloc[0]
        lr_roc_auc = best_lr_run["metrics.roc_auc"]
        if (best_roc_auc - lr_roc_auc) < PERFORMANCE_TOLERANCE:
            final_candidate_run = best_lr_run
            logging.info(f"Prioritizing simpler model 'LogisticRegression' (ROC AUC: {lr_roc_auc:.4f}) over complex one (ROC AUC: {best_roc_auc:.4f}).")

    final_candidate_run_id = final_candidate_run['run_id']
    final_candidate_model_type = final_candidate_run['params.model_type']
    final_candidate_roc_auc = final_candidate_run['metrics.roc_auc']
    
    logging.info(f"Final candidate model selected: {final_candidate_model_type} from run_id: {final_candidate_run_id} with ROC AUC: {final_candidate_roc_auc:.4f}")

    # --- 3. Register the final candidate model ---
    model_uri = f"runs:/{final_candidate_run_id}/{final_candidate_model_type}"
    try:
        client.get_registered_model(name=registry_model_name)
    except Exception:
        logging.info(f"Creating new model registry name: {registry_model_name}")
        client.create_registered_model(name=registry_model_name)

    new_model_version = mlflow.register_model(model_uri=model_uri, name=registry_model_name)
    logging.info(f"Registered model '{registry_model_name}' version {new_model_version.version}.")

    # --- 4. Compare with 'champion' model and assign alias if better ---
    try:
        current_champion_version = client.get_model_version_by_alias(registry_model_name, "champion")
        
        champion_run = client.get_run(current_champion_version.run_id)
        champion_roc_auc = champion_run.data.metrics.get("roc_auc_score", 0)
        
        logging.info(f"Comparing new candidate (ROC AUC: {final_candidate_roc_auc:.4f}) with current champion model version {current_champion_version.version} (ROC AUC: {champion_roc_auc:.4f}).")
        
        if final_candidate_roc_auc > champion_roc_auc:
            logging.info("New model is better. Setting it as the new 'champion-candidate'.")

            client.set_registered_model_alias(name=registry_model_name, alias="champion-candidate", version=new_model_version.version)
        else:
            logging.info("New model is not better than the current champion. No alias will be assigned.")
            
    except Exception as e:
        error_message = str(e)
        if ("RESOURCE_DOES_NOT_EXIST" in error_message or
            "Registered model alias champion not found" in error_message):
            logging.info("No model currently has the 'champion' alias. Setting new model as 'champion-candidate'.")
            client.set_registered_model_alias(name=registry_model_name, alias="champion-candidate", version=new_model_version.version)
        else:
            logging.error(f"An unexpected error occurred while checking for champion model: {e}")
def main():
    """Main function to run the model training and evaluation pipeline."""
    
    EXPERIMENT_NAME = "Credit_Risk_Modeling"
    REGISTRY_MODEL_NAME = "CreditRiskModel-BNPL"

    # --- 1. Setup MLflow ---
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # --- 2. Load and Prepare Data ---
    processed_data_path = 'data/processed/customer_features.csv'
    df = load_processed_data(processed_data_path)
    
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")

    # --- 3. Define Models to Train ---
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10) # Added some hyperparameters
    }

    # --- 4. Train, Evaluate, and Log Models ---
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            logging.info(f"--- Training {model_name} ---")
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            
            # Ensure all metrics are primitive types (float)
            metrics = {k: float(v) for k, v in metrics.items()}
            mlflow.log_metrics(metrics)
            logging.info(f"Metrics for {model_name}: {metrics}")
            
            cm_plot_path = plot_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(cm_plot_path, "plots")
            os.remove(cm_plot_path)
            
            mlflow.sklearn.log_model(model, artifact_path=model_name)
            logging.info(f"Successfully trained and logged {model_name}.")

    # --- 5. Identify and Register the Best Model ---
    logging.info("Identifying best model from MLflow runs...")
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    df_runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.roc_auc_score DESC"])

    if not df_runs.empty:
        best_run = df_runs.iloc[0]
        best_run_id = best_run['run_id']
        model_artifact_name = best_run['params.model_type']
        
        model_uri = f"runs:/{best_run_id}/{model_artifact_name}"
        
        logging.info(f"Best model is '{model_artifact_name}' with run_id: {best_run_id}")
        logging.info(f"Model URI for registration: {model_uri}")

        # Register the model in the MLflow Model Registry
        # Using your custom model name 'CreditRiskModel-BNPL'
        model_name_for_registry = REGISTRY_MODEL_NAME
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name_for_registry)
        
        logging.info(f"Registered model '{model_name_for_registry}' version {registered_model.version}.")

        client = mlflow.tracking.MlflowClient()
        logging.info("Setting new model alias...")
        client.set_registered_model_alias(
            name=model_name_for_registry, 
            alias="champion-candidate", 
            version=registered_model.version
        )
        logging.info(f"Alias 'champion-candidate' set for version {registered_model.version}.")

    else:
        logging.warning("Could not find any runs to select the best model from.")

    logging.info("--- Training and Promotion Pipeline Complete ---")

if __name__ == '__main__':
    main()