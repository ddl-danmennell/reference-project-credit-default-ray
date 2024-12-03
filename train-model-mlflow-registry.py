import os
import ray
import glob
import eli5
import shutil
import argparse
import json
import mlflow
from mlflow.tracking import MlflowClient
import time

import xgboost_ray as xgbr
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error

from ray import tune, ray
from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback

def get_model_version(client, model_name, run_id):
    """Helper function to get model version if it exists"""
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.run_id == run_id:
            return mv.version
    return None

def train(ray_actors,
          cpus_per_actor,
          tune_samples,
          DATA_ROOT = os.path.join("/mnt/data", os.environ["DOMINO_PROJECT_NAME"], "data"), 
          MODEL_ROOT = "/mnt/artifacts",
          TUNE_ROOT = os.path.join("/mnt/data", os.environ["DOMINO_PROJECT_NAME"], "ray_results")):
    
    print("Connecting to Ray...")
    if ray.is_initialized() == False:
        service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
        service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
        ray.init(f"ray://{service_host}:{service_port}")
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("credit_card_fraud_detection")
    client = MlflowClient()
    
    # Model name for registry
    model_name = "credit_card_fraud_detector"
    
    # Try to create the registered model (ignore if already exists)
    try:
        client.create_registered_model(model_name)
    except:
        pass
        
    print("Loading data...")    
    train_files = glob.glob(os.path.join(DATA_ROOT, "train_data*"))
    val_files = glob.glob(os.path.join(DATA_ROOT, "validation_data*"))
    test_file = os.path.join(DATA_ROOT, "test_data.csv")

    target_col = "credit"
    
    rdm_train = xgbr.RayDMatrix(train_files, label=target_col)
    rdm_val = xgbr.RayDMatrix(val_files, label=target_col)
    df_test = pd.read_csv(test_file)
    rdm_test = xgbr.RayDMatrix(df_test, label=target_col)
    
    rdm_train.assert_enough_shards_for_actors(len(train_files))
    rdm_train.assert_enough_shards_for_actors(len(val_files))
    
    xgb_ray_params = xgbr.RayParams(
        num_actors=ray_actors,
        cpus_per_actor=cpus_per_actor
    )  
        
    xgb_tune_resources = xgb_ray_params.get_tune_resources()
    print(f"It will request {xgb_tune_resources.required_resources} per trial.")
    print(f"The cluster has {ray.cluster_resources()['CPU']} CPU total.")
    print("Saving intermediate tune results to", TUNE_ROOT)
    
    # Expanded hyperparameter search space
    config = {
        "seed": 1234,
        "eta": tune.loguniform(3e-3, 3e-1),
        "max_depth": tune.randint(2, 8),
        "min_child_weight": tune.choice([1, 2, 3, 4, 5]),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "gamma": tune.uniform(0, 1),
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"]
    }  

    def my_trainer(config):
        # Start a child run for this hyperparameter trial
        with mlflow.start_run(nested=True, run_name=f"trial_{int(time.time())}") as child_run:
            # Log hyperparameters for this trial
            mlflow.log_params(config)
            
            evals_result = {}
            bst = xgbr.train(
                params=config,
                dtrain=rdm_train,
                num_boost_round=50,
                evals_result=evals_result,
                evals=[(rdm_train, "train"), (rdm_val, "val")],
                ray_params=xgb_ray_params
            )
            
            # Log training metrics
            for metric in ["error", "logloss"]:
                for epoch, value in enumerate(evals_result["train"][metric]):
                    mlflow.log_metric(f"train_{metric}", value, step=epoch)
                for epoch, value in enumerate(evals_result["val"][metric]):
                    mlflow.log_metric(f"val_{metric}", value, step=epoch)
            
            # Save model for this trial
            model_path = "model.xgb"
            bst.save_model(model_path)
            mlflow.log_artifact(model_path)
            
            # Log validation error for Ray Tune
            tune.report(val_error=evals_result["val"]["error"][-1])
    
    print("Training...")
    best_val_error = float('inf')
    best_run_id = None
    
    with mlflow.start_run(run_name="xgboost_tune_parent") as parent_run:
        # Log parent run parameters
        mlflow.log_params({
            "ray_actors": ray_actors,
            "cpus_per_actor": cpus_per_actor,
            "tune_samples": tune_samples,
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        })
        
        # Tag the parent run
        client.set_tag(parent_run.info.run_id, "run_type", "hyperparameter_sweep")
        
        analysis = tune.run(
            my_trainer,
            config=config,
            resources_per_trial=xgb_tune_resources,
            local_dir=TUNE_ROOT,
            metric="val_error",
            mode="min",
            num_samples=tune_samples,
            verbose=1,
            progress_reporter=tune.CLIReporter()
        )
        
        # Get the best trial
        best_trial = analysis.best_trial
        best_val_error = best_trial.last_result["val_error"]
        
        # Copy best model
        shutil.copy(
            os.path.join(best_trial.logdir, "model.xgb"),
            os.path.join(MODEL_ROOT, "tune_best.xgb")
        )
        
        # Load the best model
        bst = xgb.Booster(model_file=os.path.join(MODEL_ROOT, "tune_best.xgb"))

        # Make predictions on the test data
        predictions = xgbr.predict(bst, rdm_test, ray_params=xgb_ray_params)
        pred_class = (predictions > 0.5).astype("int") 
        actuals = df_test[target_col]
        
        # Calculate test metrics
        test_metrics = {
            "test_accuracy": accuracy_score(pred_class, actuals),
            "test_precision": precision_score(pred_class, actuals),
            "test_recall": recall_score(pred_class, actuals),
            "test_f1": f1_score(pred_class, actuals)
        }
        
        # Log metrics to parent run
        mlflow.log_metrics(test_metrics)
        
        # Save best model to MLflow
        mlflow.xgboost.log_model(
            bst,
            "model",
            registered_model_name=model_name
        )
        
        # Get the version number of the registered model
        model_version = get_model_version(client, model_name, parent_run.info.run_id)
        
        # Add tags and description to the model version
        if model_version:
            client.update_model_version(
                name=model_name,
                version=model_version,
                description=f"XGBoost model trained with {tune_samples} trials. "
                           f"Best validation error: {best_val_error:.4f}, "
                           f"Test accuracy: {test_metrics['test_accuracy']:.4f}"
            )
            
            # Tag the model version
            client.set_model_version_tag(model_name, model_version, "validation_error", f"{best_val_error:.4f}")
            client.set_model_version_tag(model_name, model_version, "test_accuracy", f"{test_metrics['test_accuracy']:.4f}")
            client.set_model_version_tag(model_name, model_version, "training_timestamp", time.strftime("%Y%m%d-%H%M%S"))
        
        print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        
        # Save Stats for Domino
        with open("dominostats.json", 'w') as f:
            f.write(json.dumps({"Accuracy": round(test_metrics['test_accuracy'], 3)}))
        
        # Generate and save plots
        plt.figure(figsize=(10, 6))
        ax1 = xgb.plot_importance(bst, importance_type="gain", max_num_features=10, show_values=False)
        plt.tight_layout()
        feature_importance_gain_path = os.path.join(MODEL_ROOT, "feature_importance_gain.png")
        ax1.figure.savefig(feature_importance_gain_path)
        mlflow.log_artifact(feature_importance_gain_path)
        
        plt.figure(figsize=(10, 6))
        ax2 = xgb.plot_importance(bst, importance_type="weight", max_num_features=10)
        plt.tight_layout()
        feature_importance_weight_path = os.path.join(MODEL_ROOT, "feature_importance_weight.png")
        ax2.figure.savefig(feature_importance_weight_path)
        mlflow.log_artifact(feature_importance_weight_path)
        
        plt.figure(figsize=(10, 6))
        ax3 = xgb.plot_importance(bst, importance_type="cover", max_num_features=10, show_values=False)
        plt.tight_layout()
        feature_importance_cover_path = os.path.join(MODEL_ROOT, "feature_importance_cover.png")
        ax3.figure.savefig(feature_importance_cover_path)
        mlflow.log_artifact(feature_importance_cover_path)
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ray_actors', type=int, default=3, help="Number of Ray actors.")
    parser.add_argument('--cpus_per_actor', type=int, default=4, help="CPUs per Ray actor.")
    parser.add_argument('--tune_samples', type=int, default=10, help="Number of models to try over the search space (for Ray Tune).")
    
    args = parser.parse_args()

    train(args.ray_actors, args.cpus_per_actor, args.tune_samples)

if __name__ == "__main__":
    main()
