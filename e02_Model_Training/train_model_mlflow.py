import os
import ray
import glob
import eli5
import shutil
import argparse
import json
import datetime
import socket
import warnings
from contextlib import redirect_stdout
import io

import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import DataType

import xgboost_ray as xgbr
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

from ray import tune, ray

from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback


def setup_mlflow(mlflow_server_url="http://localhost:8768"):
    """Setup MLFlow tracking with distributed Ray cluster support"""
    try:
        # For distributed Ray, use head node IP if localhost fails
        if "localhost" in mlflow_server_url:
            try:
                # Try to get head node IP for distributed cluster
                head_node_ip = socket.gethostbyname(socket.gethostname())
                mlflow_server_url = f"http://{head_node_ip}:8768"
                print(f"Using head node IP for MLFlow: {mlflow_server_url}")
            except:
                print(f"Using provided MLFlow server: {mlflow_server_url}")
        
        mlflow.set_tracking_uri(mlflow_server_url)
        experiment_name = f"Credit_Model_Training"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id
            else:
                raise
        
        print(f"MLFlow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Created/Using experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_name, experiment_id
        
    except Exception as e:
        print(f"Warning: Could not setup MLFlow: {e}")
        print("Continuing without MLFlow tracking...")
        return None, None


def create_feature_importance_plots(model, model_root, experiment_run=None):
    """Create and save feature importance plots"""
    plots = {}
    
    for importance_type in ["gain", "weight", "cover"]:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            xgb.plot_importance(model, importance_type=importance_type, 
                              max_num_features=10, show_values=False, ax=ax)
            plt.title(f"Feature Importance ({importance_type.capitalize()})")
            plt.tight_layout()
            
            # Save locally
            plot_path = os.path.join(model_root, f"feature_importance_{importance_type}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots[importance_type] = plot_path
            
            # Log to MLFlow if run is active
            if experiment_run and mlflow.active_run():
                mlflow.log_artifact(plot_path)
            
            plt.close()
            
        except Exception as e:
            print(f"Could not create {importance_type} importance plot: {e}")
    
    return plots


def save_eli5_explanations(model, test_data, target_col, model_root, experiment_run=None):
    """Save ELI5 explanations with error handling"""
    eli5_files = []
    
    try:
        # Feature weights explanation
        try:
            eli5_display = eli5.show_weights(model)
            eli5_html = eli5_display.data
        except Exception as e:
            print(f"ELI5 feature weights failed: {e}")
            # Fallback to XGBoost native importance
            importance_dict = model.get_score(importance_type='gain')
            eli5_html = "<html><body><h2>Feature Importance (XGBoost Gain)</h2>"
            for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                eli5_html += f"<p>{feature}: {importance:.4f}</p>"
            eli5_html += "</body></html>"
        
        weights_path = os.path.join(model_root, "eli5_feature_weights.html")
        with open(weights_path, 'w') as f:
            f.write(eli5_html)
        eli5_files.append(weights_path)
        
        # Individual prediction explanation
        try:
            test_X = test_data.drop(target_col, axis=1)
            sample_id = 3
            
            eli5_pred_display = eli5.show_prediction(model, test_X.iloc[sample_id], 
                                     feature_names=list(test_X.columns),
                                     show_feature_values=True)
            pred_html = eli5_pred_display.data
        except Exception as e:
            print(f"ELI5 prediction explanation failed: {e}")
            pred_html = f"""<html><body>
            <h2>Prediction Explanation for Sample {sample_id}</h2>
            <p>ELI5 explanation failed due to feature compatibility issues.</p>
            </body></html>"""
        
        pred_path = os.path.join(model_root, "eli5_prediction_explanation.html")
        with open(pred_path, 'w') as f:
            f.write(pred_html)
        eli5_files.append(pred_path)
        
        # Log to MLFlow if run is active
        if experiment_run and mlflow.active_run():
            for file_path in eli5_files:
                mlflow.log_artifact(file_path)
                
    except Exception as e:
        print(f"ELI5 explanations failed: {e}")
    
    return eli5_files


def train(ray_actors,
          cpus_per_actor,
          tune_samples,
          mlflow_server_url="http://localhost:8768",
          DATA_ROOT = os.path.join("/mnt/data", os.environ["DOMINO_PROJECT_NAME"], "data"), 
          MODEL_ROOT = "/mnt/artifacts",
          TUNE_ROOT = os.path.join("/mnt/data", os.environ["DOMINO_PROJECT_NAME"], "ray_results")):
    
    # Setup MLFlow
    experiment_name, experiment_id = setup_mlflow(mlflow_server_url)
    
    # End any existing runs
    mlflow.end_run()
    
    print("Connecting to Ray...")
    if ray.is_initialized() == False:
        service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
        service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
        ray.init(f"ray://{service_host}:{service_port}")
        
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
    
    config = {
        "seed": 1234,
        "eta": tune.loguniform(3e-3, 3e-1),
        "max_depth": tune.randint(2, 6),
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"]
    }  

    def my_trainer(config):
        """Simplified trainer for distributed Ray - MLFlow logging happens post-processing"""
        evals_result = {}
        bst = xgbr.train(
            params=config,
            dtrain=rdm_train,
            num_boost_round=50,
            evals_result=evals_result,
            evals=[(rdm_train, "train"), (rdm_val, "val")],
            ray_params=xgb_ray_params
        )
        bst.save_model("model.xgb") # This will go into the TUNE_ROOT directory
    
    # Start parent MLFlow run for hyperparameter tuning
    if experiment_id:
        with mlflow.start_run(experiment_id=experiment_id, run_name="hyperparameter_tuning") as parent_run:
            # Log hyperparameter tuning configuration
            mlflow.log_param("ray_actors", ray_actors)
            mlflow.log_param("cpus_per_actor", cpus_per_actor)
            mlflow.log_param("num_samples", tune_samples)
            mlflow.log_param("search_space_eta_min", 3e-3)
            mlflow.log_param("search_space_eta_max", 3e-1)
            mlflow.log_param("search_space_max_depth_min", 2)
            mlflow.log_param("search_space_max_depth_max", 6)
            mlflow.log_param("metric", "val-error")
            mlflow.log_param("mode", "min")
            mlflow.log_param("num_boost_rounds", 50)
            
            print("Training...")
            # Suppress Ray Tune warnings
            warnings.filterwarnings("ignore", message="The `local_dir` argument of `Experiment is deprecated")
            
            analysis = tune.run(
                my_trainer,
                config=config,
                resources_per_trial=xgb_tune_resources,
                local_dir=TUNE_ROOT,
                metric="val-error",  # Use hyphen format that XGBoost reports
                mode="min",
                num_samples=tune_samples,
                verbose=1,
                progress_reporter=tune.CLIReporter()
            )
            
            # Log best trial results to parent run
            best_trial = analysis.best_trial
            best_config = analysis.best_config
            best_result = analysis.best_result

            mlflow.log_params({f"best_{k}": v for k, v in best_config.items()})
            mlflow.log_metric("best_val_error", best_result["val-error"])
            
            # Log metrics for the best trial
            if "train-error" in best_result:
                mlflow.log_metric("best_train_error", best_result["train-error"])
            if "val-logloss" in best_result:
                mlflow.log_metric("best_val_logloss", best_result["val-logloss"])
            if "train-logloss" in best_result:
                mlflow.log_metric("best_train_logloss", best_result["train-logloss"])

            # Create child runs for each trial
            trial_results = []
            for i, trial in enumerate(analysis.trials):
                if trial.last_result:
                    with mlflow.start_run(nested=True, run_name=f"trial_{i+1}_eta_{trial.config['eta']:.4f}_depth_{trial.config['max_depth']}"):
                        # Log trial parameters
                        mlflow.log_params(trial.config)
                        
                        # Log trial metrics (using hyphen format from XGBoost)
                        result = trial.last_result
                        mlflow.log_metric("val_error", result.get("val-error", 0))
                        if "train-error" in result:
                            mlflow.log_metric("train_error", result["train-error"])
                        if "val-logloss" in result:
                            mlflow.log_metric("val_logloss", result["val-logloss"])
                        if "train-logloss" in result:
                            mlflow.log_metric("train_logloss", result["train-logloss"])
                        
                        # Try to log the model from this trial
                        try:
                            trial_model_path = os.path.join(trial.logdir, "model.xgb")
                            if os.path.exists(trial_model_path):
                                trial_model = xgb.Booster(model_file=trial_model_path)
                                sample_input = df_test.drop(target_col, axis=1).head(3)
                                sample_predictions = trial_model.predict(xgb.DMatrix(sample_input))
                                signature = infer_signature(sample_input, sample_predictions)
                                
                                mlflow.xgboost.log_model(
                                    trial_model,
                                    "model", 
                                    signature=signature,
                                    input_example=sample_input
                                )
                        except Exception as e:
                            print(f"Could not log model for trial {i+1}: {e}")
                    
                    # Add to summary
                    trial_results.append({
                        "trial_id": trial.trial_id,
                        "eta": trial.config["eta"],
                        "max_depth": trial.config["max_depth"],
                        "val_error": result.get("val-error", None),
                        "train_error": result.get("train-error", None),
                        "val_logloss": result.get("val-logloss", None),
                        "train_logloss": result.get("train-logloss", None)
                    })

            # Save trial results as JSON artifact
            trial_results_path = os.path.join(MODEL_ROOT, "trial_results.json")
            with open(trial_results_path, 'w') as f:
                json.dump(trial_results, f, indent=2)
            mlflow.log_artifact(trial_results_path)

            print(f"Best trial: {best_trial}")
            print(f"Best config: {best_config}")
            print(f"Best validation error: {best_result['val-error']:.4f}")
    else:
        # Run without MLFlow if setup failed
        print("Training without MLFlow...")
        warnings.filterwarnings("ignore", message="The `local_dir` argument of `Experiment is deprecated")
        
        analysis = tune.run(
            my_trainer,
            config=config,
            resources_per_trial=xgb_tune_resources,
            local_dir=TUNE_ROOT,
            metric="val-error",
            mode="min",
            num_samples=tune_samples,
            verbose=1,
            progress_reporter=tune.CLIReporter()
        )
    
    # Copy best model locally
    shutil.copy(
        os.path.join(analysis.best_logdir, "model.xgb"),
        os.path.join(MODEL_ROOT, "tune_best.xgb")
    )
    
    # Final best model run
    if experiment_id:
        mlflow.end_run()  # End hyperparameter tuning run
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="best_model_evaluation"):
            # Log the best hyperparameters
            best_config = analysis.best_config
            best_result = analysis.best_result
            
            mlflow.log_params(best_config)
            mlflow.log_metric("best_val_error", best_result["val-error"])
            
            # Load and log the best model to MLFlow
            best_model = xgb.Booster(model_file=os.path.join(MODEL_ROOT, "tune_best.xgb"))
            
            # Prepare sample data for signature inference
            sample_input = df_test.drop(target_col, axis=1).head(5)
            sample_predictions = best_model.predict(xgb.DMatrix(sample_input))
            signature = infer_signature(sample_input, sample_predictions)
            
            mlflow.xgboost.log_model(
                best_model, 
                "best_model",
                signature=signature,
                input_example=sample_input,
                registered_model_name=f"{experiment_name}"
            )
            
            # Make predictions on test data
            predictions = xgbr.predict(best_model, rdm_test, ray_params=xgb_ray_params)
            pred_class = (predictions > 0.5).astype("int") 
            actuals = df_test[target_col]
            test_accuracy = accuracy_score(pred_class, actuals)
            
            # Log test metrics
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_param("prediction_threshold", 0.5)
            mlflow.log_param("test_samples", len(actuals))
            
            print("Accuracy on test: {:.2f}".format(test_accuracy))
            
            # Save predictions as artifacts
            predictions_df = pd.DataFrame({
                'predictions': predictions,
                'predicted_class': pred_class,
                'actual': actuals
            })
            predictions_path = os.path.join(MODEL_ROOT, "test_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            mlflow.log_artifact(predictions_path)
            
            # Create and save plots
            print("Creating feature importance plots...")
            plots = create_feature_importance_plots(best_model, MODEL_ROOT, experiment_run=True)
            
            # Get feature importance scores and log as metrics
            try:
                importance_dict = best_model.get_score(importance_type='gain')
                for feature, importance in importance_dict.items():
                    mlflow.log_metric(f"feature_importance_{feature}", importance)
                
                # Save feature importance as JSON artifact
                importance_json_path = os.path.join(MODEL_ROOT, "feature_importance.json")
                with open(importance_json_path, 'w') as f:
                    json.dump(importance_dict, f, indent=2)
                mlflow.log_artifact(importance_json_path)
            except Exception as e:
                print(f"Could not log feature importance metrics: {e}")
            
            # Save ELI5 explanations
            print("Creating ELI5 explanations...")
            eli5_files = save_eli5_explanations(best_model, df_test, target_col, MODEL_ROOT, experiment_run=True)
            
            print(f"Best model saved to: {os.path.join(MODEL_ROOT, 'tune_best.xgb')}")
            print(f"Best model logged to MLFlow with test accuracy: {test_accuracy:.4f}")
    else:
        # Without MLFlow, still do local processing
        best_model = xgb.Booster(model_file=os.path.join(MODEL_ROOT, "tune_best.xgb"))
        predictions = xgbr.predict(best_model, rdm_test, ray_params=xgb_ray_params)
        pred_class = (predictions > 0.5).astype("int") 
        actuals = df_test[target_col]
        test_accuracy = accuracy_score(pred_class, actuals)
        
        print("Accuracy on test: {:.2f}".format(test_accuracy))
        
        # Create plots
        create_feature_importance_plots(best_model, MODEL_ROOT)
        
        # Save ELI5 explanations
        save_eli5_explanations(best_model, df_test, target_col, MODEL_ROOT)
    
    # Save Stats (for Domino compatibility)
    with open("dominostats.json", 'w') as f:
        f.write(json.dumps({"Accuracy": round(test_accuracy, 3)}))
    
    print("Training completed!")
    return test_accuracy


def main():
        
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ray_actors', type=int, default=3, help="Number of Ray actors.")
    parser.add_argument('--cpus_per_actor', type=int, default=4, help="CPUs per Ray actor.")
    parser.add_argument('--tune_samples', type=int, default=10, help="Number of models to try over the search space (for Ray Tune).")
    parser.add_argument('--mlflow_server', type=str, default="http://localhost:8768", help="MLFlow tracking server URL.")
    
    args = parser.parse_args()

    train(args.ray_actors, args.cpus_per_actor, args.tune_samples, args.mlflow_server)


if __name__ == "__main__":
    main()