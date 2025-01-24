import os
import ray
import glob
import eli5
import shutil
import argparse
import json
import mlflow

import xgboost_ray as xgbr
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error

from ray import tune, ray

from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback


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
        with mlflow.start_run(nested=True) as run:
            # Log hyperparameters
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
            
            bst.save_model("model.xgb")
            mlflow.log_artifact("model.xgb")
    
    print("Training...")
    with mlflow.start_run(run_name="xgboost_tune") as parent_run:
        mlflow.log_params({
            "ray_actors": ray_actors,
            "cpus_per_actor": cpus_per_actor,
            "tune_samples": tune_samples
        })
        
        analysis = tune.run(
            my_trainer,
            config=config,
            resources_per_trial=xgb_tune_resources,
            local_dir=TUNE_ROOT,
            metric="val-error",
            mode="min",
            num_samples=tune_samples,
            verbose=1,
            progress_reporter=tune.CLIReporter(),
            callbacks=[MLflowLoggerCallback(experiment_name="credit_card_fraud", 
                                          save_artifact=True)]
        )
    
        shutil.copy(
            os.path.join(analysis.best_logdir, "model.xgb"),
            os.path.join(MODEL_ROOT, "tune_best.xgb")
        )
        
        # Load the serialized model
        bst = xgb.Booster(model_file=os.path.join(MODEL_ROOT, "tune_best.xgb"))

        # Make predictions on the test data
        predictions = xgbr.predict(bst, rdm_test, ray_params=xgb_ray_params)
        pred_class = (predictions > 0.5).astype("int") 
        actuals = df_test[target_col]
        
        # Calculate and log test metrics
        test_accuracy = accuracy_score(pred_class, actuals)
        test_precision = precision_score(pred_class, actuals)
        test_recall = recall_score(pred_class, actuals)
        test_f1 = f1_score(pred_class, actuals)
        
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })
        
        print("Accuracy on test: {:.2f}".format(test_accuracy))
        
        # Save Stats
        with open("dominostats.json", 'w') as f:
            f.write(json.dumps({"Accuracy": round(test_accuracy, 3)}))
        
        # Save and log plots
        plt.figure()
        ax1 = xgb.plot_importance(bst, importance_type="gain", max_num_features=10, show_values=False)
        plt.tight_layout()
        feature_importance_gain_path = os.path.join(MODEL_ROOT, "feature_importance_gain.png")
        ax1.figure.savefig(feature_importance_gain_path)
        mlflow.log_artifact(feature_importance_gain_path)
        
        plt.figure()
        ax2 = xgb.plot_importance(bst, importance_type="weight", max_num_features=10)
        plt.tight_layout()
        feature_importance_weight_path = os.path.join(MODEL_ROOT, "feature_importance_weight.png")
        ax2.figure.savefig(feature_importance_weight_path)
        mlflow.log_artifact(feature_importance_weight_path)
        
        plt.figure()
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
