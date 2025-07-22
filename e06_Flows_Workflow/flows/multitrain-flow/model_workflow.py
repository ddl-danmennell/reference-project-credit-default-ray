from flytekit import workflow
from flytekit.types.file import FlyteFile
from typing import TypeVar, NamedTuple
from flytekitplugins.domino.helpers import run_domino_job_task, Input, Output
from flytekitplugins.domino.task import DatasetSnapshot
from flytekitplugins.domino.artifact import Artifact, DATA, MODEL, REPORT
import os

# Set default Compute Environment and Hardware Tier for all tasks. 
environment_name = "Credit Workshop Test"
hardware_tier_name = "Small"


# Enter the name of your project's default dataset. 
# Ensure you have taken a snapshot of that dataset in order for it to be mounted to your flow tasks.
dataset_name="Testing_Credit_Workshop"
snapshot_number=1


# Enter the command below to run this Flow. 
# pyflyte run --remote ./flows/model_workflow.py model_training_workflow


# Define Flow Artifacts to capture for each model training task
sklearn_log_regArtifact = Artifact("scikit-learn Logistic Regression", MODEL)
h2oArtifact = Artifact("H20 AutoML", MODEL)
sklearn_rfArtifact = Artifact("scikit-learn Random Forest", MODEL)
xgboostArtifact = Artifact("XGBoost", MODEL)


@workflow
def model_training_workflow():
    """
    Workflow that runs multiple model training jobs in parallel.
    Returns trained model files for each algorithm as seperate Flow Artifacts.
    """
    # Launch sklearn logistic regression training
    sklearn_log_reg_results = run_domino_job_task(
        flyte_task_name="Train Sklearn LogReg",
        command="python flows/sklearn_log_reg_train.py",
        output_specs=[Output(name="model", type=sklearn_log_regArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
        dataset_snapshots=[
            DatasetSnapshot(Name=dataset_name, Version=snapshot_number)
        ]
    )

    # Launch H2O model training
    h2o_results = run_domino_job_task(
        flyte_task_name="Train H2O Model",
        command="python flows/h2o_model_train.py",
        output_specs=[Output(name="model", type=h2oArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
        dataset_snapshots=[
            DatasetSnapshot(Name=dataset_name, Version=snapshot_number)
        ]
    )

    # Launch sklearn random forest training
    sklearn_rf_results = run_domino_job_task(
        flyte_task_name="Train Sklearn RF",
        command="python flows/sklearn_RF_train.py",
        output_specs=[Output(name="model", type=sklearn_rfArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
        dataset_snapshots=[
            DatasetSnapshot(Name=dataset_name, Version=snapshot_number)
        ]
    )

    # Launch XGBoost model training
    xgboost_results = run_domino_job_task(
        flyte_task_name="Train XGBoost",
        command="python flows/xgb_model_train.py",
        output_specs=[Output(name="model", type=xgboostArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
        dataset_snapshots=[
            DatasetSnapshot(Name=dataset_name, Version=snapshot_number)
        ]
    )

    return