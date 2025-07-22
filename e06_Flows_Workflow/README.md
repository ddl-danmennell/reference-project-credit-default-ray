# e06 Flows Workflow

This directory contains workflow orchestration examples using Flyte and Domino's workflow capabilities for building scalable ML pipelines.

## Overview

The Flows/Workflow system provides a declarative approach to orchestrating complex data processing and machine learning pipelines. It demonstrates both sequential ETL workflows and parallel model training patterns using Flyte integrated with Domino's infrastructure.

## Directory Structure

```
e06_Flows_Workflow/
├── data/                  # Sample datasets
│   ├── datasetA.csv      # Iris dataset partition A
│   ├── datasetB.csv      # Iris dataset partition B
│   ├── datasetC.csv      # Iris dataset partition C
│   └── datasetD.csv      # Iris dataset partition D
├── flows/                # Workflow definitions
│   ├── sample-flow.py    # Sequential ETL + ML pipeline
│   ├── multitrain-flow.py # Parallel model training
│   ├── sample-flow/      # Sequential workflow tasks
│   └── multitrain-flow/  # Parallel workflow tasks
└── models/              # Trained model artifacts
    ├── sklearn_logreg.pkl
    ├── sklearn_rf.pkl
    ├── xgb_clf.pkl
    └── GBM_*_AutoML_*    # H2O AutoML models
```

## Workflow Examples

### 1. Sample Flow (Sequential Pipeline)

A classic ETL + ML pipeline that demonstrates:
- **Parallel Data Loading**: Simultaneous loading of datasets A and B
- **Data Merging**: Combining datasets on common ID
- **Data Processing**: Feature engineering and transformation
- **Model Training**: Random Forest classifier training

#### Workflow Steps:
```
Load Data A ─┐
             ├─→ Merge Data → Process Data → Train Model
Load Data B ─┘
```

#### Key Features:
- Task dependencies management
- File-based data passing
- Configurable compute resources
- Error handling and logging

### 2. Multi-Train Flow (Parallel Training)

Demonstrates parallel model training with different algorithms:
- **Scikit-learn Logistic Regression**: With hyperparameter tuning
- **H2O AutoML**: Automated model selection
- **Scikit-learn Random Forest**: Ensemble learning
- **XGBoost**: Gradient boosting

#### Workflow Architecture:
```
               ┌─→ Logistic Regression
               │
Load Data ─────┼─→ H2O AutoML
               │
               ├─→ Random Forest
               │
               └─→ XGBoost
```

## Technical Components

### Core Technologies
- **Flyte/Flytekit**: Workflow orchestration engine
- **Domino Jobs**: Task execution environment
- **MLflow**: Experiment tracking and model registry
- **Python**: Task implementation language

### Task Definition

Tasks are defined using Domino's job execution pattern:
```python
task = run_domino_job_task(
    flyte_task_name='Task Name',
    command='python script.py',
    hardware_tier_name='Small|Medium|Large',
    inputs=[...],
    output_specs=[...],
    dataset_snapshots=[...]
)
```

### Input/Output Management

The system supports two execution modes:
1. **Workflow Mode**: Uses `/workflow/inputs/` and `/workflow/outputs/`
2. **Local Mode**: Uses command-line arguments and custom paths

Helper utilities in `flows.py` abstract this complexity:
```python
args = get_param_from_args(parser, 'param_name', workflow_input_dir)
handle_outputs({"output_name": value}, workflow_output_dir)
```

### Hardware Tiers

Tasks can specify compute requirements:
- **Small**: Light processing tasks
- **Medium**: Standard ML training
- **Large**: Heavy computation or large datasets

## MLflow Integration

The sklearn_log_reg_train.py showcases advanced ML practices:

### Features
- **Experiment Tracking**: Organized experiment hierarchy
- **Hyperparameter Tuning**: Grid search over C values
- **Metric Logging**: Comprehensive model evaluation
- **Artifact Storage**: Models, plots, and metadata
- **Model Registry**: Versioned model management

### Logged Metrics
- AUC (Area Under Curve)
- Log Loss
- F1 Score
- Precision
- Recall

## Usage

### Running Workflows

Execute workflows using pyflyte:
```bash
# Run sample flow
pyflyte run --remote flows/sample-flow.py my_workflow \
  --output_dir /path/to/output

# Run multi-train flow with parameters
pyflyte run --remote flows/multitrain-flow.py model_training_pipeline \
  --n_estimators 100 \
  --max_iter 1000 \
  --max_depth 5
```

### Local Development

For testing individual tasks:
```bash
# Run a single task locally
python flows/sample-flow/train-model.py \
  --data-path /path/to/data.csv \
  --output-dir /path/to/output
```

## Best Practices

### 1. Task Design
- Keep tasks focused and single-purpose
- Use clear input/output contracts
- Handle errors gracefully
- Log important information

### 2. Resource Management
- Choose appropriate hardware tiers
- Consider data size when setting resources
- Use dataset snapshots for large data

### 3. Workflow Organization
- Group related tasks in subdirectories
- Use descriptive task names
- Document dependencies clearly
- Version control workflow definitions

### 4. Data Handling
- Use consistent data formats
- Validate inputs early
- Clean up temporary files
- Consider data versioning

## Advanced Features

### Dataset Snapshots
Mount specific versions of Domino datasets:
```python
dataset_snapshots=[
    DatasetSnapshot(Name="dataset_name", Version="v1")
]
```

### Conditional Execution
Implement branching logic based on task outputs or parameters.

### Dynamic Workflows
Generate tasks programmatically based on configuration or data.

### Cross-Environment Execution
Run tasks in different compute environments as needed.

## Monitoring and Debugging

### Workflow Monitoring
- View execution status in Domino UI
- Track task duration and resource usage
- Monitor data lineage

### Debugging Tools
- Task-level logs
- Flyte console for workflow visualization
- MLflow UI for experiment tracking
- Domino job logs

## Integration Points

### Upstream
- Data sources (files, databases, APIs)
- Trained models from e02
- Feature engineering from e01

### Downstream
- Model deployment (e04)
- Model monitoring (e05)
- Business reporting systems

## Example Use Cases

1. **Daily Model Retraining**: Automated pipeline for regular model updates
2. **A/B Testing**: Parallel training of model variants
3. **Feature Engineering Pipeline**: Complex data transformation workflows
4. **Model Comparison**: Systematic evaluation of multiple algorithms
5. **Batch Prediction**: Large-scale inference pipelines

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are in environment
- **Path Issues**: Check workflow vs local execution modes
- **Resource Limits**: Adjust hardware tiers as needed
- **Data Access**: Verify dataset permissions

### Debug Steps
1. Check individual task logs
2. Verify input/output paths
3. Test tasks in isolation
4. Review Flyte execution graph
5. Monitor resource utilization