# e02 Model Training

This directory contains the model training phase of the Credit Default Prediction project, featuring distributed training with Ray and XGBoost.

## Overview

The model training phase leverages distributed computing to handle large-scale data processing and hyperparameter optimization. It uses Ray for distributed computing and XGBoost as the primary machine learning algorithm, with optional MLflow integration for experiment tracking.

## Contents

### Python Scripts
- **train_model.py**: Core training script with Ray distributed training
- **train_model_mlflow.py**: Enhanced training script with MLflow experiment tracking

### Jupyter Notebooks
- **distributed-ray-training.ipynb**: Interactive notebook demonstrating distributed training
- **distributed-ray-training-mlflow.ipynb**: Notebook version with MLflow integration

## Key Technologies

### 1. **Ray Framework**
- **Ray Core**: Distributed computing for parallel data processing
- **Ray Tune**: Hyperparameter optimization at scale
- **XGBoost-Ray**: Distributed backend for XGBoost
- **On-Demand Clusters**: Dynamic Ray cluster provisioning

### 2. **Machine Learning**
- **XGBoost**: Gradient boosting framework for binary classification
- **Objective**: `binary:logistic` for credit risk prediction
- **Metrics**: Log loss and error rate

### 3. **Experiment Tracking (MLflow version)**
- Automatic MLflow server setup
- Hierarchical experiment structure
- Model registry integration
- Comprehensive artifact logging

## Training Configuration

### Ray Cluster Setup
- **Default Configuration**: 3 actors × 4 CPUs = 12 total CPUs
- **Hardware Tier**: Medium (4 cores, 15 GiB RAM) for head and worker nodes
- **Customizable**: Adjust `RAY_ACTORS` and `RAY_CPUS_PER_ACTOR` as needed

### Hyperparameter Search Space
- **eta** (learning rate): log-uniform [0.003, 0.3]
- **max_depth**: integer [2, 6]
- **tune_samples**: Number of trials (default: 10)

## Data Handling

### Distributed Data Loading
- **Training Data**: Multiple sharded CSV files (`train_data_*.csv`)
- **Validation Data**: Sharded validation files (`validation_data_*.csv`)
- **Test Data**: Single test file (`test_data.csv`)
- **RayDMatrix**: Lazy loading and automatic sharding across Ray cluster

### Data Paths
- Input: `/mnt/data/{PROJECT_NAME}/data/`
- Models: `/mnt/artifacts/`
- Ray results: `/mnt/data/{PROJECT_NAME}/ray_results/`

## Model Explainability

Both training approaches include comprehensive model interpretability features:

### XGBoost Native
- Feature importance by gain
- Feature importance by weight
- Feature importance by cover

### ELI5 Integration
- Global feature weights
- Individual prediction explanations
- HTML reports for model interpretation

## Usage

### Command Line Arguments
```bash
python train_model.py [--ray_actors N] [--cpus_per_actor N] [--tune_samples N]
```
- `--ray_actors`: Number of Ray actors (default: 3)
- `--cpus_per_actor`: CPUs per actor (default: 4)
- `--tune_samples`: Hyperparameter tuning trials (default: 10)

### Running as Domino Job
1. Attach Ray cluster to compute environment
2. Configure cluster with appropriate resources
3. Execute script with desired parameters

### Running in Workspace
1. Create workspace with Ray cluster attached
2. Open notebook version for interactive execution
3. Follow step-by-step instructions in notebook

## Output Artifacts

### Model Files
- `tune_best.xgb`: Best model from hyperparameter tuning
- Trial models in Ray results directory

### Visualizations
- Feature importance plots (PNG format)
- ELI5 HTML explanation reports

### Tracking Files
- `dominostats.json`: Domino-compatible metrics
- `trial_results.json`: Hyperparameter tuning summary
- MLflow artifacts (MLflow version only)

## Performance Considerations

1. **Data Sharding**: Ensure sufficient shards for Ray actors
2. **Memory Management**: Monitor cluster memory usage with large datasets
3. **Tuning Samples**: Balance between exploration and training time
4. **Actor Configuration**: Match actors to available cluster resources

## Best Practices

1. Start with small `tune_samples` for testing
2. Monitor Ray dashboard for resource utilization
3. Use validation data for early stopping
4. Save intermediate results for fault tolerance
5. Review feature importance for model insights

## Troubleshooting

Common issues and solutions:
- **Ray Connection**: Check `RAY_HEAD_SERVICE_HOST` and `RAY_HEAD_SERVICE_PORT`
- **Memory Errors**: Reduce actors or increase cluster resources
- **Shard Warnings**: Ensure data files ≥ number of actors
- **MLflow Issues**: Verify MLflow server accessibility

## Next Steps

After training completes:
1. Review model performance metrics
2. Analyze feature importance plots
3. Deploy best model via Model API (see e04)
4. Set up monitoring pipeline (see e05)