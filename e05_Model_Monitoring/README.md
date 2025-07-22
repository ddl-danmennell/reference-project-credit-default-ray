# e05 Model Monitoring

This directory contains the model monitoring infrastructure for tracking model performance and data drift in production.

## Overview

The model monitoring system provides automated capabilities for tracking model predictions, registering ground truth data, and detecting data drift. It integrates with Domino Model Monitor (DMM) to provide enterprise-grade monitoring capabilities.

## Contents

### Python Scripts
- **DailyScoring.py**: Automated daily scoring pipeline with ground truth preparation
- **GT_registration.py**: Ground truth data registration with Domino Model Monitor

### Notebooks
- **CreateTrainingSet.ipynb**: Training set creation and versioning for baseline comparisons
- **Untitled.ipynb**: Additional monitoring experiments

### Monitoring Subdirectory
- **monitoring/**: Contains duplicate copies of core monitoring scripts

## Key Components

### 1. Daily Scoring Pipeline (DailyScoring.py)

Implements an automated scoring pipeline that:

#### Features
- **Synthetic Data Generation**: Uses SMOTEN to create balanced test datasets
- **Batch Scoring**: Processes 100-300 records per run
- **Customer ID Management**: Generates unique IDs with format `YYYY-MM-DD_index`
- **Model Integration**: Calls deployed model API with authentication
- **Ground Truth Storage**: Saves actual outcomes for later comparison

#### Workflow
1. Load sample data from training set
2. Generate synthetic scoring data using SMOTEN
3. Create batch of records with customer IDs
4. Score each record through model API
5. Prepare ground truth data with predictions
6. Upload to S3 bucket for centralized storage

#### Configuration
- Model API endpoint and authentication keys
- S3 bucket path: `s3://se-demo-data/<project_id>_<date>.csv`
- Feature subset: 11 most important features

### 2. Ground Truth Registration (GT_registration.py)

Automates the registration of ground truth data with Domino Model Monitor:

#### Features
- **Automated Registration**: Registers CSV files from S3 with DMM
- **API Integration**: Uses DMM v2 API endpoints
- **Dynamic Naming**: Uses project ID and current date
- **Error Handling**: Includes response validation

#### Process
1. Retrieve ground truth file path from S3
2. Create registration request with file metadata
3. Submit to DMM API for processing
4. Monitor registration status

#### Configuration
- DMM API endpoint: `/api/modelMonitor/v2/groundTruth`
- Requires DMM API key in environment
- CSV format with specific column structure

### 3. Training Set Creation (CreateTrainingSet.ipynb)

Manages baseline training sets for drift detection:

#### Features
- **Version Control**: Creates versioned training sets
- **Feature Selection**: Uses consistent 11-feature subset
- **Metadata Definition**: Configures column types for monitoring
- **Integration**: Works with Domino's TrainingSetClient

#### Training Set Structure
```python
Features:
- checking_account_A14
- credit_history_A34
- property_A121
- checking_account_A13
- other_installments_A143
- debtors_guarantors_A103
- savings_A65
- age
- employment_since_A73
- savings_A61
- customer_id

Target: credit (binary: 0=bad, 1=good)
```

## Monitoring Architecture

```
Daily Scoring → Ground Truth → S3 Storage → DMM Registration → Drift Detection
     ↓              ↓              ↓               ↓                ↓
Synthetic Data  Predictions   Cloud Storage   Model Monitor   Alerts/Reports
```

## Key Monitoring Metrics

### Data Quality
- Feature distribution changes
- Missing value patterns
- Outlier detection
- Data type consistency

### Model Performance
- Prediction accuracy vs ground truth
- Score distribution shifts
- Class imbalance changes
- Feature importance stability

### Operational Metrics
- Scoring latency
- API availability
- Data pipeline health
- Storage utilization

## Environment Requirements

### Python Dependencies
- pandas, numpy
- imblearn (for SMOTE)
- boto3 (S3 integration)
- requests (API calls)
- domino-data-capture

### Domino Configuration
- Model Monitor enabled
- API keys configured
- S3 access credentials
- Project environment variables

## Usage

### Running Daily Scoring
```bash
python DailyScoring.py
```

### Registering Ground Truth
```bash
python GT_registration.py
```

### Creating Training Sets
Run the notebook interactively to create versioned training sets.

## Best Practices

1. **Consistent Features**: Always use the same 11-feature subset
2. **Regular Scoring**: Run daily scoring at consistent times
3. **Ground Truth Lag**: Account for delay in obtaining actual outcomes
4. **Version Control**: Maintain training set versions for comparison
5. **Alert Thresholds**: Configure appropriate drift detection limits

## Integration Points

### Upstream Dependencies
- Trained model deployed as API
- S3 bucket for data storage
- Training data availability

### Downstream Consumers
- Domino Model Monitor dashboards
- Alerting systems
- Model retraining pipelines
- Business reporting tools

## Monitoring Workflow

1. **Baseline Establishment**
   - Create training set with known data distribution
   - Register with Model Monitor
   - Configure drift thresholds

2. **Continuous Monitoring**
   - Daily scoring generates new predictions
   - Ground truth collected when available
   - Automatic registration with DMM
   - Drift detection runs continuously

3. **Alert Response**
   - Investigate drift alerts
   - Analyze feature distributions
   - Determine if retraining needed
   - Update monitoring baselines

## Troubleshooting

### Common Issues
- **API Authentication**: Verify DMM API keys
- **S3 Access**: Check AWS credentials
- **Data Format**: Ensure CSV structure matches expectations
- **Missing Features**: Validate all 11 features present

### Debug Steps
1. Check API response codes
2. Validate data formats
3. Review S3 permissions
4. Examine DMM logs
5. Verify model endpoint status