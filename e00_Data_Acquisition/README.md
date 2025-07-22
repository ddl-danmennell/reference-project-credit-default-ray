# e00 Data Acquisition

This directory contains the data acquisition and preparation phase of the Credit Default Prediction project.

## Overview

The data acquisition phase focuses on preparing the training and test datasets from the Statlog (German Credit Data) dataset. This includes data balancing, upsampling, and generation of synthetic data to create a more challenging machine learning problem.

## Contents

- **data_generation.ipynb**: Main notebook for data generation and preprocessing

## Data Source

The project uses the [Statlog (German Credit Data) Data Set](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)), which classifies customers based on a set of attributes into two credit risk groups - good or bad.

## Data Processing Steps

1. **Data Loading**: Downloads the raw German credit dataset if not already present
2. **Feature Engineering**: 
   - Adds human-readable column names
   - Remaps target variable (1 = good credit, 0 = bad credit)
   - Creates indicator variables for categorical attributes
   - Scales numerical features to (0,1) range using MinMaxScaler
3. **Train/Test Split**: Splits data using 80:20 ratio
4. **Data Balancing**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the imbalanced dataset
5. **Data Generation**: Generates multiple chunks of balanced data to create:
   - Training set: 2.1 million samples (21 chunks × 100,000 samples)
   - Validation set: 300,000 samples (3 chunks × 100,000 samples)

## Feature Categories

### Numerical Features
- duration
- credit_amount
- installment_rate
- residence
- age
- credits
- dependents

### Categorical Features
- checking_account
- credit_history
- purpose
- savings
- employment_since
- status
- debtors_guarantors
- property
- other_installments
- housing
- job
- telephone
- foreign_worker

## Requirements

- Python packages: pandas, numpy, urllib, imblearn, scikit-learn
- Domino Datasets for data storage

## Usage

The notebook is designed to be run once to generate the datasets. The generated data is stored in the Domino Dataset path at `/mnt/data/{PROJECT_NAME}/data/`.

## Output Files

- `test_data.csv`: Test dataset (20% of original data)
- `train_data_[0-20].csv`: Training data chunks
- `validation_data_[0-2].csv`: Validation data chunks

## Notes

- The notebook uses a fixed random seed (1234) for reproducibility
- Feature scaling is applied globally, which introduces minor information leakage but is acceptable for this demo
- The original class imbalance (546 good credit vs 235 bad credit) is addressed through SMOTE balancing