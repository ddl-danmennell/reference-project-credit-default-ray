# e01 Data Exploration

This directory contains exploratory data analysis (EDA) notebooks for understanding and analyzing datasets used in the project.

## Overview

The data exploration phase is crucial for understanding data characteristics, identifying patterns, detecting anomalies, and informing modeling decisions. This directory contains two notebooks that demonstrate comprehensive EDA techniques.

## Contents

### 1. **data_Exploration.ipynb** - Credit Risk Dataset Analysis
The primary EDA notebook for the credit risk modeling project. This notebook performs extensive analysis on the processed credit risk dataset.

#### Key Features:
- **Data Quality Assessment**: Checks for missing values and data types
- **Feature Analysis**: 
  - Analyzes 62 numerical features (including binary-encoded categorical variables)
  - Categorizes features into continuous, binary, and discrete types
  - Examines feature distributions and skewness
- **Advanced Visualizations**:
  - Feature constellation maps
  - Correlation force fields
  - Feature spiral galaxies
  - Multi-dimensional pattern analyses
- **Statistical Insights**:
  - Identifies highly correlated feature pairs
  - Detects outliers (6.6% of data)
  - Analyzes class imbalance
- **Modeling Recommendations**: Provides guidance on preprocessing, feature engineering, and model selection

### 2. **EDA_code.ipynb** - Wine Quality Analysis Demo
A demonstration notebook showcasing EDA techniques on a wine quality dataset.

#### Key Features:
- **Data Access**: Demonstrates Domino DataSourceClient usage
- **Correlation Analysis**: Creates heatmaps for feature relationships
- **Feature Importance**: Identifies features most correlated with wine quality
- **Distribution Analysis**: Visualizes feature distributions with histograms and KDE
- **Data Export**: Shows how to save processed data

## Key Findings from Credit Risk EDA

1. **Data Quality**: No missing values found in the dataset
2. **Feature Types**:
   - 7 continuous features (scaled 0-1)
   - 54 binary features (one-hot encoded categoricals)
   - 1 discrete feature
3. **Class Distribution**: Dataset shows class imbalance requiring attention
4. **Correlations**: 5 feature pairs show high correlation (>0.8)
5. **Outliers**: 6.6% of observations identified as outliers

## Visualization Techniques

The notebooks demonstrate various advanced visualization techniques:
- Correlation heatmaps
- Distribution plots with KDE overlays
- 3D scatter plots
- Radial/spiral visualizations
- Force-directed graphs
- Constellation maps

## Requirements

- Python packages: pandas, numpy, matplotlib, seaborn, plotly, scipy
- Access to Domino datasets
- Jupyter notebook environment

## Usage

1. Run `data_Exploration.ipynb` for comprehensive analysis of the credit risk dataset
2. Reference `EDA_code.ipynb` for examples of EDA techniques on different data types

## Recommendations for Next Steps

Based on the EDA findings:
1. Apply appropriate preprocessing techniques for skewed features
2. Consider feature engineering for highly correlated variables
3. Implement strategies to handle class imbalance
4. Use robust scaling methods for outlier-prone features
5. Select models that can handle mixed data types effectively

## Output

The notebooks generate various plots and statistical summaries that inform the modeling phase. Key visualizations are displayed inline within the notebooks.