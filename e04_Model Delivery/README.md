# e04 Model Delivery

This directory contains the model deployment and delivery components of the Credit Default Prediction project.

## Overview

The model delivery phase provides two main deployment options:
1. A REST API endpoint for programmatic model scoring
2. An interactive Streamlit web application for manual credit assessment

Both implementations use the trained XGBoost model and provide explainable predictions with feature importance insights.

## Contents

- **score.py**: Model API scoring function for REST endpoint deployment
- **streamlit_app.py**: Interactive web application for credit risk assessment

## Model API (score.py)

### Features
- **Model Loading**: Uses pre-trained XGBoost model from `/mnt/artifacts/tune_best.xgb`
- **Simplified Interface**: Accepts only the 10 most important features
- **Data Capture**: Integrates with Domino Data Capture Client for monitoring
- **Customer ID Tracking**: Generates UUID if customer ID not provided
- **Binary Classification**: Returns both probability score and class prediction (0.5 threshold)

### API Function
```python
predict_credit(checking_account_A14, credit_history_A34, property_A121, 
               checking_account_A13, other_installments_A143, 
               debtors_guarantors_A103, savings_A65, age, 
               employment_since_A73, savings_A61, customer_id=None)
```

### Input Features
The API uses the top 10 most significant features identified during model training:
- `checking_account_A14`: Has checking account (binary)
- `credit_history_A34`: Full credit history available (binary)
- `property_A121`: Property ownership (binary)
- `checking_account_A13`: Checking balance > 1000 (binary)
- `other_installments_A143`: Has other installments (binary)
- `debtors_guarantors_A103`: Has guarantors (binary)
- `savings_A65`: Has savings account (binary)
- `age`: Applicant age (scaled 0-1)
- `employment_since_A73`: Employed > 5 years (binary)
- `savings_A61`: Savings > 1000 (binary)

### Sample API Request
```json
{
  "data": {
    "checking_account_A14": 0,
    "credit_history_A34": 0,
    "property_A121": 0,
    "checking_account_A13": 0,
    "other_installments_A143": 1,
    "debtors_guarantors_A103": 0,
    "savings_A65": 0,
    "age": 0.285714,
    "employment_since_A73": 1,
    "savings_A61": 1
  }
}
```

### Response Format
```json
{
  "score": 0.7234,  // Probability of good credit
  "class": 1        // Binary prediction (1=good, 0=bad)
}
```

## Streamlit Application (streamlit_app.py)

### Features
- **Interactive UI**: User-friendly web interface for credit assessment
- **Real-time Scoring**: Immediate risk assessment with visual feedback
- **Risk Categories**:
  - Green (APPROVED): Probability ≥ 0.6
  - Yellow (MANUAL REVIEW): 0.4 ≤ Probability < 0.6
  - Red (DENIED): Probability < 0.4
- **Visualizations**:
  - Gauge chart showing repayment probability
  - Feature importance tables with color gradients
  - Model weights and prediction explanations
- **Explainability**: ELI5 integration for transparent decision insights

### User Interface Components
1. **Sidebar Input Form**:
   - Checkbox inputs for binary features
   - Number input for age (20-115 years)
   - Score button to trigger assessment

2. **Main Display**:
   - Application status (Approved/Review/Denied)
   - Probability gauge visualization
   - Feature importance tables

3. **Model Insights**:
   - Global feature weights
   - Individual prediction contributions
   - Feature values impact visualization

### Technical Details
- **Model Integration**: Loads XGBoost model directly
- **Age Normalization**: Scales age to 0-1 range
- **Default Values**: Uses baseline feature vector for missing inputs
- **API Integration**: Makes REST calls to deployed model endpoint
- **Error Handling**: Graceful fallback for ELI5 exceptions

## Deployment Options

### 1. Model API Deployment
- Deploy as Domino Model API
- Supports high-volume scoring
- RESTful interface
- Authentication via API keys
- Automatic scaling capabilities

### 2. Streamlit App Deployment
- Deploy as Domino App
- Interactive user interface
- Suitable for manual review cases
- Real-time explainability
- No coding required for users

## Data Monitoring

Both implementations include data capture for monitoring:
- Feature values logging
- Prediction tracking
- Customer ID association
- Event-based capture

## Requirements

### Python Dependencies
- xgboost
- pandas
- numpy
- streamlit (for web app)
- plotly (for visualizations)
- eli5 (for explainability)
- requests (for API calls)
- domino-data-capture

### Model Artifacts
- Pre-trained model: `/mnt/artifacts/tune_best.xgb`
- Model must be trained with matching feature set

## Usage

### Running the Model API
```bash
python score.py  # For testing
# Deploy via Domino Model API interface for production
```

### Running Streamlit App
```bash
streamlit run streamlit_app.py
# Or deploy as Domino App
```

## Best Practices

1. **Feature Validation**: Ensure input features match training data distribution
2. **Monitoring**: Track prediction drift and feature distributions
3. **Explainability**: Always provide model reasoning for credit decisions
4. **Compliance**: Maintain audit trail via data capture
5. **Performance**: Monitor API response times and throughput

## Integration Notes

- The model expects normalized age values (0-1 range)
- Binary features should be 0 or 1
- Missing features use default values from training data
- Consider implementing feature validation for production use