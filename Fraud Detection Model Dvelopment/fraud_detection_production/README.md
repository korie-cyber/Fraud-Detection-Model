# Fraud Detection Model - Production Deployment

## Overview
Trained fraud detection model using XGBoost with 29 engineered features.
- **Training Date**: 2025-09-10
- **Model Performance**: AUC-ROC 0.854
- **Optimal Threshold**: 0.900
- **Expected Daily Benefit**: $29.83

## Files Description
- `model.pkl`: Trained XGBoost model
- `scaler.pkl`: Feature preprocessing scaler
- `metadata.json`: Complete model metadata and performance metrics
- `feature_engineering.py`: Feature engineering pipeline
- `monitoring_config.json`: Performance monitoring configuration
- `predict.py`: Prediction script for new transactions

## Quick Start
```python
from predict import FraudDetectionModel
fraud_model = FraudDetectionModel()
results = fraud_model.predict(new_transactions_df)
print(f"Flagged {results['flagged_transactions']} out of {results['total_transactions']} transactions")

