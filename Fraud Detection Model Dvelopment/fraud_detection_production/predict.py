
import joblib
import pandas as pd
import numpy as np
from feature_engineering import preprocess_for_prediction
import json

class FraudDetectionModel:
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl', metadata_path='metadata.json'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.threshold = self.metadata['business_metrics']['optimal_threshold']
        self.feature_names = self.metadata['feature_names']

    def predict(self, transactions_df):
        features_scaled, feature_names = preprocess_for_prediction(transactions_df, 'scaler.pkl')
        fraud_probabilities = self.model.predict_proba(features_scaled)[:, 1]
        fraud_flags = (fraud_probabilities >= self.threshold).astype(int)
        return {
            'fraud_probabilities': fraud_probabilities.tolist(),
            'fraud_flags': fraud_flags.tolist(),
            'threshold_used': self.threshold,
            'total_transactions': len(transactions_df),
            'flagged_transactions': int(fraud_flags.sum())
        }

    def predict_single(self, transaction_dict):
        df = pd.DataFrame([transaction_dict])
        return self.predict(df)

if __name__ == "__main__":
    fraud_model = FraudDetectionModel()
    sample_transaction = {
        'user_id': 12345,
        'merchant_id': 67890,
        'amount': 150.00,
        'timestamp': '2024-09-10 14:30:00'
    }
    result = fraud_model.predict_single(sample_transaction)
    print(f"Fraud probability: {result['fraud_probabilities'][0]:.3f}")
    print(f"Fraud flag: {'FRAUD' if result['fraud_flags'][0] else 'LEGITIMATE'}")
