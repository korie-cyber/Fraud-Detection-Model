import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """
    Original feature engineering with production-ready fixes
    """
    df_processed = df.copy()
    
    # Handle timestamp column - be flexible with column names
    timestamp_cols = ['timestamp', 'transaction_time', 'date', 'datetime']
    timestamp_col = None
    
    for col in timestamp_cols:
        if col in df_processed.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        # Create default timestamp if missing
        print("Warning: No timestamp column found. Using current datetime.")
        df_processed['timestamp'] = pd.Timestamp.now()
        timestamp_col = 'timestamp'
    
    # Ensure we have the timestamp column name as 'timestamp' for consistency
    if timestamp_col != 'timestamp':
        df_processed['timestamp'] = df_processed[timestamp_col]
    
    # Handle missing required columns
    if 'user_id' not in df_processed.columns:
        raise ValueError("Missing required column: user_id")
    
    if 'amount' not in df_processed.columns:
        raise ValueError("Missing required column: amount")
    
    # Set default merchant_id if missing
    if 'merchant_id' not in df_processed.columns:
        print("Warning: merchant_id column missing. Using default values.")
        df_processed['merchant_id'] = range(1000, 1000 + len(df_processed))

    # Transaction patterns
    try:
        df_processed['hour'] = pd.to_datetime(df_processed['timestamp']).dt.hour
        df_processed['is_weekend'] = (pd.to_datetime(df_processed['timestamp']).dt.weekday >= 5).astype(int)
        df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 6)).astype(int)
    except Exception as e:
        print(f"Error processing timestamp: {e}")
        # Use default values
        df_processed['hour'] = 12  # Default to noon
        df_processed['is_weekend'] = 0
        df_processed['is_night'] = 0

    # User behavioral features
    try:
        user_stats = df_processed.groupby('user_id')['amount'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        user_stats.columns = ['user_id', 'user_tx_count', 'user_avg_amount',
                             'user_amount_std', 'user_min_amount', 'user_max_amount']
        
        # Handle NaN in std (happens when user has only one transaction)
        user_stats['user_amount_std'] = user_stats['user_amount_std'].fillna(0)
        
        df_processed = df_processed.merge(user_stats, on='user_id', how='left')
    except Exception as e:
        print(f"Error creating user features: {e}")
        # Create default user features
        df_processed['user_tx_count'] = 1
        df_processed['user_avg_amount'] = df_processed['amount']
        df_processed['user_amount_std'] = 0
        df_processed['user_min_amount'] = df_processed['amount']
        df_processed['user_max_amount'] = df_processed['amount']

    # Merchant patterns
    try:
        merchant_stats = df_processed.groupby('merchant_id')['amount'].agg([
            'count', 'mean'
        ]).reset_index()
        merchant_stats.columns = ['merchant_id', 'merchant_tx_count', 'merchant_avg_amount']
        df_processed = df_processed.merge(merchant_stats, on='merchant_id', how='left')
    except Exception as e:
        print(f"Error creating merchant features: {e}")
        # Create default merchant features
        df_processed['merchant_tx_count'] = 1
        df_processed['merchant_avg_amount'] = df_processed['amount']

    # Amount-based features
    try:
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        
        # Handle z-score calculation for single transactions
        if len(df_processed) > 1:
            amount_mean = df_processed['amount'].mean()
            amount_std = df_processed['amount'].std()
            if amount_std == 0:  # All amounts are the same
                df_processed['amount_zscore'] = 0
            else:
                df_processed['amount_zscore'] = (df_processed['amount'] - amount_mean) / amount_std
        else:
            df_processed['amount_zscore'] = 0  # Default for single transaction
            
    except Exception as e:
        print(f"Error creating amount features: {e}")
        df_processed['amount_log'] = np.log1p(50.0)  # Default log value
        df_processed['amount_zscore'] = 0

    # Select only feature columns (exclude identifiers and target)
    exclude_columns = ['user_id', 'merchant_id', 'timestamp', 'is_fraud']
    feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
    
    # Fill any remaining NaN values
    feature_df = df_processed[feature_columns].fillna(0)
    
    return feature_df

def preprocess_for_prediction(df, scaler_path='scaler.pkl'):
    """
    Preprocess data for prediction with better error handling
    """
    import joblib
    
    try:
        # Load scaler
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    
    try:
        # Engineer features
        features = engineer_features(df)
        
        # Get expected feature names from scaler if available
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = scaler.feature_names_in_
            
            # Check for missing features
            missing_features = set(expected_features) - set(features.columns)
            if missing_features:
                print(f"Warning: Missing features {missing_features}. Adding with default values.")
                for feature in missing_features:
                    # Add intelligent defaults based on feature names
                    if 'amount' in feature:
                        features[feature] = features['amount'] if 'amount' in features.columns else 0
                    elif 'count' in feature:
                        features[feature] = 1
                    elif 'avg' in feature or 'mean' in feature:
                        features[feature] = features['amount'] if 'amount' in features.columns else 0
                    elif 'std' in feature:
                        features[feature] = 0
                    elif 'min' in feature or 'max' in feature:
                        features[feature] = features['amount'] if 'amount' in features.columns else 0
                    elif 'log' in feature:
                        features[feature] = 0
                    elif 'zscore' in feature:
                        features[feature] = 0
                    elif 'hour' in feature:
                        features[feature] = 12
                    else:
                        features[feature] = 0
            
            # Reorder columns to match expected features
            features = features.reindex(columns=expected_features, fill_value=0)
        
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        return features_scaled, features.columns.tolist()
        
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {str(e)}")

def validate_input_data(df):
    """
    Validate input data before processing
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Check for required columns
    required_cols = ['user_id', 'amount']
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}. Required: {required_cols}")
    
    # Validate data types
    try:
        df['user_id'] = pd.to_numeric(df['user_id'])
        df['amount'] = pd.to_numeric(df['amount'])
    except ValueError as e:
        raise ValueError(f"Error converting data types: {e}")
    
    # Check for negative amounts
    if (df['amount'] < 0).any():
        raise ValueError("Amount column contains negative values")
    
    return True

# Test function to verify the feature engineering works
def test_feature_engineering():
    """
    Test the feature engineering with sample data
    """
    print("Testing feature engineering...")
    
    # Test case 1: Single transaction
    df_single = pd.DataFrame({
        'user_id': [12345],
        'merchant_id': [67890],
        'amount': [150.0],
        'timestamp': ['2025-09-10 17:59:03']
    })
    
    print("Single transaction test:")
    try:
        features_single = engineer_features(df_single)
        print(f"✅ Single transaction - Features shape: {features_single.shape}")
        print(f"Features: {features_single.columns.tolist()}")
        print(f"Sample values: {features_single.iloc[0].to_dict()}")
    except Exception as e:
        print(f"❌ Single transaction failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test case 2: Multiple transactions
    df_multiple = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2],
        'merchant_id': [101, 102, 103, 101, 104],
        'amount': [100.0, 50.0, 200.0, 75.0, 125.0],
        'timestamp': [
            '2025-09-10 14:30:00', '2025-09-10 15:45:00', '2025-09-10 16:20:00',
            '2025-09-10 17:10:00', '2025-09-10 18:30:00'
        ]
    })
    
    print("Multiple transactions test:")
    try:
        features_multiple = engineer_features(df_multiple)
        print(f"✅ Multiple transactions - Features shape: {features_multiple.shape}")
        print(f"Features: {features_multiple.columns.tolist()}")
        print("Sample features for first transaction:")
        print(features_multiple.iloc[0].to_dict())
    except Exception as e:
        print(f"❌ Multiple transactions failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test case 3: Minimal data (missing optional columns)
    df_minimal = pd.DataFrame({
        'user_id': [999],
        'amount': [75.50]
        # No merchant_id, no timestamp
    })
    
    print("Minimal data test:")
    try:
        features_minimal = engineer_features(df_minimal)
        print(f"✅ Minimal data - Features shape: {features_minimal.shape}")
        print(f"Features: {features_minimal.columns.tolist()}")
        print(f"Sample values: {features_minimal.iloc[0].to_dict()}")
    except Exception as e:
        print(f"❌ Minimal data failed: {e}")

if __name__ == "__main__":
    test_feature_engineering()