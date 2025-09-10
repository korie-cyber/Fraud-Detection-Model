import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your feature engineering (make sure this file is in the correct path)
# from fraud_detection_production.feature_engineering import preprocess_for_prediction

# For now, I'll include the fixed feature engineering directly in this file
# You can move it back to your feature_engineering.py file

def engineer_features(df):
    """
    Your original feature engineering with production fixes
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
        st.warning("No timestamp column found. Using current datetime.")
        df_processed['timestamp'] = datetime.now()
        timestamp_col = 'timestamp'
    
    # Ensure we have the timestamp column name as 'timestamp' for consistency
    if timestamp_col != 'timestamp':
        df_processed['timestamp'] = df_processed[timestamp_col]
    
    # Handle missing required columns
    if 'user_id' not in df_processed.columns:
        st.error("Missing required column: user_id")
        st.stop()
    
    if 'amount' not in df_processed.columns:
        st.error("Missing required column: amount")
        st.stop()
    
    # Set default merchant_id if missing
    if 'merchant_id' not in df_processed.columns:
        st.warning("merchant_id column missing. Using default values.")
        df_processed['merchant_id'] = range(1000, 1000 + len(df_processed))

    # Transaction patterns
    try:
        df_processed['hour'] = pd.to_datetime(df_processed['timestamp']).dt.hour
        df_processed['is_weekend'] = (pd.to_datetime(df_processed['timestamp']).dt.weekday >= 5).astype(int)
        df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 6)).astype(int)
    except Exception as e:
        st.error(f"Error processing timestamp: {e}")
        st.stop()

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
        st.error(f"Error creating user features: {e}")
        st.stop()

    # Merchant patterns
    try:
        merchant_stats = df_processed.groupby('merchant_id')['amount'].agg([
            'count', 'mean'
        ]).reset_index()
        merchant_stats.columns = ['merchant_id', 'merchant_tx_count', 'merchant_avg_amount']
        df_processed = df_processed.merge(merchant_stats, on='merchant_id', how='left')
    except Exception as e:
        st.error(f"Error creating merchant features: {e}")
        st.stop()

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
        st.error(f"Error creating amount features: {e}")
        st.stop()

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
    try:
        # Load scaler
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        st.error(f"Scaler file not found at: {scaler_path}")
        st.stop()
    
    try:
        # Engineer features
        features = engineer_features(df)
        
        # Get expected feature names from scaler if available
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = scaler.feature_names_in_
            
            # Check for missing features
            missing_features = set(expected_features) - set(features.columns)
            if missing_features:
                st.warning(f"Missing features: {missing_features}. Adding with default values.")
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
        st.error(f"Error during preprocessing: {str(e)}")
        st.stop()

# Loading production artifacts
@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and metadata with caching for better performance"""
    MODEL_PATH = "fraud_detection_production/model.pkl"
    SCALER_PATH = "fraud_detection_production/scaler.pkl"
    METADATA_PATH = "fraud_detection_production/metadata.json"
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {e}")
        st.error("Please ensure all model files are in the 'fraud_detection_production' folder")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

def predict_transactions(df, model, scaler, metadata):
    """
    Make predictions on transaction data
    """
    threshold = metadata["business_metrics"]["optimal_threshold"]
    
    # Preprocess data
    features_scaled, feature_names = preprocess_for_prediction(df, "fraud_detection_production/scaler.pkl")
    
    # Make predictions
    fraud_probabilities = model.predict_proba(features_scaled)[:, 1]
    fraud_flags = (fraud_probabilities >= threshold).astype(int)
    
    # Prepare results
    df_results = df.copy()
    df_results["fraud_probability"] = fraud_probabilities
    df_results["fraud_flag"] = fraud_flags
    df_results["prediction"] = df_results["fraud_flag"].apply(lambda x: "FRAUD" if x == 1 else "LEGITIMATE")
    
    return df_results

# Streamlit UI
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("🚨 Fraud Detection Prediction App")
st.markdown("Use this app to detect potential fraudulent transactions using the trained model.")

# Load model artifacts
try:
    model, scaler, metadata = load_model_artifacts()
    threshold = metadata["business_metrics"]["optimal_threshold"]
except:
    st.stop()

# Sidebar info
st.sidebar.header("ℹ️ Model Info")
st.sidebar.write(f"**Model:** {metadata['model_info']['model_name']}")
st.sidebar.write(f"**Trained on:** {metadata['model_info']['training_date'][:10]}")
st.sidebar.write(f"**Optimal Threshold:** {threshold:.3f}")
st.sidebar.write(f"**Expected Daily Benefit:** ${metadata['business_metrics']['expected_daily_benefit']:,.2f}")

# Input methods
option = st.radio("Choose Input Method", ["Upload CSV", "Enter Single Transaction"])

if option == "Upload CSV":
    st.subheader("Upload CSV file with transactions")
    
    # Show requirements clearly
    st.info("**Required columns:** `user_id`, `amount`")
    st.info("**Optional columns:** `merchant_id`, `timestamp` (any datetime column)")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("📄 **Uploaded Data Preview:**")
            st.dataframe(input_df.head())
            
            st.write(f"**Data Shape:** {input_df.shape[0]} rows × {input_df.shape[1]} columns")
            st.write(f"**Columns:** {list(input_df.columns)}")
            
            # Validate required columns
            required_cols = ['user_id', 'amount']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.error("Please ensure your CSV has at minimum: user_id, amount")
            else:
                if st.button("🔍 Run Prediction"):
                    with st.spinner("Processing transactions..."):
                        results = predict_transactions(input_df, model, scaler, metadata)
                    
                    st.success(f"✅ Processed {len(results)} transactions")
                    
                    # Display results summary
                    fraud_count = results["fraud_flag"].sum()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🚩 Fraudulent Transactions", fraud_count)
                    with col2:
                        st.metric("📊 Total Transactions", len(results))
                    with col3:
                        fraud_rate = (fraud_count / len(results)) * 100
                        st.metric("📈 Fraud Rate", f"{fraud_rate:.1f}%")
                    
                    # Display detailed results
                    st.write("**Prediction Results:**")
                    display_cols = ['user_id', 'amount', 'fraud_probability', 'prediction']
                    if 'timestamp' in results.columns:
                        display_cols.insert(2, 'timestamp')
                    if 'merchant_id' in results.columns:
                        display_cols.insert(-2, 'merchant_id')
                    
                    available_display_cols = [col for col in display_cols if col in results.columns]
                    st.dataframe(results[available_display_cols].head(20))
                    
                    # Download results
                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Full Results as CSV", 
                        csv, 
                        "fraud_predictions.csv", 
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            st.error("Please check your CSV format and try again.")

elif option == "Enter Single Transaction":
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, value=12345)
        merchant_id = st.number_input("Merchant ID", min_value=1, value=67890)
        amount = st.number_input("Amount", min_value=0.01, value=150.00)
        
    with col2:
        timestamp = st.text_input(
            "Timestamp (YYYY-MM-DD HH:MM:SS)", 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    if st.button("🔍 Predict Fraud"):
        try:
            single_tx = {
                "user_id": user_id,
                "merchant_id": merchant_id,
                "amount": amount,
                "timestamp": timestamp
            }
            
            df_single = pd.DataFrame([single_tx])
            
            with st.spinner("Making prediction..."):
                result = predict_transactions(df_single, model, scaler, metadata)
            
            prob = result["fraud_probability"].iloc[0]
            flag = result["fraud_flag"].iloc[0]
            
            st.write("### 🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fraud Probability", f"{prob:.1%}")
            with col2:
                if flag:
                    st.error("🚨 **FRAUD DETECTED**")
                else:
                    st.success("✅ **LEGITIMATE TRANSACTION**")
            
            # Risk assessment
            if prob >= 0.8:
                st.error("⚠️ **HIGH RISK** - Immediate review recommended")
            elif prob >= 0.5:
                st.warning("⚡ **MEDIUM RISK** - Manual review suggested")
            else:
                st.info("✅ **LOW RISK** - Transaction appears normal")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("### 📋 Data Requirements")
st.info("""
**CSV Upload Requirements:**
- **Required:** `user_id` (numeric), `amount` (numeric)
- **Optional:** `merchant_id` (numeric), `timestamp` (datetime format)

**Supported timestamp formats:**
- YYYY-MM-DD HH:MM:SS
- YYYY-MM-DD
- MM/DD/YYYY HH:MM:SS

**Example CSV structure:**
```
user_id,merchant_id,amount,timestamp
12345,67890,150.00,2025-09-10 17:59:03
54321,98765,75.50,2025-09-10 18:30:22
```
""")

st.caption("🔒 Fraud Detection Model • Powered by XGBoost + Streamlit")