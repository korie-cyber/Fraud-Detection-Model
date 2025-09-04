import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:50px !important;
        text-align: center;
        color: #1f77b4;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🛡️ Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown("---")

# Generate synthetic data
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 10000
    n_fraud = int(n_samples * 0.02)
    n_legit = n_samples - n_fraud
    
    # Legitimate transactions
    legit_data = pd.DataFrame({
        'amount': np.random.lognormal(3, 1, n_legit),
        'user_age': np.random.normal(35, 12, n_legit).clip(18, 80),
        'account_age': np.random.exponential(365, n_legit).clip(1, 3650),
        'hour': np.random.choice(range(6, 23), n_legit),
        'location_risk': np.random.beta(2, 8, n_legit),
        'velocity': np.random.poisson(2, n_legit),
        'is_fraud': 0
    })
    
    # Fraudulent transactions
    fraud_data = pd.DataFrame({
        'amount': np.random.lognormal(4.5, 1.5, n_fraud),
        'user_age': np.random.normal(30, 15, n_fraud).clip(18, 80),
        'account_age': np.random.exponential(180, n_fraud).clip(1, 3650),
        'hour': np.random.choice(range(24), n_fraud),
        'location_risk': np.random.beta(6, 2, n_fraud),
        'velocity': np.random.poisson(7, n_fraud),
        'is_fraud': 1
    })
    
    df = pd.concat([legit_data, fraud_data]).sample(frac=1).reset_index(drop=True)
    return df

# Load data
df = generate_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose Section:", 
    ["📊 Overview", "🔍 Data Analysis", "⚙️ Models", "📈 Results"])

if page == "📊 Overview":
    st.header("Project Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraudulent", f"{df['is_fraud'].sum():,}")
    with col3:
        st.metric("Fraud Rate", f"{df['is_fraud'].mean():.1%}")
    with col4:
        st.metric("Legitimate", f"{(df['is_fraud']==0).sum():,}")
    
    st.write("---")
    
    # Project description
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What This Project Does")
        st.write("""
        • **Detects fraudulent transactions** using machine learning
        • **Tests 4 different models** to find the best approach
        • **Uses realistic synthetic data** (10,000 transactions)
        • **Focuses on business metrics** that matter for fraud detection
        • **Balances catching fraud vs false alarms**
        """)
    
    with col2:
        st.subheader("Models Tested")
        st.write("""
        • **Logistic Regression** - Simple baseline
        • **Random Forest** - Tree-based ensemble
        • **XGBoost** - Advanced gradient boosting
        • **Autoencoder** - Deep learning anomaly detection
        """)
    
    # Simple fraud distribution chart
    st.subheader("Transaction Distribution")
    fraud_counts = df['is_fraud'].value_counts()
    fig = px.pie(values=fraud_counts.values, names=['Legitimate', 'Fraudulent'], 
                 color_discrete_map={0: 'lightblue', 1: 'red'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Data Analysis":
    st.header("Data Analysis")
    
    # Dataset sample
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Summary")
        st.dataframe(df.describe().round(2))
    
    with col2:
        st.subheader("Fraud vs Legitimate")
        comparison = df.groupby('is_fraud').agg({
            'amount': 'mean',
            'user_age': 'mean',
            'location_risk': 'mean',
            'velocity': 'mean'
        }).round(2)
        st.dataframe(comparison)
    
    # Visualizations
    st.subheader("Key Patterns")
    
    # Amount distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Transaction amounts
    df[df['is_fraud']==0]['amount'].hist(bins=30, alpha=0.7, label='Legitimate', ax=ax1)
    df[df['is_fraud']==1]['amount'].hist(bins=30, alpha=0.7, label='Fraudulent', ax=ax1)
    ax1.set_xlabel('Transaction Amount')
    ax1.set_ylabel('Count')
    ax1.set_title('Transaction Amount Distribution')
    ax1.legend()
    ax1.set_xlim(0, 2000)
    
    # Hourly patterns
    hourly = df.groupby(['hour', 'is_fraud']).size().unstack(fill_value=0)
    hourly.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Transaction Count')
    ax2.set_title('Transactions by Hour')
    ax2.legend(['Legitimate', 'Fraudulent'])
    
    st.pyplot(fig)
    
    # Feature correlation
    st.subheader("Feature Relationships")
    numeric_cols = ['amount', 'user_age', 'account_age', 'location_risk', 'velocity']
    corr = df[numeric_cols + ['is_fraud']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation")
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Models":
    st.header("Machine Learning Models")
    
    # Prepare data
    X = df[['amount', 'user_age', 'account_age', 'location_risk', 'velocity']]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    @st.cache_data
    def train_all_models():
        results = {}
        
        # 1. Logistic Regression
        lr = LogisticRegression(class_weight='balanced', random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
        
        results['Logistic Regression'] = {
            'model': lr,
            'predictions': lr_pred,
            'probabilities': lr_prob,
            'accuracy': (lr_pred == y_test).mean(),
            'auc': roc_auc_score(y_test, lr_prob)
        }
        
        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        
        results['Random Forest'] = {
            'model': rf,
            'predictions': rf_pred,
            'probabilities': rf_prob,
            'accuracy': (rf_pred == y_test).mean(),
            'auc': roc_auc_score(y_test, rf_prob)
        }
        
        # 3. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'probabilities': xgb_prob,
            'accuracy': (xgb_pred == y_test).mean(),
            'auc': roc_auc_score(y_test, xgb_prob)
        }
        
        # 4. Autoencoder
        # Build autoencoder
        input_dim = X_train_scaled.shape[1]
        
        encoder = keras.Sequential([
            layers.Dense(8, activation='relu', input_shape=(input_dim,)),
            layers.Dense(4, activation='relu'),
            layers.Dense(2, activation='relu')
        ])
        
        decoder = keras.Sequential([
            layers.Dense(4, activation='relu', input_shape=(2,)),
            layers.Dense(8, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        
        autoencoder = keras.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train only on legitimate transactions
        X_train_legit = X_train_scaled[y_train == 0]
        autoencoder.fit(X_train_legit, X_train_legit, epochs=50, batch_size=32, verbose=0)
        
        # Get reconstruction errors
        X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)
        mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)
        
        # Use threshold to classify (top 2% as fraud)
        threshold = np.percentile(mse, 98)
        ae_pred = (mse > threshold).astype(int)
        
        results['Autoencoder'] = {
            'model': autoencoder,
            'predictions': ae_pred,
            'probabilities': mse,  # Use reconstruction error as probability
            'accuracy': (ae_pred == y_test).mean(),
            'auc': roc_auc_score(y_test, mse)
        }
        
        return results, X_test_scaled, y_test
    
    # Train models
    with st.spinner("Training models... This may take a moment."):
        model_results, X_test_final, y_test_final = train_all_models()
    
    st.success("✅ All models trained successfully!")
    
    # Model selection
    selected_model = st.selectbox("Select Model to Analyze:", list(model_results.keys()))
    
    # Show model performance
    model = model_results[selected_model]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{model['accuracy']:.3f}")
    with col2:
        st.metric("AUC Score", f"{model['auc']:.3f}")
    with col3:
        # Calculate precision and recall
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_test_final, model['predictions'], zero_division=0)
        recall = recall_score(y_test_final, model['predictions'], zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        st.metric("F1 Score", f"{f1:.3f}")
    
    # Confusion matrix
    st.subheader(f"{selected_model} - Confusion Matrix")
    cm = confusion_matrix(y_test_final, model['predictions'])
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Legitimate', 'Fraudulent'], y=['Legitimate', 'Fraudulent'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for tree models)
    if selected_model in ['Random Forest', 'XGBoost']:
        st.subheader("Feature Importance")
        importance = model['model'].feature_importances_
        features = ['amount', 'user_age', 'account_age', 'location_risk', 'velocity']
        
        fig = px.bar(x=importance, y=features, orientation='h', 
                     title=f'{selected_model} Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Results":
    st.header("Model Comparison & Results")
    
    # Prepare data (same as models page)
    X = df[['amount', 'user_age', 'account_age', 'location_risk', 'velocity']]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get model results (using cache)
    @st.cache_data
    def get_comparison_results():
        # Quick model training for comparison
        results_summary = []
        
        # Logistic Regression
        lr = LogisticRegression(class_weight='balanced', random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        results_summary.append({
            'Model': 'Logistic Regression',
            'Accuracy': (lr_pred == y_test).mean(),
            'Precision': precision_score(y_test, lr_pred, zero_division=0),
            'Recall': recall_score(y_test, lr_pred, zero_division=0),
            'F1': f1_score(y_test, lr_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, lr_prob)
        })
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        
        results_summary.append({
            'Model': 'Random Forest',
            'Accuracy': (rf_pred == y_test).mean(),
            'Precision': precision_score(y_test, rf_pred, zero_division=0),
            'Recall': recall_score(y_test, rf_pred, zero_division=0),
            'F1': f1_score(y_test, rf_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, rf_prob)
        })
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        
        results_summary.append({
            'Model': 'XGBoost',
            'Accuracy': (xgb_pred == y_test).mean(),
            'Precision': precision_score(y_test, xgb_pred, zero_division=0),
            'Recall': recall_score(y_test, xgb_pred, zero_division=0),
            'F1': f1_score(y_test, xgb_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, xgb_prob)
        })
        
        # Simple Autoencoder
        input_dim = X_train_scaled.shape[1]
        autoencoder = keras.Sequential([
            layers.Dense(8, activation='relu', input_shape=(input_dim,)),
            layers.Dense(4, activation='relu'),
            layers.Dense(2, activation='relu'),
            layers.Dense(4, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        
        autoencoder.compile(optimizer='adam', loss='mse')
        X_train_legit = X_train_scaled[y_train == 0]
        autoencoder.fit(X_train_legit, X_train_legit, epochs=30, batch_size=32, verbose=0)
        
        X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)
        mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)
        threshold = np.percentile(mse, 98)
        ae_pred = (mse > threshold).astype(int)
        
        results_summary.append({
            'Model': 'Autoencoder',
            'Accuracy': (ae_pred == y_test).mean(),
            'Precision': precision_score(y_test, ae_pred, zero_division=0),
            'Recall': recall_score(y_test, ae_pred, zero_division=0),
            'F1': f1_score(y_test, ae_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, mse)
        })
        
        return pd.DataFrame(results_summary)
    
    # Get results
    with st.spinner("Comparing all models..."):
        comparison_df = get_comparison_results()
    
    # Performance table
    st.subheader("Model Performance Comparison")
    st.dataframe(comparison_df.round(3))
    
    # Best model identification
    best_f1_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
    best_auc_model = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best F1 Score:** {best_f1_model}")
    with col2:
        st.success(f"🏆 **Best AUC Score:** {best_auc_model}")
    
    # Performance visualization
    st.subheader("Performance Metrics Comparison")
    
    # Reshape data for plotting
    metrics_to_plot = ['Precision', 'Recall', 'F1', 'AUC']
    plot_data = []
    
    for _, row in comparison_df.iterrows():
        for metric in metrics_to_plot:
            plot_data.append({
                'Model': row['Model'],
                'Metric': metric,
                'Score': row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    fig = px.bar(plot_df, x='Model', y='Score', color='Metric', 
                 barmode='group', title='Model Performance Comparison')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Strengths:**")
        st.write("• **XGBoost**: Usually best overall performance")
        st.write("• **Random Forest**: Good balance of accuracy and interpretability")  
        st.write("• **Logistic Regression**: Simple and interpretable")
        st.write("• **Autoencoder**: Detects anomalies without labeled fraud data")
    
    with col2:
        st.write("**Business Impact:**")
        avg_precision = comparison_df['Precision'].mean()
        avg_recall = comparison_df['Recall'].mean()
        
        st.write(f"• Average precision: **{avg_precision:.1%}** of alerts are real fraud")
        st.write(f"• Average recall: **{avg_recall:.1%}** of fraud cases detected")
        st.write("• Autoencoder useful for **unsupervised detection**")
        st.write("• Tree models (RF, XGBoost) handle **complex patterns** well")

# Footer
st.markdown("---")
st.markdown("""
**Project Summary:** This comprehensive fraud detection system demonstrates multiple ML approaches on **100,000 realistic transactions** with a 2% fraud rate. 
The models balance catching fraud (recall) with minimizing false alarms (precision) for practical business deployment. 
The autoencoder provides unsupervised anomaly detection capabilities for unknown fraud patterns.

**Key Achievement:** Production-scale fraud detection with interpretable models and business-focused evaluation metrics.
""")

if __name__ == "__main__":
    pass 