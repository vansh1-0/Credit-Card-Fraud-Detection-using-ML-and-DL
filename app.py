import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #2C2F3A;
        color: #F3F1FA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #9C8ADE !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #4A4F68;
    }
    
    [data-testid="stSidebar"] * {
        color: #F3F1FA !important;
    }
    
    /* Make Alert/Info boxes use the theme strictly */
    div[data-testid="stAlert"] {
        background-color: rgba(156, 138, 222, 0.15) !important;
        border: 1px solid #9C8ADE !important;
    }
    div[data-testid="stAlert"] * {
        color: #F3F1FA !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #9C8ADE;
        color: #2C2F3A;
        border-radius: 6px;
        border: 1px solid #C9CBD6;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #F3F1FA;
        color: #2C2F3A;
        border-color: #C9CBD6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(156, 138, 222, 0.4);
    }
    
    /* Cards for metrics */
    .metric-card {
        background: linear-gradient(145deg, #4A4F68, #2C2F3A);
        border: 1px solid #9C8ADE;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        text-align: center;
        transition: transform 0.3s ease;
        color: #F3F1FA;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #C9CBD6;
    }
    
    .fraud-alert {
        background: rgba(248, 81, 73, 0.1);
        border-left: 5px solid #f85149;
        padding: 20px;
        border-radius: 8px;
        color: #ff7b72;
    }
    
    .safe-alert {
        background: rgba(63, 185, 80, 0.1);
        border-left: 5px solid #3fb950;
        padding: 20px;
        border-radius: 8px;
        color: #56d364;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL CACHED ---
@st.cache_resource
def load_model():
    model_path = 'random_forest_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

@st.cache_data
def load_sample_data():
    data_path = 'creditcard.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path, nrows=5000) # Load subset for samples
    return None

sample_df = load_sample_data()

# --- HELPER FUNCTIONS ---
def preprocess_data(df):
    """Applies the exact transformations used during model training."""
    df_processed = df.copy()
    
    # Check if target 'Class' exists and drop it
    if 'Class' in df_processed.columns:
        df_processed = df_processed.drop('Class', axis=1)
        
    # Feature Engineering
    if 'Amount' in df_processed.columns:
        df_processed['Amount_log'] = np.log1p(df_processed['Amount'])
    else:
        df_processed['Amount_log'] = 0.0 # Fallback
        
    if 'Time' in df_processed.columns:
        df_processed['Time_hour'] = (df_processed['Time'] // 3600) % 24
        df_processed['Time_sin'] = np.sin(2 * np.pi * df_processed['Time'] / (24 * 3600))
        df_processed['Time_cos'] = np.cos(2 * np.pi * df_processed['Time'] / (24 * 3600))
    else:
        df_processed['Time_hour'] = 0.0
        df_processed['Time_sin'] = 0.0
        df_processed['Time_cos'] = 0.0
        
    # Drop original columns
    cols_to_drop = [col for col in ['Amount', 'Time'] if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(cols_to_drop, axis=1)
        
    # Ensure correct column order
    expected_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_log', 'Time_hour', 'Time_sin', 'Time_cos']
    
    # Fill missing columns with 0
    for col in expected_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0.0
            
    return df_processed[expected_cols]

def predict(data_df):
    X = preprocess_data(data_df).values
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    return preds, probs

# --- SIDEBAR UI ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2621/2621040.png", width=60)
st.sidebar.title("Guard configurations")

mode = st.sidebar.radio("Select Analysis Mode", 
                       ["Single Transaction", "Batch File Upload", "Sample Data Explorer"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("Model: **Random Forest**\n\nFeatures: **32 Engineered**\n\nStatus: **Active** ✅")

if not model:
    st.error("Model file 'random_forest_model.pkl' not found. Please ensure it is in the active directory.")
    st.stop()

# --- MAIN PAGE UI ---
st.markdown("<h1>🛡️ Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("### Advanced Machine Learning Dashboard")
st.markdown("Analyze transactions instantly using an advanced Random Forest algorithm trained on multi-dimensional patterns.")
st.markdown("---")

# 1. SINGLE TRANSACTION MODE
if mode == "Single Transaction":
    st.subheader("🔍 Single Transaction Inspector")
    
    # Initialize random values in session state if not present
    if "V1" not in st.session_state:
        st.session_state["V1"] = -1.2
        st.session_state["V2"] = 0.8
        st.session_state["V3"] = 1.5
        st.session_state["V4"] = 0.2
        for i in range(5, 29):
            st.session_state[f"V{i}"] = 0.0
            
    if st.button("🎲 Randomize PCA Values"):
        for i in range(1, 29):
            val = round(np.random.normal(0, 3.0), 2)
            # Clip to slider bounds for V1-V4
            if i <= 4:
                val = max(-10.0, min(10.0, val))
            st.session_state[f"V{i}"] = val
            
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_val = st.number_input("Time (seconds since start)", min_value=0.0, value=12000.0, step=100.0)
        amount_val = st.number_input("Amount ($)", min_value=0.0, value=150.0, step=10.0)
        
    with col2:
        st.slider("V1 Component (PCA)", -10.0, 10.0, key="V1")
        st.slider("V2 Component (PCA)", -10.0, 10.0, key="V2")
        
    with col3:
        st.slider("V3 Component (PCA)", -10.0, 10.0, key="V3")
        st.slider("V4 Component (PCA)", -10.0, 10.0, key="V4")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Show all 28 PCA Components"):
        grid_cols = st.columns(4)
        for i in range(5, 29):
            col_idx = (i - 1) % 4
            grid_cols[col_idx].number_input(f"V{i}", step=0.1, key=f"V{i}")

    if st.button("Analyze Transaction", use_container_width=True):
        with st.spinner("Analyzing high-dimensional space..."):
            time.sleep(1) # simulate deep analysis for user experience
            
            input_dict = {'Time': time_val, 'Amount': amount_val}
            for i in range(1, 29):
                input_dict[f'V{i}'] = st.session_state[f"V{i}"]
            
            input_df = pd.DataFrame([input_dict])
            pred, prob = predict(input_df)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if pred[0] == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h2>🚨 HIGH RISK TRANSACTION DETECTED</h2>
                    <p>Fraud Probability: <strong>{prob[0]*100:.1f}%</strong></p>
                    <p>This transaction matches historical markers of fraudulent activity. Recommended action: <b>BLOCK & INVESTIGATE</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    <h2>✅ TRANSACTION VERIFIED</h2>
                    <p>Fraud Probability: <strong>{prob[0]*100:.1f}%</strong></p>
                    <p>This transaction appears legitimate based on current threat intelligence.</p>
                </div>
                """, unsafe_allow_html=True)

# 2. BATCH UPLOAD MODE
elif mode == "Batch File Upload":
    st.subheader("📁 Batch Transaction Processing")
    st.markdown("Upload a CSV file containing transactions. Must contain 'Time', 'Amount', and 'V1'-'V28' columns.")
    
    uploaded_file = st.file_uploader("Drop your dataset here", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File loaded! Total transactions: {len(df)}")
            
            if st.button("Execute Batch Analysis", type="primary"):
                with st.spinner("Processing batch pipeline..."):
                    start_time = time.time()
                    preds, probs = predict(df)
                    df['Fraud_Prediction'] = preds
                    df['Risk_Score'] = probs
                    proc_time = time.time() - start_time
                    
                fraud_count = df['Fraud_Prediction'].sum()
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"<div class='metric-card'><h3>Records Scanned</h3><h1>{len(df):,}</h1></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='metric-card'><h3>Threats Found</h3><h1 style='color:#f85149;'>{fraud_count:,}</h1></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-card'><h3>Processing Time</h3><h1>{proc_time:.2f}s</h1></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Risk Analysis Table")
                
                fraud_df = df[df['Fraud_Prediction'] == 1].sort_values(by="Risk_Score", ascending=False)
                if len(fraud_df) > 0:
                    st.error(f"Displaying top {min(100, len(fraud_df))} highest risk transactions:")
                    st.dataframe(fraud_df.head(100), use_container_width=True)
                else:
                    st.success("No fraudulent transactions detected in this batch!")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# 3. SAMPLE DATA EXPLORER
elif mode == "Sample Data Explorer":
    st.subheader("🎲 Live Datastream Simulator")
    st.markdown("Pull random samples from the `creditcard.csv` dataset to see how the model behaves in real-time.")
    
    if sample_df is not None:
        sample_size = st.slider("Select number of sample transactions", 5, 200, 50)
        
        if st.button(f"Pull & Analyze {sample_size} Transactions"):
            with st.spinner("Extracting from live database..."):
                test_df = sample_df.sample(n=sample_size).copy()
                actual_fraud = test_df['Class'].sum() if 'Class' in test_df.columns else "Unknown"
                
                preds, probs = predict(test_df)
                test_df['Fraud_Prediction'] = preds
                test_df['Risk_Score'] = probs
                
                threats_detected = test_df['Fraud_Prediction'].sum()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Sample Size", f"{sample_size}")
            with c2:
                st.metric("Detected Threats", f"{threats_detected}", delta=f"{(threats_detected/sample_size)*100:.1f}% risk rate", delta_color="inverse")
            with c3:
                st.metric("Actual Threats (Ground Truth)", f"{actual_fraud}")
                
            st.markdown("---")
            
            # Show high risk
            high_risk = test_df[test_df['Fraud_Prediction'] == 1]
            if len(high_risk) > 0:
                st.error("Detected anomalies within context window:")
                display_cols = ['Time', 'Amount', 'Risk_Score'] + [col for col in test_df.columns if col.startswith('V')]
                st.dataframe(high_risk[display_cols].style.background_gradient(subset=['Risk_Score'], cmap='Reds'))
            else:
                st.success("No anomalies detected in the current window. System clear.")
    else:
        st.warning("Sample dataset `creditcard.csv` not found in the directory.")
    
st.markdown("<br><hr><center><small>Powered by Random Forest • Credit Card Fraud Detection System v1.0</small></center>", unsafe_allow_html=True)
