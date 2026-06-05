import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import google.generativeai as genai
import datetime 

# --- 1. PERSISTENCE HELPERS (PHYSICALLY SEPARATED) ---
MANUAL_FILE = "manual_watchlist.csv"
BULK_CACHE_FILE = "bulk_history.csv"

def load_data(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Ensure proper naming for UI/Logic consistency
            if 'Amount' in df.columns and 'Invoice_Amount' not in df.columns:
                df = df.rename(columns={'Amount': 'Invoice_Amount'})
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

# --- 2. ASSET LOADING & AI SETUP ---
@st.cache_resource
def load_models():
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_risk = joblib.load('label_encoder.pkl')
    return model, scaler, le_risk

genai.configure(api_key="AIzaSyBDAHDJI8TFfGsgvC3JHZfu8Aj07lXUyH0")
ai_brain = genai.GenerativeModel('models/gemini-1.5-flash-preview')

def get_ai_response(user_query, context_data):
    prompt = f"You are 'Growfin-Bot'. Context: {context_data}\nQuestion: {user_query}"
    try:
        response = ai_brain.generate_content(prompt)
        return response.text
    except:
        return "AI is currently unavailable."

# --- 3. LOGIC HELPERS ---
def apply_business_rules(risk_label, delay, dispute):
    if delay > 20 or dispute == 1: return 'High Risk'
    return risk_label

def process_batch(df_input):
    required_cols = ['Invoice_Amount', 'Payment_Method', 'Dispute', 'Avg_Past_Delay']
    method_map = {"ACH": 0, "Check": 1, "Credit Card": 2, "Wire": 3}
    missing = [col for col in required_cols if col not in df_input.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return None
    
    df_model = df_input[required_cols].copy()
    if df_model['Dispute'].dtype == object:
        df_model['Dispute'] = df_model['Dispute'].str.strip().str.capitalize().map({"Yes": 1, "No": 0}).fillna(0)
    else:
        df_model['Dispute'] = pd.to_numeric(df_model['Dispute'], errors='coerce').fillna(0).astype(int)
    
    df_model['Payment_Method'] = df_model['Payment_Method'].map(method_map).fillna(0)
    
    try:
        model, scaler, le_risk = load_models()
        X_scaled = scaler.transform(df_model.astype(float))
        preds = model.predict(X_scaled)
        df_input['Risk_Level'] = le_risk.inverse_transform(preds)
        df_input['Date'] = datetime.date.today()
        
        df_input['Risk_Level'] = df_input.apply(
            lambda x: apply_business_rules(
                x['Risk_Level'], x['Avg_Past_Delay'], 
                1 if str(x['Dispute']).strip().capitalize() in ["Yes", "1"] else 0
            ), axis=1
        )
        return df_input
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# --- 4. UI INITIALIZATION ---
st.set_page_config(page_title="AI Cash Flow Risk Detector", layout="wide")
st.title("🛡️ AI Cash Flow Risk Detector")

# Session State for Chats
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'bulk_chat_history' not in st.session_state: st.session_state.bulk_chat_history = []

# --- 5. SIDEBAR (MANUAL ENTRY) ---
st.sidebar.header("Individual Invoice Details")
cust_id = st.sidebar.text_input("Customer ID", value="CUST-001")
invoice_amt = st.sidebar.number_input("Invoice Amount ($)", min_value=0, value=5000)
payment_method = st.sidebar.selectbox("Payment Method", ["ACH", "Check", "Credit Card", "Wire"])
dispute_radio = st.sidebar.radio("Active Dispute?", ["No", "Yes"])
past_delay = st.sidebar.slider("Average Past Delays (Days)", 0, 60, 5)

if st.sidebar.button("Analyze & Add to Watchlist"):
    dis_val = 1 if dispute_radio == "Yes" else 0
    m_map = {"ACH": 0, "Check": 1, "Credit Card": 2, "Wire": 3}
    model, scaler, le_risk = load_models()
    feat = np.array([[invoice_amt, m_map[payment_method], dis_val, past_delay]])
    scaled = scaler.transform(feat)
    pred = le_risk.inverse_transform(model.predict(scaled))[0]
    final_risk = apply_business_rules(pred, past_delay, dis_val)
    
    new_entry = pd.DataFrame([{
        'Date': datetime.date.today(),
        'Customer_ID': cust_id, 'Invoice_Amount': invoice_amt, 
        'Payment_Method': payment_method, 'Dispute': dispute_radio, 
        'Avg_Past_Delay': past_delay, 'Risk_Level': final_risk
    }])
    
    old_manual = load_data(MANUAL_FILE)
    save_data(pd.concat([old_manual, new_entry], ignore_index=True), MANUAL_FILE)
    st.rerun()

if st.sidebar.button("Clear All Saved Data"):
    if os.path.exists(MANUAL_FILE): os.remove(MANUAL_FILE)
    if os.path.exists(BULK_CACHE_FILE): os.remove(BULK_CACHE_FILE)
    st.session_state.chat_history = []
    st.session_state.bulk_chat_history = []
    st.rerun()

tab1, tab2 = st.tabs(["Live Monitoring Watchlist", "Bulk Upload (CSV)"])

# --- TAB 1: INDIVIDUAL WATCHLIST ---
with tab1:
    w_df = load_data(MANUAL_FILE)
    if not w_df.empty:
        st.subheader("📈 Daily Exposure Trend (Manual Invoices)")
        daily_trend = w_df.groupby('Date')['Invoice_Amount'].sum().reset_index()
        fig_trend = px.line(daily_trend, x='Date', y='Invoice_Amount', title="Total Portfolio Value Over Time", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.divider()
        c1, c2 = st.columns(2)
        counts = w_df['Risk_Level'].value_counts()
        with c1:
            fig_pie = px.pie(names=counts.index, values=counts.values, hole=0.4, title="Risk Mix",
                             color=counts.index, color_discrete_map={'High Risk': '#ff4b4b', 'Medium Risk': '#ffa500', 'Low Risk': '#28a745'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.bar_chart(counts)
            
        st.dataframe(w_df, use_container_width=True)
        
        if user_q := st.chat_input("Ask about your manual watchlist...", key="q1"):
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            reply = get_ai_response(user_q, w_df.to_string())
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).write(m["content"])
    else:
        st.info("Watchlist is empty.")

# --- TAB 2: BULK PREDICTION (WITH PERSISTENT STACKING) ---
with tab2:
    st.header("📂 Bulk Invoice Analysis")
    
    # 1. NEW: Initialize bulk storage from DISK so it persists across refreshes
    # Using BULK_CACHE_FILE specifically to keep it separate from Tab 1
    if 'bulk_results' not in st.session_state:
        if os.path.exists(BULK_CACHE_FILE):
            st.session_state.bulk_results = pd.read_csv(BULK_CACHE_FILE)
        else:
            st.session_state.bulk_results = pd.DataFrame()

    uploaded_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv")
    
    if uploaded_file:
        try:
            raw_bulk_df = pd.read_csv(uploaded_file)
            new_results = process_batch(raw_bulk_df)
            
            if new_results is not None:
                new_results = new_results.rename(columns={'Invoice_Amount': 'Amount', 'Predicted_Risk': 'Risk_Level'})
                
                if st.button("➕ Add these records to Local Session"):
                    st.session_state.bulk_results = pd.concat(
                        [st.session_state.bulk_results, new_results], 
                        ignore_index=True
                    ).drop_duplicates()
                    
                    # NEW: Save to disk immediately after adding
                    st.session_state.bulk_results.to_csv(BULK_CACHE_FILE, index=False)
                    st.success(f"Added {len(new_results)} records permanently!")
                    st.rerun()

        except Exception as e:
            st.error(f"Analysis failed: {e}")

    if not st.session_state.bulk_results.empty:
        res_df = st.session_state.bulk_results
        risk_col = 'Risk_Level'
        amt_col = 'Amount'

        b1, b2, b3 = st.columns(3)
        b_counts = res_df[risk_col].value_counts()
        b1.metric("Total Invoices", len(res_df))
        b2.metric("High Risk Identified", b_counts.get('High Risk', 0))
        b3.metric("Total Exposure", f"${res_df[amt_col].sum():,.2f}")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🚀 Push ALL Session Data to Shared Dashboard"):
                # FUNCTIONALITY: We save this to a NEW shared file instead of WATCHLIST_FILE
                # to prevent bulk data from leaking into the Tab 1 Manual Watchlist
                SHARED_BULK_FILE = "shared_bulk_dashboard.csv"
                st.session_state.bulk_results.to_csv(SHARED_BULK_FILE, index=False)
                st.success("All session records published to Bulk Client Portal!")
                
        with btn_col2:
            if st.button("🗑️ Clear Local Session"):
                st.session_state.bulk_results = pd.DataFrame()
                if os.path.exists(BULK_CACHE_FILE): os.remove(BULK_CACHE_FILE) # Clear the disk file
                st.rerun()

        st.divider()

        st.subheader("📈 Bulk Upload Exposure Trend")
        if 'Date' in res_df.columns:
            res_df['Date'] = pd.to_datetime(res_df['Date']).dt.date
            bulk_trend = res_df.groupby('Date')[amt_col].sum().reset_index()
            fig_bulk_trend = px.line(bulk_trend, x='Date', y=amt_col, title="Portfolio Exposure Trend", markers=True)
            fig_bulk_trend.update_traces(line_color='#00d1b2')
            st.plotly_chart(fig_bulk_trend, use_container_width=True)

        st.divider()

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📊 Risk Count")
            fig_bulk_bar = px.bar(x=b_counts.index, y=b_counts.values, labels={'x': 'Risk Tier', 'y': 'Invoices'},
                color=b_counts.index, color_discrete_map={'High Risk': '#ff4b4b', 'Medium Risk': '#ffa500', 'Low Risk': '#28a745'})
            st.plotly_chart(fig_bulk_bar, use_container_width=True)
        with col_g2:
            st.subheader("💰 Value at Risk")
            fig_bulk_pie = px.pie(res_df, values=amt_col, names=risk_col, hole=0.4, color=risk_col,
                color_discrete_map={'High Risk': '#ff4b4b', 'Medium Risk': '#ffa500', 'Low Risk': '#28a745'})
            st.plotly_chart(fig_bulk_pie, use_container_width=True)

        st.subheader("📋 Analysis Details")
        st.dataframe(res_df, use_container_width=True)
        
        st.divider()
        st.subheader("💬 Bulk Portfolio Analyst")
        
        full_stat_summary = res_df.groupby(risk_col).agg({amt_col: ['sum', 'count', 'mean'], 'Avg_Past_Delay': 'mean'}).to_string()

        if bulk_q := st.chat_input("Ask about the combined data...", key="q2"):
            st.session_state.bulk_chat_history.append({"role": "user", "content": bulk_q})
            full_context = f"PORTFOLIO STATS:\n{full_stat_summary}\nPREVIEW:\n{res_df.head(len(res_df)).to_string()}"
            reply = get_ai_response(bulk_q, full_context)
            st.session_state.bulk_chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        for bm in st.session_state.bulk_chat_history:
            st.chat_message(bm["role"]).write(bm["content"])
