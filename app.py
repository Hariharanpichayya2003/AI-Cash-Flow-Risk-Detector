import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import google.generativeai as genai

# --- 1. PERSISTENCE HELPERS ---
WATCHLIST_FILE = "watchlist_data.csv"

def load_saved_watchlist():
    """Loads data from CSV into a list of dictionaries for session state"""
    if os.path.exists(WATCHLIST_FILE):
        try:
            df = pd.read_csv(WATCHLIST_FILE)
            return df.to_dict('records')
        except:
            return []
    return []

def save_watchlist_to_disk(watchlist_list):
    """Saves the current session state watchlist to a CSV file"""
    if watchlist_list:
        df = pd.DataFrame(watchlist_list)
        df.to_csv(WATCHLIST_FILE, index=False)

# --- 2. ASSET LOADING & AI SETUP ---
@st.cache_resource
def load_models():
    # Ensure these files exist in your folder!
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_risk = joblib.load('label_encoder.pkl')
    return model, scaler, le_risk

model, scaler, le_risk = load_models()

# AI Configuration
genai.configure(api_key="AIzaSyCsZclH1eVn9DkFBy7YazPDqsw3YDfWlQQ")
ai_brain = genai.GenerativeModel('models/gemini-3-flash-preview')

def get_ai_response(user_query, context_data):
    prompt = f"""
    You are 'Growfin-Bot', a Finance AI Assistant. 
    Context of current data: {context_data}
    
    Answer the user's question based strictly on this data. Be concise and professional.
    User Question: {user_query}
    """
    response = ai_brain.generate_content(prompt)
    return response.text

# --- 3. LOGIC HELPERS ---
def apply_business_rules(risk_label, delay, dispute):
    if delay > 20 or dispute == 1:
        return 'High Risk'
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
        df_model['Dispute'] = df_model['Dispute'].map({"Yes": 1, "No": 0}).fillna(0)
    
    df_model['Payment_Method'] = df_model['Payment_Method'].map(method_map).fillna(0)
    X_scaled = scaler.transform(df_model)
    preds = model.predict(X_scaled)
    df_input['Predicted_Risk'] = le_risk.inverse_transform(preds)
    
    df_input['Predicted_Risk'] = df_input.apply(
        lambda x: apply_business_rules(x['Predicted_Risk'], x['Avg_Past_Delay'], 1 if x['Dispute'] in [1, "Yes"] else 0), axis=1
    )
    return df_input

# --- 4. UI INITIALIZATION ---
st.set_page_config(page_title="AI Cash Flow Risk Detector", layout="wide")
st.title("🛡️ AI Cash Flow Risk Detector")

# Initialize Session State with saved data
if 'watchlist' not in st.session_state: 
    st.session_state.watchlist = load_saved_watchlist()
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'bulk_chat_history' not in st.session_state: st.session_state.bulk_chat_history = []

# --- 5. SIDEBAR ---
st.sidebar.header("Individual Invoice Details")
cust_id = st.sidebar.text_input("Customer ID", value="CUST-001")
invoice_amt = st.sidebar.number_input("Invoice Amount ($)", min_value=0, value=5000)
payment_method = st.sidebar.selectbox("Payment Method", ["ACH", "Check", "Credit Card", "Wire"])
dispute_radio = st.sidebar.radio("Active Dispute?", ["No", "Yes"])
past_delay = st.sidebar.slider("Average Past Delays (Days)", 0, 60, 5)

if st.sidebar.button("Analyze & Add to Watchlist"):
    dispute_num = 1 if dispute_radio == "Yes" else 0
    method_map = {"ACH": 0, "Check": 1, "Credit Card": 2, "Wire": 3}
    
    input_df = pd.DataFrame([[invoice_amt, method_map[payment_method], dispute_num, past_delay]], 
                            columns=['Invoice_Amount', 'Payment_Method', 'Dispute', 'Avg_Past_Delay'])
    
    scaled = scaler.transform(input_df)
    pred = le_risk.inverse_transform(model.predict(scaled))[0]
    final_risk = apply_business_rules(pred, past_delay, dispute_num)
    
    st.session_state.watchlist.append({
        'Customer_ID': cust_id, 'Invoice_Amount': invoice_amt, 
        'Payment_Method': payment_method, 'Dispute': dispute_radio, 
        'Avg_Past_Delay': past_delay, 'Risk_Level': final_risk
    })
    save_watchlist_to_disk(st.session_state.watchlist)
    st.rerun()

if st.sidebar.button("Clear All Saved Data"):
    st.session_state.watchlist = []
    if os.path.exists(WATCHLIST_FILE):
        os.remove(WATCHLIST_FILE)
    st.rerun()

tab1, tab2 = st.tabs(["Live Monitoring Watchlist", "Bulk Upload (CSV)"])

# --- TAB 1: SINGLE PREDICTION & WATCHLIST ---
with tab1:
    if st.session_state.watchlist:
        w_df = pd.DataFrame(st.session_state.watchlist)
        c1, c2 = st.columns(2)
        counts = w_df['Risk_Level'].value_counts()
        with c1:
            st.plotly_chart(px.pie(names=counts.index, values=counts.values, hole=0.4, title="Watchlist Risk Mix",
                                 color=counts.index, color_discrete_map={'High Risk': '#ff4b4b', 'Medium Risk': '#ffa500', 'Low Risk': '#28a745'}))
        with c2:
            st.bar_chart(counts)
            
        st.dataframe(w_df, use_container_width=True)
        
        st.subheader("💬 Live Chatbot")
        if user_q := st.chat_input("Ask about your watchlist...", key="q1"):
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            reply = get_ai_response(user_q, w_df.to_string())
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).write(m["content"])
    else:
        st.info("Your watchlist is empty. Add an invoice from the sidebar to see analysis!")

# --- TAB 2: BULK PREDICTION ---
with tab2:
    st.header("📂 Bulk Invoice Analysis")
    uploaded_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv")
    
    if uploaded_file:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            results_df = process_batch(bulk_df)
            
            if results_df is not None:
                b1, b2, b3 = st.columns(3)
                b_counts = results_df['Predicted_Risk'].value_counts()
                b1.metric("Total Invoices", len(results_df))
                b2.metric("High Risk Identified", b_counts.get('High Risk', 0))
                b3.metric("Total Exposure", f"${results_df['Invoice_Amount'].sum():,.2f}")
                
                st.plotly_chart(px.bar(x=b_counts.index, y=b_counts.values, title="Risk Counts", labels={'x':'Risk', 'y':'Count'}), use_container_width=True)
                st.dataframe(results_df, use_container_width=True)
                
                st.divider()
                st.subheader("💬 Bulk Portfolio Analyst")
                
                # Context logic for Bulk Chat
                summary_context = results_df.groupby('Predicted_Risk').agg({'Invoice_Amount': 'sum'}).to_string()
                
                if bulk_q := st.chat_input("Ask about the CSV data...", key="q2"):
                    st.session_state.bulk_chat_history.append({"role": "user", "content": bulk_q})
                    full_context = f"Summary: {summary_context}\nData Preview: {results_df.head(10).to_string()}"
                    reply = get_ai_response(bulk_q, full_context)
                    st.session_state.bulk_chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

                for bm in st.session_state.bulk_chat_history:
                    st.chat_message(bm["role"]).write(bm["content"])
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
