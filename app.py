import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import google.generativeai as genai

# --- 1. ASSET LOADING & AI SETUP ---
@st.cache_resource
def load_models():
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_risk = joblib.load('label_encoder.pkl')
    return model, scaler, le_risk

model, scaler, le_risk = load_models()

# AI Configuration (Using your provided key)
genai.configure(api_key="AIzaSyDnzC_eiK0QY_bGa6R49VGKCalRBQ-O-Mk")
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

# --- 2. LOGIC HELPERS ---
def apply_business_rules(risk_label, delay, dispute):
    if delay > 20 or dispute == 1:
        return 'High Risk'
    return risk_label

def process_batch(df_input):
    required_cols = ['Invoice_Amount', 'Payment_Method', 'Dispute', 'Avg_Past_Delay']
    method_map = {"ACH": 0, "Check": 1, "Credit Card": 2, "Wire": 3}
    
    # Check for missing required columns
    missing = [col for col in required_cols if col not in df_input.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return None

    df_model = df_input[required_cols].copy()
    
    # Handle mixed data types for Dispute
    if df_model['Dispute'].dtype == object:
        df_model['Dispute'] = df_model['Dispute'].map({"Yes": 1, "No": 0}).fillna(0)
    
    df_model['Payment_Method'] = df_model['Payment_Method'].map(method_map).fillna(0)
    X_scaled = scaler.transform(df_model)
    preds = model.predict(X_scaled)
    df_input['Predicted_Risk'] = le_risk.inverse_transform(preds)
    
    # Apply Business Overlays
    df_input['Predicted_Risk'] = df_input.apply(
        lambda x: apply_business_rules(x['Predicted_Risk'], x['Avg_Past_Delay'], 1 if x['Dispute'] in [1, "Yes"] else 0), axis=1
    )
    return df_input

# --- 3. UI INITIALIZATION ---
st.set_page_config(page_title="AI Cash Flow Risk Detector", layout="wide")
st.title("🛡️ AI Cash Flow Risk Detector")

if 'watchlist' not in st.session_state: st.session_state.watchlist = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'bulk_chat_history' not in st.session_state: st.session_state.bulk_chat_history = []

# --- 4. SIDEBAR ---
st.sidebar.header("Individual Invoice Details")
cust_id = st.sidebar.text_input("Customer ID", value="CUST-001")
invoice_amt = st.sidebar.number_input("Invoice Amount ($)", min_value=0, value=5000)
payment_method = st.sidebar.selectbox("Payment Method", ["ACH", "Check", "Credit Card", "Wire"])
dispute_radio = st.sidebar.radio("Active Dispute?", ["No", "Yes"])
past_delay = st.sidebar.slider("Average Past Delays (Days)", 0, 60, 5)

tab1, tab2 = st.tabs(["Live Monitoring Watchlist", "Bulk Upload (CSV)"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
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

    if st.session_state.watchlist:
        w_df = pd.DataFrame(st.session_state.watchlist)
        c1, c2 = st.columns(2)
        counts = w_df['Risk_Level'].value_counts()
        with c1:
            st.plotly_chart(px.pie(names=counts.index, values=counts.values, hole=0.4, 
                         color=counts.index, color_discrete_map={'High Risk': '#ff4b4b', 'Medium Risk': '#ffa500', 'Low Risk': '#28a745'}))
        with c2:
            st.bar_chart(counts)
            
        st.dataframe(w_df, use_container_width=True)
        
        st.subheader("💬 Hi ! HOW CAN  I HELP YOU ")
        if user_q := st.chat_input("Ask about these specific customers...", key="q1"):
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            reply = get_ai_response(user_q, w_df.to_string())
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).write(m["content"])

# --- TAB 2: BULK PREDICTION ---
with tab2:
    st.header("📂 Bulk Invoice Analysis")
    uploaded_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv")
    
    if uploaded_file:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            results_df = process_batch(bulk_df)
            
            if results_df is not None:
                # Show Visuals
                b1, b2, b3 = st.columns(3)
                b_counts = results_df['Predicted_Risk'].value_counts()
                b1.metric("Total Invoices", len(results_df))
                b2.metric("High Risk Identified", b_counts.get('High Risk', 0))
                b3.metric("Total Exposure", f"${results_df['Invoice_Amount'].sum():,.2f}")
                
                st.plotly_chart(px.bar(x=b_counts.index, y=b_counts.values, title="Risk Counts", labels={'x':'Risk', 'y':'Count'}), use_container_width=True)
                st.dataframe(results_df, use_container_width=True)
                
                st.divider()
                st.subheader("💬 Bulk Portfolio Analyst")
                st.info("Growfin-Bot is now analyzing your entire CSV. You can ask about trends or specific totals.")
                
                # --- FLEXIBLE SUMMARY LOGIC ---
                # Check if 'Customer_ID' exists to avoid the Label error
                if 'Customer_ID' in results_df.columns:
                    summary_context = results_df.groupby('Predicted_Risk').agg({
                        'Invoice_Amount': 'sum',
                        'Customer_ID': 'count'
                    }).to_string()
                else:
                    # Fallback if Customer_ID is missing
                    summary_context = results_df.groupby('Predicted_Risk').agg({
                        'Invoice_Amount': 'sum',
                        'Predicted_Risk': 'count'
                    }).rename(columns={'Predicted_Risk': 'Record_Count'}).to_string()
                
                if bulk_q := st.chat_input("Ask about the CSV data (e.g. 'How much money is in High Risk?')", key="q2"):
                    st.session_state.bulk_chat_history.append({"role": "user", "content": bulk_q})
                    full_context = f"Summary: {summary_context}\nFull Data Preview: {results_df.head(20).to_string()}"
                    reply = get_ai_response(bulk_q, full_context)
                    st.session_state.bulk_chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

                for bm in st.session_state.bulk_chat_history:
                    st.chat_message(bm["role"]).write(bm["content"])
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
