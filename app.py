import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_models():
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_risk = joblib.load('label_encoder.pkl')
    return model, scaler, le_risk

model, scaler, le_risk = load_models()

# --- 2. NOTIFICATION LOGIC ---
def trigger_notification(amount, risk, delay, is_batch=False, count=0):
    """Handles both individual alerts and batch summary alerts"""
    if risk == "High Risk":
        if is_batch:
            # Summary notification for bulk upload
            st.toast(f"Batch Alert: {count} High Risk accounts identified!", icon="⚠️")
            st.warning(f"🚨 **Action Required:** {count} critical risks detected in this batch. Recovery Team notified.")
        else:
            # Individual notification for watchlist
            st.toast(f"Escalated ${amount} to Recovery Team", icon="🚨")
            st.error(f"⚠️ High Risk Action: Automated notification sent to Collections (Delay: {delay} days).")

# --- 3. UI SETUP ---
st.set_page_config(page_title="AI Cash Flow Risk Detector", layout="wide")
st.title("🛡️ AI Cash Flow Risk Detector")

# --- 4. HELPER FUNCTIONS ---
def apply_business_rules(risk_label, delay, dispute):
    if delay > 15 or dispute == 1:
        return 'High Risk'
    return risk_label

def process_batch(df_input):
    required_cols = ['Invoice_Amount', 'Payment_Method', 'Dispute', 'Avg_Past_Delay']
    method_map = {"ACH": 0, "Check": 1, "Credit Card": 2, "Wire": 3}
    df_model = df_input[required_cols].copy()
    df_model['Payment_Method'] = df_model['Payment_Method'].map(method_map).fillna(0)
    X_scaled = scaler.transform(df_model)
    preds = model.predict(X_scaled)
    df_input['Predicted_Risk'] = le_risk.inverse_transform(preds)
    df_input['Predicted_Risk'] = df_input.apply(
        lambda x: apply_business_rules(x['Predicted_Risk'], x['Avg_Past_Delay'], x['Dispute']), axis=1
    )
    return df_input

# --- 5. SESSION STATE INITIALIZATION ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# --- 6. SIDEBAR ---
st.sidebar.header("Individual Invoice Details")
invoice_amt = st.sidebar.number_input("Invoice Amount ($)", min_value=0, value=5000)
payment_method = st.sidebar.selectbox("Payment Method", ["ACH", "Check", "Credit Card", "Wire"])
dispute_radio = st.sidebar.radio("Active Dispute?", ["No", "Yes"])
past_delay = st.sidebar.slider("Average Past Delays (Days)", 0, 60, 5)

# --- 7. TABS INTERFACE ---
tab1, tab2 = st.tabs(["Live Monitoring Watchlist", "Bulk Upload (CSV)"])

with tab1:
    st.subheader("Add to Watchlist")
    if st.sidebar.button("Analyze & Add to Watchlist"):
        method_map = {"ACH": 0, "Check": 1, "Credit Card": 2, "Wire": 3}
        dispute_val = 1 if dispute_radio == "Yes" else 0
        input_data = pd.DataFrame([[invoice_amt, method_map[payment_method], dispute_val, past_delay]], 
                                 columns=['Invoice_Amount', 'Payment_Method', 'Dispute', 'Avg_Past_Delay'])
        scaled = scaler.transform(input_data)
        result = le_risk.inverse_transform(model.predict(scaled))[0]
        final_result = apply_business_rules(result, past_delay, dispute_val)
        
        st.session_state.watchlist.append({
            'Invoice_Amount': invoice_amt, 'Payment_Method': payment_method,
            'Dispute': dispute_radio, 'Avg_Past_Delay': past_delay, 'Risk_Level': final_result
        })

        # Individual Notification Trigger
        trigger_notification(invoice_amt, final_result, past_delay)

    if st.session_state.watchlist:
        watch_df = pd.DataFrame(st.session_state.watchlist)
        risk_counts = watch_df['Risk_Level'].value_counts()
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Risk Distribution (Donut Chart)**")
            fig = px.pie(names=risk_counts.index, values=risk_counts.values, hole=0.4,
                         color=risk_counts.index, color_discrete_map={
                             'High Risk': '#ff4b4b', 'Medium Risk': '#ffa500', 'Low Risk': '#28a745'
                         })
            fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("**Risk Volume (Bar Chart)**")
            st.bar_chart(risk_counts)
        st.dataframe(watch_df, use_container_width=True)
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            watchlist_csv = watch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", watchlist_csv, "watchlist_report.csv", "text/csv")
        with col_btn2:
            if st.button("🗑️ Clear List"):
                st.session_state.watchlist = []
                st.rerun()

with tab2:
    st.header("📂 Bulk Invoice Analysis")
    st.write("Upload a CSV file containing: `Invoice_Amount`, `Payment_Method`, `Dispute`, `Avg_Past_Delay`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        try:
            results_df = process_batch(raw_df)
            counts = results_df['Predicted_Risk'].value_counts()
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Low Risk ✅", counts.get('Low Risk', 0))
            m2.metric("Medium Risk ⚠️", counts.get('Medium Risk', 0))
            m3.metric("High Risk 🚨", counts.get('High Risk', 0))

            # Batch Notification Trigger
            high_risk_count = counts.get('High Risk', 0)
            if high_risk_count > 0:
                trigger_notification(None, "High Risk", None, is_batch=True, count=high_risk_count)

            def color_risk(val):
                color = 'red' if val == 'High Risk' else ('orange' if val == 'Medium Risk' else 'green')
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(results_df.style.map(color_risk, subset=['Predicted_Risk']), use_container_width=True)
            batch_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Analysis", batch_csv, "batch_risk_report.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
