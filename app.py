import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import google.generativeai as genai

import threading
from gtts import gTTS
import base64
import speech_recognition as sr
import time
import sounddevice as sd
import wave
import tempfile
import gc

# --- 1. CORE FUNCTIONS: VOICE & SPEECH ---

def speak(text):
    """Thread-safe Text-to-Speech output"""
    def run_speak():
        pythoncom.CoInitialize()
        try:
            local_engine = pyttsx3.init("sapi5")
            local_engine.setProperty('rate', 175)
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()
        except Exception as e:
            print(f"Voice Output Error: {e}")
        finally:
            pythoncom.CoUninitialize()

    threading.Thread(target=run_speak, daemon=True).start()

def takecommand():
    fs = 44100  
    seconds = 7 
    
    # 1. Reset hardware
    try:
        sd.stop()
        sd._terminate()
        sd._initialize()
    except Exception:
        pass

    # Create a unique filename for THIS specific attempt
    temp_filename = f"growfin_{int(time.time())}.wav"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
    
    try:
        st.toast("🎤 Listening...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait() 
        
        # Write the file
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) 
            wf.setframerate(fs)
            wf.writeframes(myrecording.tobytes())
        
        # 2. Use the recognizer inside a strict block
        query = "None"
        r = sr.Recognizer()
        
        # The 'with' block ensures the file is released after reading
        with sr.AudioFile(temp_path) as source:
            audio = r.record(source) 
            try:
                query = r.recognize_google(audio, language='en-in')
            except:
                query = "None"
        
        # --- CRITICAL FIX FOR WINERROR 32 ---
        del audio  # Clear audio data from memory
        gc.collect() # Force garbage collection to release file handles
        
        # Try to delete, but don't crash if Windows is still being stubborn
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except PermissionError:
            print(f"Temporary file {temp_filename} is still locked, skipping deletion.")
            
        return query
            
    except Exception as e:
        print(f"CRITICAL VOICE ERROR: {e}")
        return "None"
    finally:
        sd.stop()
# --- 2. PERSISTENCE & ASSETS ---
WATCHLIST_FILE = "watchlist_data.csv"

def load_saved_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            df = pd.read_csv(WATCHLIST_FILE)
            return df.to_dict('records')
        except:
            return []
    return []

def save_watchlist_to_disk(watchlist_list):
    if watchlist_list:
        df = pd.DataFrame(watchlist_list)
        df.to_csv(WATCHLIST_FILE, index=False)

@st.cache_resource
def load_models():
    # Ensure these files exist in your directory!
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_risk = joblib.load('label_encoder.pkl')
    return model, scaler, le_risk

model, scaler, le_risk = load_models()

# AI Configuration (Growfin-Bot)
genai.configure(api_key="AIzaSyCXFM5rfhS8lgNc4DhPZZijkdrScFPDv5M") # Replace with your key
ai_brain = genai.GenerativeModel('models/gemini-3-flash-preview')

import google.api_core.exceptions

def get_ai_response(user_query, context_data):
    try:
        prompt = f"""
        You are 'Growfin-Bot', a Finance AI Assistant. 
        Context of current data: {context_data}
        Answer the user's question based strictly on this data. Be concise and professional.
        User Question: {user_query}
        """
        response = ai_brain.generate_content(prompt)
        return response.text
    except google.api_core.exceptions.ResourceExhausted:
        error_msg = "⚠️ Quota exceeded. Please try again in a few seconds."
        st.warning(error_msg)
        return error_msg
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- 3. LOGIC HELPERS ---
def apply_business_rules(risk_label, delay, dispute):
    # Dispute is passed as 1 or 0 here
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
    
    # Handle "Yes/No" vs 1/0
    if df_model['Dispute'].dtype == object:
        df_model['Dispute'] = df_model['Dispute'].map({"Yes": 1, "No": 0}).fillna(0)
    else:
        df_model['Dispute'] = pd.to_numeric(df_model['Dispute'], errors='coerce').fillna(0)
    
    df_model['Payment_Method'] = df_model['Payment_Method'].map(method_map).fillna(0)
    
    try:
        X_scaled = scaler.transform(df_model)
        preds = model.predict(X_scaled)
        df_input['Predicted_Risk'] = le_risk.inverse_transform(preds)
        
        # Safe business rule application
        df_input['Predicted_Risk'] = df_input.apply(
            lambda x: apply_business_rules(
                x['Predicted_Risk'], 
                x['Avg_Past_Delay'], 
                1 if str(x['Dispute']).strip().lower() in ['1', 'yes', '1.0'] else 0
            ), axis=1
        )
        return df_input
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

# --- 4. UI INITIALIZATION ---
st.set_page_config(page_title="AI Cash Flow Risk Detector", layout="wide")
st.title("🛡️ AI Cash Flow Risk Detector")

# Initialize Session States
states = {
    'watchlist': load_saved_watchlist(),
    'chat_history': [],
    'bulk_chat_history': [],
    'bulk_context': ""
}
for key, val in states.items():
    if key not in st.session_state:
        st.session_state[key] = val

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
    if os.path.exists(WATCHLIST_FILE): os.remove(WATCHLIST_FILE)
    st.rerun()

# --- 6. TABS ---
tab1, tab2 = st.tabs(["Live Monitoring Watchlist", "Bulk Upload (CSV)"])

with tab1:
    if st.session_state.watchlist:
        w_df = pd.DataFrame(st.session_state.watchlist)
        counts = w_df['Risk_Level'].value_counts()
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(names=counts.index, values=counts.values, hole=0.4, title="Watchlist Risk Mix",
                                 color=counts.index, color_discrete_map={'High Risk': '#ff4b4b', 'Medium Risk': '#ffa500', 'Low Risk': '#28a745'}), use_container_width=True)
        with c2:
            st.bar_chart(counts)
        st.dataframe(w_df, use_container_width=True)
        
        st.divider()
        st.subheader("💬 Growfin-Bot (Individual Analysis)")

        chat_col, voice_col = st.columns([0.9, 0.1])
        with voice_col:
            voice_btn = st.button("🎤", key="tab1_mic")
        with chat_col:
            user_input = st.chat_input("Ask about your watchlist...")

        final_q = None
        if voice_btn:
            voice_q = takecommand()
            if voice_q != "None": final_q = voice_q
        elif user_input:
            final_q = user_input

        if final_q:
            st.session_state.chat_history.append({"role": "user", "content": final_q})
            reply = get_ai_response(final_q, w_df.to_string())
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            speak(reply)
            st.rerun()

        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).write(m["content"])
    else:
        st.info("Watchlist is empty. Add data from the sidebar!")

with tab2:
    st.header("📂 Bulk Invoice Analysis")
    uploaded_file = st.file_uploader("Upload Test_Data.csv", type="csv")
    
    if uploaded_file:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            results_df = process_batch(bulk_df)
            
            if results_df is not None:
                b_counts = results_df['Predicted_Risk'].value_counts()
                high_risk_df = results_df[results_df['Predicted_Risk'] == 'High Risk']
                top_risk_customers = high_risk_df.nlargest(5, 'Invoice_Amount')
                
                # Payment Method Insight
                risk_by_method = results_df.groupby('Payment_Method')['Avg_Past_Delay'].mean().sort_values(ascending=False)
                riskiest_method = risk_by_method.index[0]
                avg_delay_for_method = risk_by_method.values[0]

                # Update Context
                st.session_state.bulk_context = f"""
                PORTFOLIO OVERVIEW:
                - Total Exposure: ${results_df['Invoice_Amount'].sum():,.2f}
                - High Risk Count: {len(high_risk_df)}
                
                TOP RISKS:
                {top_risk_customers[['Customer_ID', 'Invoice_Amount', 'Avg_Past_Delay']].to_string(index=False)}
                
                RISKIEST METHOD: {riskiest_method} with {avg_delay_for_method:.1f} days avg delay.
                """
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Invoices", len(results_df))
                m2.metric("High Risk Found", b_counts.get('High Risk', 0))
                m3.metric("Portfolio Value", f"${results_df['Invoice_Amount'].sum():,.2f}")
                
                st.plotly_chart(px.bar(x=b_counts.index, y=b_counts.values, title="Bulk Risk Distribution"), use_container_width=True)
                st.dataframe(results_df, use_container_width=True)
                
                st.divider()
                st.subheader("💬 Portfolio Analyst")

                b_chat_col, b_voice_col = st.columns([0.9, 0.1])
                with b_voice_col:
                    b_mic_clicked = st.button("🎤", key="tab2_mic")
                with b_chat_col:
                    b_text_q = st.chat_input("Analyze portfolio trends...")

                final_bulk_q = None
                if b_mic_clicked:
                    v_q = takecommand()
                    if v_q != "None": final_bulk_q = v_q
                elif b_text_q:
                    final_bulk_q = b_text_q

                if final_bulk_q:
                    st.session_state.bulk_chat_history.append({"role": "user", "content": final_bulk_q})
                    reply = get_ai_response(final_bulk_q, st.session_state.bulk_context)
                    st.session_state.bulk_chat_history.append({"role": "assistant", "content": reply})
                    speak(reply)
                    st.rerun()

                for bm in st.session_state.bulk_chat_history:
                    st.chat_message(bm["role"]).write(bm["content"])
        except Exception as e:
            st.error(f"Analysis failed: {e}")
