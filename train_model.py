import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. LOAD DATA
df = pd.read_csv('cash_flow_data.csv')

# 2. CLEANING PHASE
# A. Remove Duplicates
df = df.drop_duplicates()

# B. Clean 'Invoice_Amount' (Remove $ and , then convert to float)
df['Invoice_Amount'] = df['Invoice_Amount'].replace(r'[\$,]', '', regex=True).astype(float)

# C. Handle Outliers (Remove impossible amounts)
df = df[(df['Invoice_Amount'] > 0) & (df['Invoice_Amount'] < 1000000)]

# D. Handle Missing Values (Imputation)
df['Payment_Method'] = df['Payment_Method'].fillna('Unknown')
df['Avg_Past_Delay'] = df['Avg_Past_Delay'].fillna(df['Avg_Past_Delay'].median())
df['Dispute'] = df['Dispute'].fillna(0) # Assume no dispute if missing

# 3. LABELING (Logic for Training)
def assign_risk_refined(row):
    # Rule 1: HIGH RISK (Red Flag)
    # Active dispute OR very long delay (e.g., > 15 days)
    if row['Dispute'] == 1 or row['Avg_Past_Delay'] > 15:
        return 'High Risk'
    
    # Rule 2: MEDIUM RISK (Warning)
    # Moderate delay (7 to 15 days) OR large invoice with some delay
    elif row['Avg_Past_Delay'] > 7:
        return 'Medium Risk'
    elif row['Invoice_Amount'] > 30000 and row['Avg_Past_Delay'] > 2:
        return 'Medium Risk'
        
    # Rule 3: LOW RISK (Safe)
    # Small delays (0 to 7 days) are considered normal business behavior
    return 'Low Risk'

df['Risk_Level'] = df.apply(assign_risk_refined, axis=1)



# 4. PREPROCESSING & TRAINING
le = LabelEncoder()
df['Payment_Method'] = le.fit_transform(df['Payment_Method'])

X = df[['Invoice_Amount', 'Payment_Method', 'Dispute', 'Avg_Past_Delay']]
y = df['Risk_Level']

le_risk = LabelEncoder()
y = le_risk.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save everything for app.py
joblib.dump(model, 'risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_risk, 'label_encoder.pkl')

print("Data Cleaned and Model Trained Successfully!")