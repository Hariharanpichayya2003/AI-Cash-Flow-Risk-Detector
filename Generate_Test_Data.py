import pandas as pd
import numpy as np

# Set seed so you get the same data every time
np.random.seed(99)

n_test = 100
data = {
    'Invoice_Amount': np.random.randint(1000, 50000, n_test),
    'Payment_Method': np.random.choice(['ACH', 'Check', 'Credit Card', 'Wire'], n_test),
    'Dispute': np.random.choice([0, 1], n_test, p=[0.8, 0.2]),
    'Avg_Past_Delay': np.random.randint(0, 30, n_test)
}

test_df = pd.DataFrame(data)
test_df.to_csv('Test_Data.csv', index=False)
print("✅ Created 'test_invoices.csv' with 100 rows for testing!")