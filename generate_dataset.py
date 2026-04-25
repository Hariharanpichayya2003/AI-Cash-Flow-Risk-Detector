import pandas as pd
import numpy as np

# Set seed for consistency
np.random.seed(42)
n_rows = 1000

data = {
    'Customer_ID': [f'C-{np.random.randint(1000, 1100)}' for _ in range(n_rows)],
    'Invoice_Amount': np.random.choice(['$5,000', '$12,500.50', '2500', '-500', '999999999', '$8,400'], n_rows),
    'Payment_Method': np.random.choice(['ACH', 'Check', 'Credit Card', None], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
    'Dispute': np.random.choice([0, 1, np.nan], n_rows, p=[0.8, 0.1, 0.1]),
    'Avg_Past_Delay': np.random.choice([2, 5, 10, 25, 100, np.nan], n_rows)
}

df = pd.DataFrame(data)

# Add some duplicate rows to test your cleaning skills
df = pd.concat([df, df.iloc[:10]], ignore_index=True)

df.to_csv('cash_flow_data.csv', index=False)
print("Dirty dataset 'cash_flow_data.csv' created!")