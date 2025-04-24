import pandas as pd
df = pd.read_csv(r'E:\workspace\DataMining\DoAn\data\processed\processed_data.csv')
print(df.describe())  # Thống kê cột số
for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']:
    print(f"\nTần suất {col}:\n", df[col].value_counts())