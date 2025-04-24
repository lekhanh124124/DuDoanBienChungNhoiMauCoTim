import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

# Đặt thư mục làm việc là thư mục chứa file mã
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Thư mục làm việc hiện tại:", os.getcwd())

# Định nghĩa đường dẫn tương đối đến các file dữ liệu
data_paths = {
    'cleveland': r'..\data\raw\processed.cleveland.data',
    'hungarian': r'..\data\raw\processed.hungarian.data',
    'switzerland': r'..\data\raw\processed.switzerland.data',
    'va': r'..\data\raw\processed.va.data'
}

# Đường dẫn để lưu file kết quả
output_path = r'..\data\processed\processed_data.csv'

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Định nghĩa tên các cột
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Bước 1: Load dữ liệu từ 4 file
dataframes = {}
for name, path in data_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} không tồn tại. Kiểm tra đường dẫn.")
    dataframes[name] = pd.read_csv(path, header=None, names=columns)

# Bước 2: Xử lý dữ liệu thiếu
# Thay thế "?" và "???" bằng np.nan
for df in dataframes.values():
    df.replace(['?', '???'], np.nan, inplace=True)

# Xác định các cột số và phân loại
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']

# Kiểm tra và chuyển đổi dữ liệu số
for df in dataframes.values():
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Kiểm tra giá trị không hợp lệ trong cột phân loại
for name, df in dataframes.items():
    print(f"\nKiểm tra dữ liệu trong {name}:")
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"Cột {col}: {unique_vals}")

# Xử lý dữ liệu thiếu cho các cột số (sử dụng median)
imputer_num = SimpleImputer(strategy='median', missing_values=np.nan)
for df in dataframes.values():
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

# Xử lý dữ liệu thiếu cho các cột phân loại (sử dụng mode)
imputer_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
for df in dataframes.values():
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# Bước 3: Kết hợp các dataframe thành một
combined_data = pd.concat(dataframes.values(), ignore_index=True)

# Bước 4: Kiểm tra tính nhất quán
print(f"\nSố mẫu Cleveland: {len(dataframes['cleveland'])}")
print(f"Số mẫu Hungarian: {len(dataframes['hungarian'])}")
print(f"Số mẫu Switzerland: {len(dataframes['switzerland'])}")
print(f"Số mẫu VA: {len(dataframes['va'])}")
print(f"Tổng số mẫu: {len(combined_data)}")

# Bước 5: Lưu kết quả vào file processed_data.csv
combined_data.to_csv(output_path, index=False)
print(f"Dữ liệu đã được lưu vào: {output_path}")