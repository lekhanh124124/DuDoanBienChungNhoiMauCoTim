import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib

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

# Bước 5: Áp dụng KNN imputation (phương pháp thay thế) cho dữ liệu số
print("\nPhương pháp thay thế: KNN imputation cho dữ liệu số")
# Lưu bản sao dữ liệu đã xử lý bằng phương pháp median để so sánh
median_imputed_data = combined_data.copy()
median_imputed_data.to_csv(r'..\data\processed\median_imputed_data.csv', index=False)

# Áp dụng KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
combined_data[numerical_cols] = knn_imputer.fit_transform(combined_data[numerical_cols])
print("Đã áp dụng KNN imputation cho các cột số")

# Lưu mô hình KNN imputer để sử dụng lại sau này
os.makedirs(r'..\models', exist_ok=True)
joblib.dump(knn_imputer, r'..\models\knn_imputer.pkl')

# Bước 6: Chuẩn hóa dữ liệu
# 6.1: Scaling dữ liệu số
scaler = StandardScaler()
combined_data[numerical_cols] = scaler.fit_transform(combined_data[numerical_cols])
print("\nĐã chuẩn hóa dữ liệu số bằng StandardScaler")

# Lưu scaler để sử dụng lại sau này
joblib.dump(scaler, r'..\models\standard_scaler.pkl')

# 6.2: Xử lý các cột phân loại - chuyển đổi tất cả về chuỗi
categorical_features_to_encode = [col for col in categorical_cols if col != 'num']
print("\nChuyển đổi cột phân loại về kiểu dữ liệu chuỗi...")

# Đảm bảo tất cả giá trị phân loại đều là kiểu chuỗi
for col in categorical_features_to_encode:
    combined_data[col] = combined_data[col].astype(str)
    print(f"Cột {col} đã chuyển thành kiểu chuỗi")

# One-hot encoding cho các biến phân loại
print("\nÁp dụng One-Hot Encoding cho các biến phân loại...")
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = encoder.fit_transform(combined_data[categorical_features_to_encode])

# Tạo DataFrame mới với tên cột sau khi encoding
encoded_feature_names = []
for i, feature in enumerate(categorical_features_to_encode):
    categories = encoder.categories_[i][1:]  # Drop first category
    for category in categories:
        encoded_feature_names.append(f"{feature}_{category}")

encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)
print(f"Đã mã hóa {len(categorical_features_to_encode)} cột thành {encoded_df.shape[1]} đặc trưng")

# Kết hợp dữ liệu đã chuẩn hóa
combined_data_scaled = pd.concat([
    pd.DataFrame(combined_data[numerical_cols], columns=numerical_cols),
    encoded_df,
    pd.DataFrame(combined_data['num'], columns=['num'])
], axis=1)

print(f"Dữ liệu sau khi chuẩn hóa có {combined_data_scaled.shape[1]} cột")

# Lưu encoder để sử dụng lại sau này
joblib.dump(encoder, r'..\models\one_hot_encoder.pkl')

# Bước 7: Lưu kết quả vào file processed_data_scaled.csv
scaled_output_path = r'..\data\processed\processed_data_scaled.csv'
combined_data_scaled.to_csv(scaled_output_path, index=False)
print(f"\nDữ liệu đã được chuẩn hóa và lưu vào: {scaled_output_path}")

# Lưu kết quả gốc vào file processed_data.csv (không chuẩn hóa)
combined_data.to_csv(output_path, index=False)
print(f"Dữ liệu gốc đã được lưu vào: {output_path}")