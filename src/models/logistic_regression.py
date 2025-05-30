import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV và nhị phân hóa biến mục tiêu.
    
    Parameters:
        file_path (str): Đường dẫn đến file CSV.
    
    Returns:
        tuple: (X, y) - Đặc trưng và nhãn.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} không tồn tại.")
    
    df = pd.read_csv(file_path)
    X = df.drop('num', axis=1)  # Đặc trưng
    y = df['num'].apply(lambda x: 1 if x > 0 else 0)  # Nhị phân hóa: 0 (không biến chứng), 1 (có biến chứng)
    
    print(f"Đã đọc dữ liệu: {file_path}")
    print(f"Kích thước dữ liệu: {X.shape}")
    print(f"Phân bố nhãn:\n{y.value_counts()}")
    
    return X, y

def split_and_balance_data(X, y, test_size=0.2, random_state=42):
    """
    Tách dữ liệu thành tập huấn luyện và kiểm tra, cân bằng lớp bằng SMOTE.
    
    Parameters:
        X (pd.DataFrame): Đặc trưng.
        y (pd.Series): Nhãn.
        test_size (float): Tỷ lệ tập kiểm tra.
        random_state (int): Seed cho ngẫu nhiên.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Cân bằng lớp bằng SMOTE cho tập huấn luyện
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}")
    print(f"Phân bố nhãn sau SMOTE:\n{pd.Series(y_train).value_counts()}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """
    Huấn luyện mô hình Logistic Regression với GridSearchCV.
    
    Parameters:
        X_train (pd.DataFrame): Đặc trưng huấn luyện.
        y_train (pd.Series): Nhãn huấn luyện.
    
    Returns:
        model: Mô hình đã huấn luyện.
    """
    lr = LogisticRegression(random_state=42, max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Tham số chính quy hóa
        'penalty': ['l1', 'l2'],        # Loại chính quy hóa
        'solver': ['liblinear']         # Solver hỗ trợ l1 và l2
    }
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Logistic Regression - Tham số tốt nhất: {grid.best_params_}")
    print(f"Điểm F1 tốt nhất: {grid.best_score_:.4f}")
    
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình trên tập kiểm tra.
    
    Parameters:
        model: Mô hình đã huấn luyện.
        X_test (pd.DataFrame): Đặc trưng kiểm tra.
        y_test (pd.Series): Nhãn kiểm tra.
    
    Returns:
        dict: Các chỉ số đánh giá.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    return metrics, y_prob

def plot_roc_curve(model, X_test, y_test, save_path):
    """
    Vẽ đường cong ROC cho mô hình Logistic Regression.
    
    Parameters:
        model: Mô hình đã huấn luyện.
        X_test (pd.DataFrame): Đặc trưng kiểm tra.
        y_test (pd.Series): Nhãn kiểm tra.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    plt.figure(figsize=(8, 6))
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def main():
    """
    Hàm chính để chạy xây dựng và đánh giá mô hình Logistic Regression.
    """
    # Đặt thư mục làm việc
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Thư mục làm việc hiện tại:", os.getcwd())

    # Định nghĩa đường dẫn
    data_path = r'..\..\data\processed\processed_data_scaled.csv'
    models_dir = r'..\..\models'
    results_dir = r'..\..\results'
    figures_dir = r'..\..\figures\plots'

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Đọc dữ liệu
    X, y = load_data(data_path)

    # Tách và cân bằng dữ liệu
    X_train, X_test, y_train, y_test = split_and_balance_data(X, y)

    # Huấn luyện mô hình Logistic Regression
    print("\nHuấn luyện mô hình Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    # Đánh giá mô hình
    print("\nĐánh giá mô hình trên tập kiểm tra...")
    metrics, y_prob = evaluate_model(lr_model, X_test, y_test)
    results_df = pd.DataFrame([metrics])
    print("\nKết quả đánh giá Logistic Regression:")
    print(results_df)

    # Lưu kết quả đánh giá
    results_df.to_csv(os.path.join(results_dir, 'logistic_regression_metrics.csv'), index=False)
    print(f"Kết quả đánh giá đã lưu: {os.path.join(results_dir, 'logistic_regression_metrics.csv')}")

    # Vẽ và lưu đường cong ROC
    plot_roc_curve(lr_model, X_test, y_test, os.path.join(figures_dir, 'roc_curve_logistic_regression.png'))

    # Lưu mô hình
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    print(f"Mô hình Logistic Regression đã lưu: {os.path.join(models_dir, 'logistic_regression.pkl')}")

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8')  # Đặt style cho biểu đồ
    main()