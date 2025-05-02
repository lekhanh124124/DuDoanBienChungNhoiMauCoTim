import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import joblib

def load_metrics(metrics_files, results_dir):
    """
    Đọc các file CSV chứa kết quả đánh giá của các mô hình.
    
    Parameters:
        metrics_files (dict): Từ điển {tên mô hình: tên file CSV}.
        results_dir (str): Thư mục chứa file CSV.
    
    Returns:
        pd.DataFrame: Bảng tổng hợp kết quả.
    """
    results = []
    for model_name, file_name in metrics_files.items():
        file_path = os.path.join(results_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")
        df = pd.read_csv(file_path)
        results.append(df)
    
    combined_results = pd.concat(results, ignore_index=True)
    print("\nKết quả đánh giá từ các mô hình:")
    print(combined_results)
    
    return combined_results

def load_data_and_split(file_path, test_size=0.2, random_state=42):
    """
    Đọc dữ liệu và tách tập kiểm tra để dự đoán ROC.
    
    Parameters:
        file_path (str): Đường dẫn đến file CSV.
        test_size (float): Tỷ lệ tập kiểm tra.
        random_state (int): Seed cho ngẫu nhiên.
    
    Returns:
        tuple: (X_test, y_test)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} không tồn tại.")
    
    df = pd.read_csv(file_path)
    X = df.drop('num', axis=1)
    y = df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Kích thước tập kiểm tra: {X_test.shape}")
    return X_test, y_test

def load_models(model_files, models_dir):
    """
    Đọc các mô hình đã lưu từ file .pkl.
    
    Parameters:
        model_files (dict): Từ điển {tên mô hình: tên file .pkl}.
        models_dir (str): Thư mục chứa file .pkl.
    
    Returns:
        dict: Từ điển {tên mô hình: mô hình}.
    """
    models = {}
    for model_name, file_name in model_files.items():
        file_path = os.path.join(models_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")
        models[model_name] = joblib.load(file_path)
        print(f"Đã tải mô hình {model_name} từ {file_path}")
    
    return models

def plot_roc_curve(models, X_test, y_test, save_path):
    """
    Vẽ đường cong ROC cho các mô hình.
    
    Parameters:
        models (dict): Từ điển {tên mô hình: mô hình}.
        X_test (pd.DataFrame): Đặc trưng kiểm tra.
        y_test (pd.Series): Nhãn kiểm tra.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Model Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def main():
    """
    Hàm chính để so sánh các mô hình.
    """
    # Đặt thư mục làm việc
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Thư mục làm việc hiện tại:", os.getcwd())

    # Định nghĩa đường dẫn
    data_path = r'..\..\data\processed\processed_data_scaled.csv'
    results_dir = r'..\..\results'
    models_dir = r'..\..\models'
    figures_dir = r'..\..\figures\plots'

    # Định nghĩa file kết quả và mô hình
    metrics_files = {
        'Logistic Regression': 'logistic_regression_metrics.csv',
        'SVM': 'svm_metrics.csv',
        'Random Forest': 'random_forest_metrics.csv'
    }
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'SVM': 'svm.pkl',
        'Random Forest': 'random_forest.pkl'
    }

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Tải kết quả đánh giá
    combined_results = load_metrics(metrics_files, results_dir)

    # Lưu bảng so sánh
    comparison_path = os.path.join(results_dir, 'model_comparison.csv')
    combined_results.to_csv(comparison_path, index=False)
    print(f"Bảng so sánh đã lưu: {comparison_path}")

    # Tải dữ liệu kiểm tra
    X_test, y_test = load_data_and_split(data_path)

    # Tải mô hình
    models = load_models(model_files, models_dir)

    # Vẽ và lưu đường cong ROC
    plot_roc_curve(models, X_test, y_test, os.path.join(figures_dir, 'roc_curve_comparison.png'))

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8')  # Đặt style cho biểu đồ
    main()