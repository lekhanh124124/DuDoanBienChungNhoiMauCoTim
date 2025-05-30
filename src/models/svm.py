import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import joblib

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV.
    
    Parameters:
        file_path (str): Đường dẫn đến file CSV.
    
    Returns:
        tuple: (X, y) - Đặc trưng và nhãn.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} không tồn tại.")
    
    print(f"Đọc dữ liệu từ {file_path}")
    df = pd.read_csv(file_path)
    
    # Tách nhãn, chuyển thành nhị phân (0: không biến chứng, 1: có biến chứng)
    X = df.drop('num', axis=1)
    y = df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    print(f"Kích thước dữ liệu: {X.shape}")
    print(f"Phân bố nhãn:\n{y.value_counts()}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành tập huấn luyện và kiểm tra.
    
    Parameters:
        X (pd.DataFrame): Đặc trưng.
        y (pd.Series): Nhãn.
        test_size (float): Tỷ lệ tập kiểm tra.
        random_state (int): Seed cho ngẫu nhiên.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test).
    """
    print(f"Chia dữ liệu với test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def tune_svm_model(X_train, y_train, cv=5, n_jobs=-1):
    """
    Tìm siêu tham số tối ưu cho mô hình SVM.
    
    Parameters:
        X_train (pd.DataFrame): Đặc trưng huấn luyện.
        y_train (pd.Series): Nhãn huấn luyện.
        cv (int): Số fold cho cross-validation.
        n_jobs (int): Số luồng xử lý song song.
    
    Returns:
        dict: Siêu tham số tối ưu.
    """
    print(f"Tìm siêu tham số tối ưu cho SVM với {cv}-fold cross-validation")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True),
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Kết quả tìm tham số tốt nhất:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

def train_svm_model(X_train, y_train, params):
    """
    Huấn luyện mô hình SVM với tham số đã cho.
    
    Parameters:
        X_train (pd.DataFrame): Đặc trưng huấn luyện.
        y_train (pd.Series): Nhãn huấn luyện.
        params (dict): Siêu tham số.
    
    Returns:
        SVC: Mô hình đã huấn luyện.
    """
    print(f"Huấn luyện SVM với tham số: {params}")
    
    model = SVC(probability=True, **params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình trên tập kiểm tra.
    
    Parameters:
        model (SVC): Mô hình SVM đã huấn luyện.
        X_test (pd.DataFrame): Đặc trưng kiểm tra.
        y_test (pd.Series): Nhãn kiểm tra.
    
    Returns:
        dict: Kết quả đánh giá.
    """
    print("Đánh giá mô hình")
    
    # Dự đoán
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Tính các độ đo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Tính ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    
    # In kết quả
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Đóng gói kết quả
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'cm': cm
    }
    
    return results

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    Vẽ đường cong ROC.
    
    Parameters:
        fpr (array): False positive rate.
        tpr (array): True positive rate.
        roc_auc (float): Diện tích dưới đường cong ROC.
        save_path (str): Đường dẫn để lưu biểu đồ.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for SVM Classifier')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    """
    Vẽ ma trận nhầm lẫn.
    
    Parameters:
        cm (array): Ma trận nhầm lẫn.
        save_path (str): Đường dẫn để lưu biểu đồ.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['No Complication', 'Complication'])
    plt.yticks([0.5, 1.5], ['No Complication', 'Complication'])
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_results(results, metrics_path):
    """
    Lưu kết quả đánh giá vào file CSV.
    
    Parameters:
        results (dict): Kết quả đánh giá.
        metrics_path (str): Đường dẫn để lưu kết quả.
    """
    print(f"Lưu kết quả đánh giá vào {metrics_path}")
    
    # Tạo DataFrame
    metrics_df = pd.DataFrame({
        'Model': ['SVM'],
        'Accuracy': [results['accuracy']],
        'Precision': [results['precision']],
        'Recall': [results['recall']],
        'F1-Score': [results['f1']],
        'ROC-AUC': [results['roc_auc']]
    })
    
    # Lưu vào file CSV
    metrics_df.to_csv(metrics_path, index=False)

def main():
    """
    Hàm chính để huấn luyện và đánh giá mô hình SVM.
    """
    # Đặt thư mục làm việc
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Thư mục làm việc hiện tại:", os.getcwd())

    # Định nghĩa đường dẫn
    data_path = r'..\..\data\processed\processed_data_scaled.csv'
    model_path = r'..\..\models\svm.pkl'
    metrics_path = r'..\..\results\svm_metrics.csv'
    figures_dir = r'..\..\figures\plots'
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Đọc dữ liệu
    X, y = load_data(data_path)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Tìm siêu tham số tối ưu
    best_params = tune_svm_model(X_train, y_train)
    
    # Huấn luyện mô hình
    model = train_svm_model(X_train, y_train, best_params)
    
    # Đánh giá mô hình
    results = evaluate_model(model, X_test, y_test)
    
    # Vẽ và lưu đường cong ROC
    plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'], os.path.join(figures_dir, 'roc_curve_svm.png'))
    
    # Vẽ và lưu ma trận nhầm lẫn
    plot_confusion_matrix(results['cm'], os.path.join(figures_dir, 'confusion_matrix_svm.png'))
    
    # Lưu kết quả đánh giá
    save_results(results, metrics_path)
    
    # Lưu mô hình
    joblib.dump(model, model_path)
    print(f"Đã lưu mô hình vào {model_path}")

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8')  # Đặt style cho biểu đồ
    main()