import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
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
    required_columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    results = []
    
    for model_name, file_name in metrics_files.items():
        file_path = os.path.join(results_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")
        
        df = pd.read_csv(file_path)
        # Đổi tên cột 'F1-Score' thành 'F1-score' nếu có
        if 'F1-Score' in df.columns:
            df = df.rename(columns={'F1-Score': 'F1-score'})
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"File {file_name} thiếu một số cột: {required_columns}")
        
        # Kiểm tra giá trị trống và hợp lệ
        if df[required_columns[1:]].isnull().any().any():
            raise ValueError(f"File {file_name} chứa giá trị trống.")
        if not df[required_columns[1:]].apply(lambda x: x.between(0, 1)).all().all():
            raise ValueError(f"File {file_name} chứa giá trị không hợp lệ (ngoài [0, 1]).")
        
        results.append(df[required_columns])
    
    combined_results = pd.concat(results, ignore_index=True)
    print("\nKết quả đánh giá từ các mô hình:")
    print(combined_results)
    
    return combined_results

def load_test_data(test_data_files, results_dir):
    """
    Tải tập kiểm tra từ các file CSV đã lưu.
    
    Parameters:
        test_data_files (dict): Từ điển {tên mô hình: tên file CSV chứa X_test, y_test}.
        results_dir (str): Thư mục chứa file CSV.
    
    Returns:
        tuple: (X_test, y_test) - Dữ liệu kiểm tra chung (lấy từ file đầu tiên).
    """
    for model_name, file_name in test_data_files.items():
        file_path = os.path.join(results_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")
        
        df = pd.read_csv(file_path)
        if 'num' not in df.columns:
            raise ValueError(f"File {file_name} không có cột 'num'.")
        
        X_test = df.drop('num', axis=1)
        y_test = df['num'].apply(lambda x: 1 if x > 0 else 0)
        print(f"Đã tải tập kiểm tra từ {file_name}: {X_test.shape}")
        return X_test, y_test
    
    raise ValueError("Không tìm thấy file tập kiểm tra hợp lệ.")

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
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
        except AttributeError:
            print(f"Mô hình {name} không hỗ trợ predict_proba. Bỏ qua ROC curve.")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Model Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def plot_metrics_comparison(combined_results, save_path):
    """
    Vẽ biểu đồ cột so sánh các chỉ số của các mô hình.
    
    Parameters:
        combined_results (pd.DataFrame): Bảng tổng hợp kết quả.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    combined_results_melted = combined_results.melt(id_vars=['Model'], value_vars=metrics, 
                                                   var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Value', hue='Model', data=combined_results_melted)
    plt.title('So sánh hiệu quả các mô hình')
    plt.ylabel('Giá trị')
    plt.ylim(0, 1)
    plt.legend(title='Mô hình')
    
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.2f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrices(models, X_test, y_test, save_path):
    """
    Vẽ ma trận nhầm lẫn cho các mô hình.
    
    Parameters:
        models (dict): Từ điển {tên mô hình: mô hình}.
        X_test (pd.DataFrame): Đặc trưng kiểm tra.
        y_test (pd.Series): Nhãn kiểm tra.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_xticks([0.5, 1.5], ['No Complication', 'Complication'])
        ax.set_yticks([0.5, 1.5], ['No Complication', 'Complication'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def analyze_best_model(combined_results):
    """
    Phân tích và in ra mô hình tốt nhất dựa trên các chỉ số.
    
    Parameters:
        combined_results (pd.DataFrame): Bảng tổng hợp kết quả.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    best_models = {}
    
    for metric in metrics:
        best_model = combined_results.loc[combined_results[metric].idxmax(), 'Model']
        best_value = combined_results[metric].max()
        best_models[metric] = (best_model, best_value)
    
    print("\nMô hình tốt nhất cho từng chỉ số:")
    for metric, (model, value) in best_models.items():
        print(f"{metric}: {model} ({value:.4f})")
    
    best_f1_model = best_models['F1-score'][0]
    print(f"\nKết luận: Mô hình {best_f1_model} là tốt nhất dựa trên F1-score.")
    print("Lưu ý: SVM sử dụng dữ liệu gốc (không SMOTE), trong khi Logistic Regression và Random Forest sử dụng SMOTE, có thể ảnh hưởng đến so sánh.")

def main():
    """
    Hàm chính để so sánh các mô hình.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Thư mục làm việc hiện tại:", os.getcwd())

    results_dir = r'..\..\results'
    models_dir = r'..\..\models'
    figures_dir = r'..\..\figures\plots'

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
    test_data_files = {
        'Logistic Regression': 'test_data_logistic.csv',
        'SVM': 'test_data_svm.csv',
        'Random Forest': 'test_data_random_forest.csv'
    }

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    combined_results = load_metrics(metrics_files, results_dir)
    plot_metrics_comparison(combined_results, os.path.join(figures_dir, 'metrics_comparison.png'))
    analyze_best_model(combined_results)

    comparison_path = os.path.join(results_dir, 'model_comparison.csv')
    combined_results.to_csv(comparison_path, index=False)
    print(f"Bảng so sánh đã lưu: {comparison_path}")

    try:
        X_test, y_test = load_test_data(test_data_files, results_dir)
        models = load_models(model_files, models_dir)
        plot_roc_curve(models, X_test, y_test, os.path.join(figures_dir, 'roc_curve_comparison.png'))
        plot_confusion_matrices(models, X_test, y_test, os.path.join(figures_dir, 'confusion_matrices_comparison.png'))
    except FileNotFoundError as e:
        print(f"Lỗi: {e}. Vui lòng chạy lại các file mô hình để lưu tập kiểm tra.")

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8')
    main()