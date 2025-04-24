import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """
    Đọc file CSV và kiểm tra dữ liệu cơ bản.
    
    Parameters:
        file_path (str): Đường dẫn đến file CSV.
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu.
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        abs_path = os.path.abspath(file_path)
        raise FileNotFoundError(f"File {file_path} không tồn tại. Đường dẫn tuyệt đối: {abs_path}")
    
    # Đọc file
    try:
        df = pd.read_csv(file_path)
        print("\nDữ liệu đã được đọc:")
        print(f"Số mẫu: {len(df)}")
        print(f"Cột: {df.columns.tolist()}")
        print("\nKiểm tra giá trị thiếu:\n", df.isna().sum())
        print("\nKiểu dữ liệu:\n", df.dtypes)
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        exit(1)

def descriptive_statistics(df, numerical_cols):
    """
    Tính thống kê mô tả cho các cột số.
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        numerical_cols (list): Danh sách các cột số.
    """
    print("\nThống kê mô tả cho các cột số:")
    print(df[numerical_cols].describe())

def frequency_analysis(df, categorical_cols):
    """
    Tính tần suất giá trị cho các cột phân loại.
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        categorical_cols (list): Danh sách các cột phân loại.
    """
    print("\nTần suất giá trị cho các cột phân loại:")
    for col in categorical_cols:
        print(f"\nTần suất {col}:\n", df[col].value_counts())

def plot_numerical_distributions(df, numerical_cols, save_dir=None):
    """
    Vẽ histogram và boxplot cho các cột số.
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        numerical_cols (list): Danh sách các cột số.
        save_dir (str, optional): Thư mục để lưu biểu đồ.
    """
    # Histogram
    df[numerical_cols].hist(bins=20, figsize=(15, 10))
    plt.suptitle("Phân phối các cột số")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "numerical_histograms.png"))
    plt.show()

    # Boxplot
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 3, i+1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.suptitle("Boxplot các cột số")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "numerical_boxplots.png"))
    plt.show()

def plot_categorical_distributions(df, categorical_cols, group_col, save_dir=None):
    """
    Vẽ bar plot cho các cột phân loại theo nhóm.
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        categorical_cols (list): Danh sách các cột phân loại.
        group_col (str): Cột để phân nhóm (ví dụ: 'has_complication').
        save_dir (str, optional): Thư mục để lưu biểu đồ.
    """
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=col, hue=group_col, data=df)
        plt.title(f"Tần suất {col} theo {group_col}")
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"barplot_{col}.png"))
        plt.show()

def correlation_analysis(df, numerical_cols, group_col, save_dir=None):
    """
    Vẽ ma trận tương quan cho các cột số và cột nhóm.
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        numerical_cols (list): Danh sách các cột số.
        group_col (str): Cột để phân nhóm (ví dụ: 'has_complication').
        save_dir (str, optional): Thư mục để lưu biểu đồ.
    """
    plt.figure(figsize=(10, 8))
    corr_cols = numerical_cols + [group_col]
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Ma trận tương quan")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "correlation_matrix.png"))
    plt.show()

def group_analysis(df, numerical_cols, categorical_cols, group_col):
    """
    Phân tích theo nhóm (trung bình cột số, tỷ lệ cột phân loại).
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        numerical_cols (list): Danh sách các cột số.
        categorical_cols (list): Danh sách các cột phân loại.
        group_col (str): Cột để phân nhóm (ví dụ: 'has_complication').
    """
    print("\nThống kê theo nhóm (không biến chứng vs có biến chứng):")
    print("\nTrung bình các cột số:")
    print(df.groupby(group_col)[numerical_cols].mean())
    
    print("\nTỷ lệ các cột phân loại:")
    for col in categorical_cols:
        print(f"\nTỷ lệ {col} theo {group_col}:")
        print(pd.crosstab(df[col], df[group_col], normalize='index'))

def detect_outliers(df, col):
    """
    Phát hiện ngoại lai trong một cột bằng phương pháp IQR.
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        col (str): Tên cột cần kiểm tra.
    
    Returns:
        pd.DataFrame: DataFrame chứa các mẫu ngoại lai.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"\nNgoại lai trong {col}:\n", outliers[[col, 'has_complication']])
    return outliers

def main():
    """
    Hàm chính để chạy phân tích EDA.
    """
    # Đặt thư mục làm việc
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Thư mục làm việc hiện tại:", os.getcwd())

    # Định nghĩa đường dẫn và cột
    data_path = r'..\data\processed\processed_data.csv'
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    group_col = 'has_complication'

    # Tạo thư mục để lưu biểu đồ
    save_dir = r'..\figures\plots'
    os.makedirs(save_dir, exist_ok=True)

    # Đọc dữ liệu
    df = load_data(data_path)

    # Tạo cột has_complication
    df['has_complication'] = (df['num'] > 0).astype(int)

    # Thống kê mô tả
    descriptive_statistics(df, numerical_cols)

    # Tần suất giá trị
    frequency_analysis(df, categorical_cols + ['num'])

    # Trực quan hóa phân phối số
    plot_numerical_distributions(df, numerical_cols, save_dir)

    # Trực quan hóa phân phối phân loại
    plot_categorical_distributions(df, categorical_cols, group_col, save_dir)

    # Phân tích tương quan
    correlation_analysis(df, numerical_cols, group_col, save_dir)

    # Phân tích theo nhóm
    group_analysis(df, numerical_cols, categorical_cols, group_col)

    # Phát hiện ngoại lai (ví dụ cho chol)
    detect_outliers(df, 'chol')

if __name__ == "__main__":
    # Sử dụng style thay thế hoặc để mặc định
    plt.style.use('seaborn-v0_8')  # Hoặc bỏ dòng này nếu không cần style cụ thể
    main()