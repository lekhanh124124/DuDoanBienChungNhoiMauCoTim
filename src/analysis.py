#Phân tích yếu tố nguy cơ và mối quan hệ tiền sử bệnh lý với biến chứng nhồi máu cơ tim

import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV và tạo biến mục tiêu.
    
    Parameters:
        file_path (str): Đường dẫn đến file dữ liệu.
    
    Returns:
        pd.DataFrame: DataFrame với dữ liệu đã được xử lý.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file {file_path}")
    
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Tạo biến mục tiêu 'has_complication' dựa vào cột 'num'
    df['has_complication'] = (df['num'] > 0).astype(int)
    
    print(f"Đã đọc dữ liệu từ {file_path}")
    print(f"Kích thước dữ liệu: {df.shape}")
    print(f"Phân bố nhóm:\n{df['has_complication'].value_counts()}")
    
    return df

def risk_factor_odds_ratio(df, categorical_features, numerical_features):
    """
    Phân tích tỉ số odds (odds ratio) cho các yếu tố nguy cơ.
    
    Parameters:
        df (pd.DataFrame): DataFrame với dữ liệu.
        categorical_features (list): Danh sách các đặc trưng phân loại.
        numerical_features (list): Danh sách các đặc trưng số.
    """
    print("\n=== PHÂN TÍCH TỈ SỐ ODDS (ODDS RATIO) CHO CÁC YẾU TỐ NGUY CƠ ===\n")
    
    # Phân tích odds ratio cho biến phân loại
    results_cat = []
    
    for feature in categorical_features:
        contingency_tables = []
        unique_values = sorted(df[feature].unique())
        
        # Tạo bảng chéo (contingency table)
        for value in unique_values:
            # Bảng 2x2 cho mỗi giá trị của đặc trưng
            has_feature_has_comp = len(df[(df[feature] == value) & (df['has_complication'] == 1)])
            has_feature_no_comp = len(df[(df[feature] == value) & (df['has_complication'] == 0)])
            no_feature_has_comp = len(df[(df[feature] != value) & (df['has_complication'] == 1)])
            no_feature_no_comp = len(df[(df[feature] != value) & (df['has_complication'] == 0)])
            
            # Tạo bảng chéo
            table = np.array([[has_feature_has_comp, has_feature_no_comp],
                              [no_feature_has_comp, no_feature_no_comp]])
            
            # Tính odds ratio
            try:
                odds_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
                # Kiểm định Chi-square
                chi2, p, _, _ = stats.chi2_contingency(table)
                
                # Khoảng tin cậy 95% cho odds ratio
                log_odds_ratio = np.log(odds_ratio)
                se_log_odds_ratio = np.sqrt(sum(1/table.flatten()))
                ci_lower = np.exp(log_odds_ratio - 1.96 * se_log_odds_ratio)
                ci_upper = np.exp(log_odds_ratio + 1.96 * se_log_odds_ratio)
                
                # Thêm vào kết quả
                results_cat.append({
                    'Feature': feature,
                    'Value': value,
                    'Odds Ratio': odds_ratio,
                    'CI Lower': ci_lower,
                    'CI Upper': ci_upper,
                    'p-value': p,
                    'Significant': p < 0.05
                })
            except:
                # Xử lý trường hợp chia cho 0
                continue
    
    # Hiển thị kết quả
    results_cat_df = pd.DataFrame(results_cat)
    if not results_cat_df.empty:
        print("Kết quả phân tích odds ratio cho các đặc trưng phân loại:")
        print(results_cat_df.sort_values('p-value'))
    
    # Phân tích cho các đặc trưng số
    results_num = []
    
    for feature in numerical_features:
        # Tách thành nhóm dưới và trên trung vị
        median_val = df[feature].median()
        df[f'{feature}_high'] = (df[feature] > median_val).astype(int)
        
        # Bảng 2x2
        high_has_comp = len(df[(df[f'{feature}_high'] == 1) & (df['has_complication'] == 1)])
        high_no_comp = len(df[(df[f'{feature}_high'] == 1) & (df['has_complication'] == 0)])
        low_has_comp = len(df[(df[f'{feature}_high'] == 0) & (df['has_complication'] == 1)])
        low_no_comp = len(df[(df[f'{feature}_high'] == 0) & (df['has_complication'] == 0)])
        
        table = np.array([[high_has_comp, high_no_comp],
                         [low_has_comp, low_no_comp]])
        
        # Tính odds ratio
        try:
            odds_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
            # Kiểm định Chi-square
            chi2, p, _, _ = stats.chi2_contingency(table)
            
            # Khoảng tin cậy 95%
            log_odds_ratio = np.log(odds_ratio)
            se_log_odds_ratio = np.sqrt(sum(1/table.flatten()))
            ci_lower = np.exp(log_odds_ratio - 1.96 * se_log_odds_ratio)
            ci_upper = np.exp(log_odds_ratio + 1.96 * se_log_odds_ratio)
            
            # Thêm vào kết quả
            results_num.append({
                'Feature': feature,
                'Threshold': f'> {median_val:.2f}',
                'Odds Ratio': odds_ratio,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'p-value': p,
                'Significant': p < 0.05
            })
            
            # Xoá cột tạm
            df.drop(columns=[f'{feature}_high'], inplace=True)
        except:
            # Xử lý trường hợp chia cho 0
            df.drop(columns=[f'{feature}_high'], inplace=True)
            continue
    
    # Hiển thị kết quả
    results_num_df = pd.DataFrame(results_num)
    if not results_num_df.empty:
        print("\nKết quả phân tích odds ratio cho các đặc trưng số:")
        print(results_num_df.sort_values('p-value'))
    
    # Phân tích đa biến (logistic regression)
    print("\n=== PHÂN TÍCH ĐA BIẾN (LOGISTIC REGRESSION) ===\n")
    
    # Chuẩn bị dữ liệu
    X = pd.get_dummies(df[categorical_features + numerical_features], drop_first=True)
    y = df['has_complication']
    
    # Thêm hằng số cho mô hình
    X = sm.add_constant(X)
    
    # Xây dựng mô hình
    model = sm.Logit(y, X)
    
    try:
        result = model.fit(disp=0)  # disp=0 để không hiển thị thông báo tối ưu
        
        # In kết quả
        print(result.summary())
        
        # Trích xuất odds ratio từ kết quả
        odds_ratios = pd.DataFrame({
            'Odds Ratio': np.exp(result.params),
            'CI Lower': np.exp(result.params - 1.96 * result.bse),
            'CI Upper': np.exp(result.params + 1.96 * result.bse),
            'p-value': result.pvalues,
            'Significant': result.pvalues < 0.05
        })
        
        print("\nOdds Ratio từ mô hình đa biến:")
        print(odds_ratios.sort_values('p-value'))
        
    except Exception as e:
        print(f"Lỗi khi xây dựng mô hình đa biến: {e}")

def analyze_medical_history_relationship(df):
    """
    Phân tích mối quan hệ giữa tiền sử bệnh lý và biến chứng.
    
    Parameters:
        df (pd.DataFrame): DataFrame với dữ liệu.
    """
    print("\n=== PHÂN TÍCH MỐI QUAN HỆ GIỮA TIỀN SỬ BỆNH LÝ VÀ BIẾN CHỨNG ===\n")
    
    # Các biến liên quan đến tiền sử bệnh lý
    medical_history_vars = {
        'cp': 'Đau thắt ngực',
        'fbs': 'Đường huyết lúc đói > 120 mg/dl',
        'restecg': 'Kết quả ECG lúc nghỉ',
        'exang': 'Đau thắt ngực khi tập thể dục',
        'slope': 'Độ dốc phân đoạn ST khi tập thể dục',
        'ca': 'Số mạch máu chính',
        'thal': 'Thalassemia'
    }
    
    # Phân tích tỷ lệ biến chứng theo từng biến tiền sử
    for var, label in medical_history_vars.items():
        # Tạo bảng chéo
        ct = pd.crosstab(df[var], df['has_complication'])
        
        # Tính tỷ lệ phần trăm
        ct_percent = ct.div(ct.sum(axis=1), axis=0) * 100
        
        # Kiểm định Chi-square
        chi2, p, _, _ = stats.chi2_contingency(ct)
        
        print(f"\nMối quan hệ giữa {label} và biến chứng:")
        print(f"Chi-square: {chi2:.4f}, p-value: {p:.4f}")
        print(f"Có ý nghĩa thống kê: {'Có' if p < 0.05 else 'Không'}")
        
        # In ra bảng chéo
        print("\nBảng chéo (số lượng):")
        print(ct)
        print("\nTỷ lệ biến chứng (%):")
        print(ct_percent[1].sort_values(ascending=False))
    
    # Phân tích mối quan hệ giữa số lượng yếu tố tiền sử và tỷ lệ biến chứng
    # Tạo biến đếm số lượng yếu tố nguy cơ từ tiền sử
    risk_factors = []
    
    # Đếm yếu tố nguy cơ dựa trên giá trị đã biết
    df['risk_count'] = 0
    
    if 'fbs' in df.columns:
        df.loc[df['fbs'] == '1', 'risk_count'] += 1  # Đường huyết cao
    
    if 'exang' in df.columns:
        df.loc[df['exang'] == '1', 'risk_count'] += 1  # Đau thắt ngực khi tập thể dục
    
    if 'cp' in df.columns:
        df.loc[df['cp'] == '2', 'risk_count'] += 1  # Đau thắt ngực không ổn định
    
    if 'thal' in df.columns:
        df.loc[df['thal'].isin(['6', '7']), 'risk_count'] += 1  # Bất thường thalassemia
    
    if 'ca' in df.columns:
        df.loc[df['ca'] != '0', 'risk_count'] += 1  # Có mạch máu bị tắc nghẽn
    
    risk_complication = df.groupby('risk_count')['has_complication'].mean() * 100
    
    print("\nTỷ lệ biến chứng (%) theo số lượng yếu tố nguy cơ:")
    print(risk_complication)
    
    # Đánh giá xu hướng
    if len(risk_complication) > 2:
        correlation, p_value = stats.pearsonr(risk_complication.index, risk_complication.values)
        print(f"\nMối tương quan giữa số lượng yếu tố nguy cơ và tỷ lệ biến chứng:")
        print(f"Hệ số tương quan: {correlation:.4f}, p-value: {p_value:.4f}")
        print(f"Kết luận: {'Có' if p_value < 0.05 else 'Không'} có mối tương quan có ý nghĩa thống kê")

def analyze_group_differences(df, numerical_features, categorical_features):
    """
    Phân tích sự khác biệt giữa nhóm có và không có biến chứng.
    
    Parameters:
        df (pd.DataFrame): DataFrame với dữ liệu.
        numerical_features (list): Danh sách các đặc trưng số.
        categorical_features (list): Danh sách các đặc trưng phân loại.
    """
    print("\n=== PHÂN TÍCH SỰ KHÁC BIỆT GIỮA NHÓM CÓ VÀ KHÔNG CÓ BIẾN CHỨNG ===\n")
    
    # 1. Phân tích cho các đặc trưng số
    print("Phân tích cho các đặc trưng số:")
    
    for feature in numerical_features:
        # Phân tích thống kê cơ bản cho từng nhóm
        group_stats = df.groupby('has_complication')[feature].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print(f"\nThống kê cho {feature} theo nhóm:")
        print(group_stats)
        
        # Kiểm định t-test độc lập
        group0 = df[df['has_complication'] == 0][feature].dropna()
        group1 = df[df['has_complication'] == 1][feature].dropna()
        
        t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)  # Welch's t-test không giả định phương sai bằng nhau
        print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"Kết luận: {'Có' if p_value < 0.05 else 'Không'} sự khác biệt có ý nghĩa thống kê")
        
        # Tính ROC AUC để đánh giá khả năng phân loại của đặc trưng
        fpr, tpr, thresholds = roc_curve(df['has_complication'], df[feature])
        roc_auc = auc(fpr, tpr)
        print(f"AUC cho {feature}: {roc_auc:.4f}")
    
    # 2. Phân tích cho các đặc trưng phân loại
    print("\nPhân tích cho các đặc trưng phân loại:")
    
    for feature in categorical_features:
        # Tạo bảng chéo
        ct = pd.crosstab(df[feature], df['has_complication'])
        print(f"\nBảng chéo cho {feature}:")
        print(ct)
        
        # Tính tỷ lệ theo hàng
        ct_percent = ct.div(ct.sum(axis=1), axis=0) * 100
        print(f"\nTỷ lệ % theo hàng:")
        print(ct_percent)
        
        # Kiểm định Chi-square
        chi2, p_value, dof, expected = stats.chi2_contingency(ct)
        print(f"Chi-square test: chi2={chi2:.4f}, p={p_value:.4f}, dof={dof}")
        print(f"Kết luận: {'Có' if p_value < 0.05 else 'Không'} mối liên hệ có ý nghĩa thống kê")
    
    # 3. Phân tích đa biến (ROC Curve trên nhiều đặc trưng)
    print("\nPhân tích đa biến (ROC Curve):")
    
    # Chuẩn bị dữ liệu
    X = pd.get_dummies(df[categorical_features + numerical_features], drop_first=True)
    y = df['has_complication']
    
    # Thêm hằng số cho mô hình
    X = sm.add_constant(X)
    
    # Xây dựng mô hình
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        
        # Dự đoán xác suất
        y_pred_prob = result.predict(X)
        
        # Tính ROC AUC
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        print(f"AUC cho mô hình đa biến: {roc_auc:.4f}")
    except Exception as e:
        print(f"Lỗi khi xây dựng mô hình đa biến: {e}")

def main():
    # Đặt thư mục làm việc
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Thư mục làm việc hiện tại:", os.getcwd())
    
    # Định nghĩa đường dẫn và đặc trưng
    data_path = r'..\data\processed\processed_data.csv'
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Đọc dữ liệu
    df = load_data(data_path)
    
    # 1. Phân tích yếu tố nguy cơ
    risk_factor_odds_ratio(df, categorical_features, numerical_features)
    
    # 2. Phân tích mối quan hệ giữa tiền sử bệnh lý và biến chứng
    analyze_medical_history_relationship(df)
    
    # 3. Phân tích sự khác biệt giữa nhóm có và không có biến chứng
    analyze_group_differences(df, numerical_features, categorical_features)

if __name__ == "__main__":
    # Bắt đầu phân tích
    main()