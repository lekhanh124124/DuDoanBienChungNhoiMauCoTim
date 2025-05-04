import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import statsmodels.api as sm
from scipy import stats
import joblib

def load_data():
    """
    Đọc dữ liệu từ file CSV và chuẩn bị dữ liệu cho phân tích thuốc.
    
    Returns:
        pd.DataFrame: DataFrame với dữ liệu đã được xử lý.
    """
    # Đặt thư mục làm việc
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Thư mục làm việc hiện tại:", os.getcwd())
    
    # Đường dẫn dữ liệu
    data_path = r'..\data\processed\processed_data.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file {data_path}")
    
    # Đọc dữ liệu
    df = pd.read_csv(data_path)
    
    # Tạo biến mục tiêu 'has_complication' dựa vào cột 'num'
    df['has_complication'] = (df['num'] > 0).astype(int)
    
    print(f"Đã đọc dữ liệu từ {data_path}")
    print(f"Kích thước dữ liệu: {df.shape}")
    
    return df

def create_medication_proxy(df):
    """
    Tạo biến đại diện cho việc sử dụng thuốc dựa vào các đặc trưng hiện có.
    
    Vì dữ liệu không có thông tin trực tiếp về thuốc, chúng ta tạo biến đại diện dựa trên
    các thông số lâm sàng có thể chỉ ra việc sử dụng thuốc.
    
    Parameters:
        df (pd.DataFrame): DataFrame với dữ liệu.
    
    Returns:
        pd.DataFrame: DataFrame với biến đại diện thuốc.
    """
    # Tạo bản sao để không làm thay đổi dữ liệu gốc
    data = df.copy()
    
    # 1. Beta blockers: Thuốc điều trị nhịp tim và huyết áp
    # Bệnh nhân có nhịp tim thấp hơn có thể đang dùng thuốc chẹn beta
    median_thalach = data['thalach'].median()
    data['likely_beta_blockers'] = (data['thalach'] < median_thalach).astype(int)
    
    # 2. Statin: Thuốc hạ cholesterol
    # Bệnh nhân có cholesterol dưới ngưỡng có thể đang dùng statin
    median_chol = data['chol'].median()
    data['likely_statins'] = (data['chol'] < median_chol).astype(int)
    
    # 3. Nitrate: Thuốc điều trị đau thắt ngực
    # Bệnh nhân có tiền sử đau thắt ngực nhưng không đau khi gắng sức có thể đang dùng nitrate
    # Cột cp=1 là đau thắt ngực điển hình
    data['likely_nitrates'] = ((data['cp'] == '1') & (data['exang'] == '0')).astype(int)
    
    # 4. Thuốc chống kết tập tiểu cầu: Dựa vào kết quả ECG và đau thắt ngực
    # Nếu bệnh nhân có ECG bất thường (restecg = 1 hoặc 2) nhưng không có đau thắt ngực khi gắng sức
    data['likely_antiplatelets'] = ((data['restecg'].isin(['1', '2'])) & 
                                   (data['exang'] == '0')).astype(int)
    
    # 5. Tạo biến tổng hợp: Có khả năng đang điều trị bằng thuốc
    data['medication_treatment'] = ((data['likely_beta_blockers'] == 1) | 
                                   (data['likely_statins'] == 1) | 
                                   (data['likely_nitrates'] == 1) | 
                                   (data['likely_antiplatelets'] == 1)).astype(int)
    
    print("\nĐã tạo biến đại diện cho việc sử dụng thuốc:")
    print(f"Có khả năng dùng thuốc chẹn beta: {data['likely_beta_blockers'].sum()} bệnh nhân")
    print(f"Có khả năng dùng statin: {data['likely_statins'].sum()} bệnh nhân")
    print(f"Có khả năng dùng nitrate: {data['likely_nitrates'].sum()} bệnh nhân")
    print(f"Có khả năng dùng thuốc chống kết tập tiểu cầu: {data['likely_antiplatelets'].sum()} bệnh nhân")
    print(f"Tổng số bệnh nhân có khả năng đang dùng thuốc: {data['medication_treatment'].sum()} bệnh nhân")
    
    return data

def analyze_medication_impact(df):
    """
    Phân tích tác động của việc sử dụng thuốc lên biến chứng nhồi máu cơ tim.
    
    Parameters:
        df (pd.DataFrame): DataFrame với dữ liệu và biến đại diện thuốc.
    """
    print("\n=== PHÂN TÍCH TÁC ĐỘNG CỦA THUỐC LÊN BIẾN CHỨNG NHỒI MÁU CƠ TIM ===\n")
    
    medication_types = [
        'likely_beta_blockers', 
        'likely_statins', 
        'likely_nitrates', 
        'likely_antiplatelets',
        'medication_treatment'
    ]
    
    medication_labels = [
        'Thuốc chẹn beta', 
        'Statin', 
        'Nitrate', 
        'Thuốc chống kết tập tiểu cầu',
        'Bất kỳ thuốc nào'
    ]
    
    results = []
    
    for med_var, med_label in zip(medication_types, medication_labels):
        # Tạo bảng chéo
        ct = pd.crosstab(df[med_var], df['has_complication'])
        print(f"\nBảng chéo cho {med_label}:")
        print(ct)
        
        # Tỷ lệ biến chứng - với thuốc
        if 1 in ct.index:
            total_with_med = ct.loc[1].sum() 
            complications_with_med = ct.loc[1, 1] if 1 in ct.columns else 0
            comp_rate_with_med = complications_with_med / total_with_med * 100 if total_with_med > 0 else 0
        else:
            total_with_med = 0
            complications_with_med = 0
            comp_rate_with_med = 0
        
        # Tỷ lệ biến chứng - không thuốc
        if 0 in ct.index:
            total_without_med = ct.loc[0].sum()
            complications_without_med = ct.loc[0, 1] if 1 in ct.columns else 0
            comp_rate_without_med = complications_without_med / total_without_med * 100 if total_without_med > 0 else 0
        else:
            total_without_med = 0
            complications_without_med = 0
            comp_rate_without_med = 0
        
        # Tính odds ratio
        try:
            table = np.array([[complications_with_med, total_with_med - complications_with_med],
                             [complications_without_med, total_without_med - complications_without_med]])
            
            # Kiểm tra để tránh chia cho 0
            if (table[0, 1] * table[1, 0]) > 0:
                odds_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
                
                # Kiểm định Chi-square
                chi2, p, _, _ = stats.chi2_contingency(ct)
                
                # Khoảng tin cậy 95% cho odds ratio
                if odds_ratio > 0:
                    log_odds_ratio = np.log(odds_ratio)
                    # Tránh chia cho 0 trong sqrt
                    se_log_odds_ratio = np.sqrt(sum(1/entry if entry > 0 else 0 for entry in table.flatten()))
                    ci_lower = np.exp(log_odds_ratio - 1.96 * se_log_odds_ratio)
                    ci_upper = np.exp(log_odds_ratio + 1.96 * se_log_odds_ratio)
                else:
                    ci_lower = ci_upper = np.nan
            else:
                odds_ratio = chi2 = p = ci_lower = ci_upper = np.nan
            
        except Exception as e:
            odds_ratio = chi2 = p = ci_lower = ci_upper = np.nan
            print(f"Lỗi khi phân tích {med_label}: {e}")
        
        results.append({
            'Medication': med_label,
            'With Medication': f"{total_with_med} patients",
            'Complication Rate (With Med)': f"{comp_rate_with_med:.1f}%",
            'Without Medication': f"{total_without_med} patients",
            'Complication Rate (Without Med)': f"{comp_rate_without_med:.1f}%",
            'Odds Ratio': odds_ratio,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            'p-value': p if 'p' in locals() else np.nan,
            'Significant': p < 0.05 if 'p' in locals() and not np.isnan(p) else np.nan
        })
    
    # Hiển thị bảng kết quả
    results_df = pd.DataFrame(results)
    print("\nKết quả phân tích tác động của thuốc:")
    print(results_df[['Medication', 'Complication Rate (With Med)', 
                     'Complication Rate (Without Med)', 'Odds Ratio', 
                     'p-value', 'Significant']])
    
    # Lưu kết quả
    os.makedirs(r'..\results', exist_ok=True)
    results_df.to_csv(r'..\results\medication_impact_analysis.csv', index=False)
    print("\nĐã lưu kết quả phân tích vào file: ..\results\medication_impact_analysis.csv")

def analyze_medication_interaction(df):
    """
    Phân tích tương tác giữa việc sử dụng thuốc và các đặc trưng khác.
    
    Parameters:
        df (pd.DataFrame): DataFrame với dữ liệu và biến đại diện thuốc.
    """
    print("\n=== PHÂN TÍCH TƯƠNG TÁC GIỮA THUỐC VÀ CÁC ĐẶC TRƯNG KHÁC ===\n")
    
    # Các đặc trưng quan trọng
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Phân tích cho mỗi loại thuốc
    for med_var in ['likely_beta_blockers', 'likely_statins', 'likely_nitrates', 'likely_antiplatelets']:
        med_label = {
            'likely_beta_blockers': 'Thuốc chẹn beta',
            'likely_statins': 'Statin',
            'likely_nitrates': 'Nitrate',
            'likely_antiplatelets': 'Thuốc chống kết tập tiểu cầu'
        }[med_var]
        
        print(f"\n--- Phân tích cho {med_label} ---\n")
        
        # 1. Phân tích cho các đặc trưng số
        print(f"Tác động của {med_label} lên các đặc trưng số:")
        
        for feature in numerical_features:
            # So sánh giữa nhóm có và không dùng thuốc
            with_med = df[df[med_var] == 1][feature]
            without_med = df[df[med_var] == 0][feature]
            
            # T-test
            t_stat, p_value = stats.ttest_ind(with_med.dropna(), without_med.dropna(), equal_var=False)
            
            print(f"- {feature}: ")
            print(f"  Trung bình ở nhóm dùng thuốc: {with_med.mean():.2f}")
            print(f"  Trung bình ở nhóm không dùng thuốc: {without_med.mean():.2f}")
            print(f"  T-test p-value: {p_value:.4f}")
            print(f"  Có sự khác biệt có ý nghĩa thống kê: {'Có' if p_value < 0.05 else 'Không'}")
        
        # 2. Phân tích tương tác giữa thuốc và biến chứng
        # Tạo bảng chéo chi tiết hơn
        for feature in categorical_features:
            # Tạo bảng chéo 3 chiều: đặc trưng × thuốc × biến chứng
            unique_values = sorted(df[feature].unique())
            
            print(f"\nTương tác giữa {med_label}, {feature} và biến chứng:")
            
            for value in unique_values:
                # Bệnh nhân có đặc trưng này và dùng thuốc
                n11 = len(df[(df[feature] == value) & (df[med_var] == 1)])
                n11_comp = len(df[(df[feature] == value) & (df[med_var] == 1) & (df['has_complication'] == 1)])
                rate11 = n11_comp / n11 * 100 if n11 > 0 else 0
                
                # Bệnh nhân có đặc trưng này và không dùng thuốc
                n10 = len(df[(df[feature] == value) & (df[med_var] == 0)])
                n10_comp = len(df[(df[feature] == value) & (df[med_var] == 0) & (df['has_complication'] == 1)])
                rate10 = n10_comp / n10 * 100 if n10 > 0 else 0
                
                print(f"- Giá trị {feature} = {value}:")
                print(f"  Tỷ lệ biến chứng khi dùng thuốc: {rate11:.1f}% ({n11_comp}/{n11})")
                print(f"  Tỷ lệ biến chứng khi không dùng thuốc: {rate10:.1f}% ({n10_comp}/{n10})")
                
                # Tính tỷ lệ giảm biến chứng
                if rate10 > 0:
                    reduction = (rate10 - rate11) / rate10 * 100
                    print(f"  Giảm tỷ lệ biến chứng: {reduction:.1f}%")
                else:
                    print("  Không thể tính tỷ lệ giảm biến chứng")

def create_visualizations(df):
    """
    Tạo các biểu đồ trực quan về tác động của thuốc lên biến chứng.
    
    Parameters:
        df (pd.DataFrame): DataFrame với dữ liệu và biến đại diện thuốc.
    """
    print("\n=== TẠO BIỂU ĐỒ TRỰC QUAN VỀ TÁC ĐỘNG CỦA THUỐC ===\n")
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(r'..\figures\medication_impact', exist_ok=True)
    
    # Định nghĩa các đặc trưng số và phân loại
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # 1. Biểu đồ tỷ lệ biến chứng theo từng nhóm thuốc
    plt.figure(figsize=(14, 8))
    
    medication_types = [
        'likely_beta_blockers', 
        'likely_statins', 
        'likely_nitrates', 
        'likely_antiplatelets'
    ]
    
    medication_labels = [
        'Thuốc chẹn beta', 
        'Statin', 
        'Nitrate', 
        'Thuốc chống kết tập tiểu cầu'
    ]
    
    # Tính tỷ lệ biến chứng
    comp_rates = []
    
    for med_var in medication_types:
        # Với thuốc
        with_med = df[df[med_var] == 1]
        rate_with = with_med['has_complication'].mean() * 100
        
        # Không có thuốc
        without_med = df[df[med_var] == 0]
        rate_without = without_med['has_complication'].mean() * 100
        
        comp_rates.append((rate_with, rate_without))
    
    # Vẽ biểu đồ cột ghép
    x = np.arange(len(medication_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, [rate[0] for rate in comp_rates], width, label='Có thuốc')
    rects2 = ax.bar(x + width/2, [rate[1] for rate in comp_rates], width, label='Không có thuốc')
    
    ax.set_xlabel('Loại thuốc')
    ax.set_ylabel('Tỷ lệ biến chứng (%)')
    ax.set_title('Tỷ lệ biến chứng nhồi máu cơ tim theo loại thuốc')
    ax.set_xticks(x)
    ax.set_xticklabels(medication_labels)
    ax.legend()
    
    # Thêm nhãn giá trị lên các cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(r'..\figures\medication_impact\complication_rate_by_medication.png')
    print("Đã lưu biểu đồ tỷ lệ biến chứng theo loại thuốc")
    
    # 2. Biểu đồ odds ratio cho từng loại thuốc
    plt.figure(figsize=(10, 6))
    
    odds_ratios = []
    ci_lower = []
    ci_upper = []
    
    for med_var in medication_types:
        # Tạo bảng chéo
        ct = pd.crosstab(df[med_var], df['has_complication'])
        
        # Tính odds ratio
        try:
            a = ct.loc[1, 1] if 1 in ct.index and 1 in ct.columns else 0
            b = ct.loc[1, 0] if 1 in ct.index and 0 in ct.columns else 0
            c = ct.loc[0, 1] if 0 in ct.index and 1 in ct.columns else 0
            d = ct.loc[0, 0] if 0 in ct.index and 0 in ct.columns else 0
            
            odds_ratio = (a * d) / (b * c) if b * c > 0 else np.nan
            
            # Khoảng tin cậy
            log_or = np.log(odds_ratio) if not np.isnan(odds_ratio) and odds_ratio > 0 else np.nan
            se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if a*b*c*d > 0 else np.nan
            
            lower = np.exp(log_or - 1.96 * se_log_or) if not np.isnan(log_or) and not np.isnan(se_log_or) else np.nan
            upper = np.exp(log_or + 1.96 * se_log_or) if not np.isnan(log_or) and not np.isnan(se_log_or) else np.nan
            
            odds_ratios.append(odds_ratio)
            ci_lower.append(lower)
            ci_upper.append(upper)
        except:
            odds_ratios.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 7))
    plt.errorbar(medication_labels, odds_ratios, yerr=[
        [or_val - lower for or_val, lower in zip(odds_ratios, ci_lower)],
        [upper - or_val for or_val, upper in zip(odds_ratios, ci_upper)]
    ], fmt='o')
    
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
    plt.title('Odds Ratio cho tác động của thuốc lên biến chứng nhồi máu cơ tim')
    plt.ylabel('Odds Ratio (95% CI)')
    plt.xticks(rotation=15)
    plt.ylim(0, max([upper for upper in ci_upper if not np.isnan(upper)]) * 1.1 if any(not np.isnan(val) for val in ci_upper) else 5)
    
    plt.tight_layout()
    plt.savefig(r'..\figures\medication_impact\medication_odds_ratio.png')
    print("Đã lưu biểu đồ odds ratio cho tác động của thuốc")
    
    # 3. Biểu đồ tương tác giữa thuốc và các yếu tố nguy cơ
    # Chọn một số yếu tố nguy cơ quan trọng
    risk_factors = ['age', 'sex', 'cp', 'trestbps', 'chol']
    
    for med_var, med_label in zip(medication_types, medication_labels):
        plt.figure(figsize=(14, 10))
        
        # Tạo biểu đồ con
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for i, feature in enumerate(risk_factors):
            if i < len(axes):
                ax = axes[i]
                
                if feature in numerical_features:  # Đặc trưng số
                    # Chia thành nhóm dựa vào median
                    median_val = df[feature].median()
                    df[f'{feature}_group'] = (df[feature] > median_val).map({True: 'Cao', False: 'Thấp'})
                    
                    # Tỷ lệ biến chứng theo nhóm và thuốc
                    groups = df.groupby([f'{feature}_group', med_var])['has_complication'].mean() * 100
                    
                    # Tạo DataFrame để dễ vẽ biểu đồ
                    plot_data = []
                    for name, group in df.groupby([f'{feature}_group', med_var]):
                        feature_group, med_group = name
                        comp_rate = group['has_complication'].mean() * 100
                        n_patients = len(group)
                        med_status = "Có thuốc" if med_group == 1 else "Không có thuốc"
                        plot_data.append([feature_group, med_status, comp_rate, n_patients])
                    
                    plot_df = pd.DataFrame(plot_data, columns=['Risk Group', 'Medication Status', 'Complication Rate', 'Patients'])
                    
                    # Vẽ biểu đồ cột
                    sns.barplot(x='Risk Group', y='Complication Rate', hue='Medication Status', data=plot_df, ax=ax)
                    
                    # Xóa biến tạm
                    df.drop(columns=[f'{feature}_group'], inplace=True)
                    
                else:  # Đặc trưng phân loại
                    # Chọn tối đa 4 giá trị phổ biến nhất
                    top_values = df[feature].value_counts().nlargest(4).index.tolist()
                    feature_data = df[df[feature].isin(top_values)]
                    
                    # Tính tỷ lệ biến chứng
                    plot_data = []
                    for name, group in feature_data.groupby([feature, med_var]):
                        feature_val, med_group = name
                        comp_rate = group['has_complication'].mean() * 100
                        n_patients = len(group)
                        med_status = "Có thuốc" if med_group == 1 else "Không có thuốc"
                        plot_data.append([feature_val, med_status, comp_rate, n_patients])
                    
                    plot_df = pd.DataFrame(plot_data, columns=['Feature Value', 'Medication Status', 'Complication Rate', 'Patients'])
                    
                    # Vẽ biểu đồ cột
                    sns.barplot(x='Feature Value', y='Complication Rate', hue='Medication Status', data=plot_df, ax=ax)
                
                ax.set_title(f'Tác động của {med_label} theo {feature}')
                ax.set_ylabel('Tỷ lệ biến chứng (%)')
                
                # Thêm giá trị lên cột
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.1f}%", 
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha = 'center', va = 'bottom',
                              xytext = (0, 5), textcoords = 'offset points')
        
        # Ẩn các biểu đồ con còn lại nếu có
        for j in range(len(risk_factors), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(fr'..\figures\medication_impact\{med_var}_interaction.png')
        print(f"Đã lưu biểu đồ tương tác cho {med_label}")

def main():
    # 1. Đọc dữ liệu
    df = load_data()
    
    # 2. Tạo biến đại diện cho việc sử dụng thuốc
    df_with_med = create_medication_proxy(df)
    
    # 3. Phân tích tác động của thuốc
    analyze_medication_impact(df_with_med)
    
    # 4. Phân tích tương tác giữa thuốc và các đặc trưng
    analyze_medication_interaction(df_with_med)
    
    # 5. Tạo biểu đồ trực quan
    create_visualizations(df_with_med)
    
    print("\nHoàn thành phân tích tác động của thuốc lên biến chứng nhồi máu cơ tim.")

if __name__ == "__main__":
    main()