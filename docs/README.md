# Dùng để tải các thư viện cần thiết
pip install -r docs/requirements.txt

# Dự đoán biến chứng nhồi máu cơ tim

Dự án này tập trung vào phân tích và xây dựng mô hình dự đoán biến chứng nhồi máu cơ tim dựa trên bộ dữ liệu UCI Heart Disease.

## Mô tả dữ liệu

Bộ dữ liệu được sử dụng là **UCI Heart Disease Dataset** (processed.cleveland.data), chứa 303 mẫu với 14 thuộc tính liên quan đến các yếu tố sức khỏe và nguy cơ nhồi máu cơ tim. Dưới đây là mô tả ý nghĩa và kiểu dữ liệu của từng cột:

| Tên cột      | Ý nghĩa                                                                 | Kiểu dữ liệu | Giá trị |
|--------------|-------------------------------------------------------------------------|--------------|---------|
| `age`        | Tuổi của bệnh nhân (tính bằng năm).                                     | Số thực      | Liên tục (29-77) |
| `sex`        | Giới tính của bệnh nhân.                                                | Phân loại    | 0 = nữ, 1 = nam |
| `cp`         | Loại đau ngực (chest pain type).                                        | Phân loại    | 1 = đau thắt ngực điển hình, 2 = đau thắt ngực không điển hình, 3 = không đau thắt ngực, 4 = không có triệu chứng |
| `trestbps`   | Huyết áp tâm trương lúc nghỉ (resting blood pressure, mm Hg).           | Số thực      | Liên tục (94-200) |
| `chol`       | Mức cholesterol huyết thanh (serum cholesterol, mg/dl).                 | Số thực      | Liên tục (126-564) |
| `fbs`        | Đường huyết lúc đói (fasting blood sugar > 120 mg/dl).                  | Phân loại    | 0 = không, 1 = có |
| `restecg`    | Kết quả điện tâm đồ lúc nghỉ (resting electrocardiographic results).    | Phân loại    | 0 = bình thường, 1 = có bất thường sóng ST-T, 2 = phì đại thất trái |
| `thalach`    | Nhịp tim tối đa đạt được (maximum heart rate achieved).                 | Số thực      | Liên tục (71-202) |
| `exang`      | Đau thắt ngực do tập thể dục (exercise induced angina).                 | Phân loại    | 0 = không, 1 = có |
| `oldpeak`    | Độ chênh ST do tập thể dục so với lúc nghỉ (ST depression).             | Số thực      | Liên tục (0-6.2) |
| `slope`      | Độ dốc của đoạn ST khi tập thể dục (slope of peak exercise ST segment). | Phân loại    | 1 = dốc lên, 2 = phẳng, 3 = dốc xuống |
| `ca`         | Số lượng mạch máu chính được nhuộm màu (number of major vessels).       | Phân loại    | 0-3 (có thể thiếu) |
| `thal`       | Kết quả kiểm tra thalassemia (thalassemia test).                        | Phân loại    | 3 = bình thường, 6 = khuyết tật cố định, 7 = khuyết tật có thể đảo ngược (có thể thiếu) |
| `target`     | Chẩn đoán bệnh tim (biến chứng nhồi máu cơ tim).                         | Phân loại    | 0 = không có bệnh, 1-4 = có bệnh (độ nghiêm trọng tăng dần) |

### Ghi chú
- Một số cột như `ca` và `thal` có thể chứa giá trị thiếu (missing values), được biểu thị bằng "?" trong dữ liệu gốc.
- Cột `target` thường được chuyển đổi thành nhị phân (0 = không có bệnh, 1 = có bệnh) để phù hợp với bài toán phân lớp nhị phân trong dự án này.
- Các cột phân loại (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) sẽ được mã hóa (encoding) trong quá trình tiền xử lý dữ liệu.
- Các cột số (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) sẽ được chuẩn hóa (scaling) để đảm bảo tính đồng nhất.


# /data/ là nơi chứa bộ dữ liệu
    -   /raw chứa bộ dữ liệu lấy từ heart+disease.zip
    -   /processed chứa bộ dữ liệu được gộp và tiền xử lí từ /raw

# /src/ là nơi chứa code cho các nhiệm vụ chính
    -   data_preprocessing.py: mục đích là xử lý dữ liệu thiếu và tổng hợp các dữ liệu lại

# /reports/ là nơi chứa word và ppt đã được tổng hợp và sẽ dùng để trình bày.

# /figures/ là nơi chứa các hình ảnh được tạo từ code.
    - correlation_heatmap: Biểu đồ heatmap tương quan giữa các biến.
    - model_performance: Biểu đồ ROC-AUC hoặc confusion matrix của các mô hình.
    - risk_factor_analysis: Biểu đồ phân tích yếu tố nguy cơ (ví dụ: boxplot tuổi/giới tính).
    - bla bla

# /models/ Chứa các mô hình đã huấn luyện (tùy chọn).
    - logistic_regression_model: Mô hình Logistic Regression lưu bằng joblib/pickle.
    - svm_model: Mô hình SVM.
    - random_forest_model: Mô hình Random Forest.