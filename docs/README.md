# Dùng để tải các thư viện cần thiết
pip install -r docs/requirements.txt

# /data/ là nơi chứa bộ dữ liệu
    -   /raw chứa bộ dữ liệu lấy từ heart+disease.zip
    -   /processed chứa bộ dữ liệu được gộp và tiền xử lí từ /raw

# /src/ là nơi chứa code cho các nhiệm vụ chính

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