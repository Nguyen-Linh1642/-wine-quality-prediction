# 🍷 WINE QUALITY PREDICTION – NHÓM 17

## 1. Giới thiệu đề tài

**Tên đề tài:** Dự đoán điểm chất lượng rượu vang (Wine Quality Prediction).

**Lĩnh vực:** Machine Learning – Học máy.

**Ngôn ngữ lập trình:** Python.

**Môi trường phát triển:** Visual Studio Code (VS Code).

**Ứng dụng:** Streamlit.

Dự án xây dựng mô hình Machine Learning nhằm dự đoán điểm chất lượng rượu vang đỏ dựa trên 11 đặc trưng hóa học của rượu.

Hệ thống cho phép người dùng nhập các thông số hóa học, sau đó mô hình học máy phân tích và đưa ra điểm chất lượng dự đoán.

## 2. Mục tiêu dự án

- Tìm hiểu và ứng dụng các thuật toán Machine Learning vào bài toán hồi quy.
- Phân tích và tiền xử lý dữ liệu rượu vang.
- Xây dựng và so sánh nhiều mô hình dự đoán.
- Đánh giá hiệu suất mô hình bằng các chỉ số MAE, RMSE và R².
- Tối ưu hóa mô hình Random Forest.
- Xây dựng ứng dụng web dự đoán chất lượng rượu vang bằng Streamlit.

## 3. Bộ dữ liệu

**Tên dữ liệu:** Wine Quality – Red Wine.

**Nguồn dữ liệu:** UCI Machine Learning Repository.

Link: https://archive.ics.uci.edu/dataset/186/wine+quality

Bộ dữ liệu bao gồm 11 đặc trưng đầu vào và một biến mục tiêu `quality`.

### 11 đặc trưng đầu vào

| STT | Đặc trưng | Ý nghĩa |
|---|---|---|
| 1 | fixed acidity | Độ axit cố định |
| 2 | volatile acidity | Độ axit bay hơi |
| 3 | citric acid | Axit citric |
| 4 | residual sugar | Đường dư |
| 5 | chlorides | Hàm lượng clorua |
| 6 | free sulfur dioxide | Lưu huỳnh dioxide tự do |
| 7 | total sulfur dioxide | Tổng lưu huỳnh dioxide |
| 8 | density | Khối lượng riêng |
| 9 | pH | Độ pH |
| 10 | sulphates | Hàm lượng sulphates |
| 11 | alcohol | Nồng độ cồn |

**Biến mục tiêu:** `quality` – Điểm chất lượng rượu vang.

## 4. Các thuật toán sử dụng

Dự án triển khai và so sánh các mô hình:

1. Dummy Regressor – Mô hình cơ sở.
2. K-Nearest Neighbors (KNN) Regression.
3. Decision Tree Regression.
4. Random Forest Regression.

Sau quá trình so sánh, Random Forest được lựa chọn làm mô hình chính để phát triển ứng dụng.

Dự án tiếp tục sử dụng RandomizedSearchCV kết hợp 5-Fold Cross Validation để tìm kiếm bộ tham số phù hợp cho Random Forest.

## 5. Kết quả thực nghiệm

### So sánh các mô hình ban đầu

| Mô hình | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 0.7049 | 0.8443 | -0.0063 |
| KNN | 0.5397 | 0.7129 | 0.2825 |
| Decision Tree | 0.5515 | 0.7760 | 0.1498 |
| Random Forest | **0.4682** | **0.6145** | **0.4670** |

Random Forest đạt kết quả tốt nhất trong bốn mô hình thử nghiệm ban đầu.

### Kết quả sau tối ưu Random Forest

- MAE: 0.4699
- RMSE: 0.6181
- R²: 0.4607

Kết quả cho thấy mô hình sau tối ưu chưa cải thiện so với Random Forest ban đầu. Vì vậy, dự án lựa chọn Random Forest ban đầu làm mô hình chính.

## 6. Công nghệ sử dụng

| Công nghệ | Chức năng |
|---|---|
| Python | Ngôn ngữ lập trình |
| Pandas | Xử lý dữ liệu |
| NumPy | Tính toán số học |
| Matplotlib | Trực quan hóa dữ liệu |
| Scikit-learn | Xây dựng và đánh giá mô hình |
| Joblib | Lưu và tải mô hình |
| Streamlit | Xây dựng giao diện ứng dụng |
| VS Code | Môi trường phát triển |

## 7. Cấu trúc thư mục dự án

```text
BTL/
│
├── data/
│   └── winequality-red.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── random_forest_tuned.pkl
│   └── ...
│
├── reports/
│   ├── model_comparison.csv
│   ├── tuned_model_result.csv
│   ├── prediction_errors.csv
│   ├── worst_predictions.csv
│   ├── feature_importance.csv
│   └── các biểu đồ đánh giá
│
├── src/
│   ├── evaluate_model.py
│   ├── optimize_model.py
│   └── các file xử lý và huấn luyện
│
├── app.py
├── README.md
└── requirements.txt
```

*Lưu ý: Cấu trúc trên mô tả các thành phần chính của dự án; hãy đối chiếu với các file thực tế trước khi đưa lên GitHub.*

## 8. Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Cài đặt Python

Tải Python tại:

https://www.python.org/downloads/

### Bước 2: Clone dự án

```bash
git clone <GITHUB_REPOSITORY_URL>
cd BTL
```

### Bước 3: Tạo môi trường ảo

Windows:

```powershell
python -m venv .venv
```

Kích hoạt môi trường:

```powershell
.venv\Scripts\activate
```

### Bước 4: Cài đặt thư viện

```powershell
python -m pip install -r requirements.txt
```

### Bước 5: Chạy ứng dụng

```powershell
streamlit run app.py
```

### Bước 6: Truy cập website

Mở trình duyệt tại:

http://localhost:8501

## 9. Chức năng ứng dụng

Ứng dụng Streamlit cung cấp các chức năng:

- Nhập 11 thông số hóa học của rượu vang.
- Điều chỉnh các thông số đầu vào.
- Thực hiện dự đoán bằng mô hình Random Forest.
- Hiển thị điểm chất lượng dự đoán.
- Hiển thị đánh giá chất lượng tham khảo.
- Giới thiệu thông tin mô hình Machine Learning.

## 10. Hạn chế và hướng phát triển

### Hạn chế

- Hiệu suất dự đoán của mô hình vẫn còn khả năng cải thiện.
- Dữ liệu chất lượng rượu vang chưa phân bố đồng đều giữa các mức điểm.
- Mô hình còn gặp khó khăn khi dự đoán các mẫu có chất lượng cực thấp hoặc cực cao.

### Hướng phát triển

- Thử nghiệm thêm XGBoost, Gradient Boosting và các thuật toán hồi quy khác.
- Tiếp tục tối ưu tham số mô hình.
- Cải thiện khả năng dự đoán đối với các nhóm dữ liệu ít mẫu.
- Phát triển giao diện trực quan và thân thiện hơn.
- Triển khai ứng dụng lên nền tảng trực tuyến.

## 11. Kết luận

Dự án đã xây dựng thành công hệ thống dự đoán chất lượng rượu vang sử dụng các thuật toán Machine Learning.

Thông qua quá trình thực nghiệm, Random Forest Regression đạt hiệu suất tốt nhất trong bốn mô hình ban đầu với MAE = 0.4682, RMSE = 0.6145 và R² = 0.4670.

Ứng dụng Streamlit cho phép người dùng nhập các thông số hóa học và nhận kết quả dự đoán trực tiếp.

Dự án giúp nhóm vận dụng kiến thức về xử lý dữ liệu, học máy, đánh giá mô hình và phát triển ứng dụng thực tế.

---

**PROJECT: WINE QUALITY PREDICTION**

**GROUP: 17**

**TECHNOLOGY: PYTHON – MACHINE LEARNING – STREAMLIT**
