
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

# =====================================
# 1. DUONG DAN
# =====================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_PATH = ROOT / "models" / "random_forest.pkl"

# =====================================
# 2. CAU HINH GIAO DIEN
# =====================================

st.set_page_config(
    page_title="Wine Quality Prediction - Group 17",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 WINE QUALITY PREDICTION")
st.subheader("NHÓM 17 - DỰ ĐOÁN CHẤT LƯỢNG RƯỢU VANG")

st.write(
    "Ứng dụng sử dụng thuật toán Random Forest "
    "để dự đoán chất lượng rượu vang dựa trên 11 thông số hóa học."
)

# =====================================
# 3. TAI DU LIEU VA MO HINH
# =====================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")


try:
    model = load_model()
    df = load_data()

except Exception as e:
    st.error(f"Không thể tải dữ liệu hoặc mô hình: {e}")
    st.stop()

features = df.drop(columns=["quality"]).columns.tolist()

# =====================================
# 4. KIEM TRA DU LIEU DAU VAO
# =====================================

def validate_input(input_df, training_df):

    errors = []

    # Kiem tra du lieu thieu
    if input_df.isnull().any().any():
        errors.append("Dữ liệu đầu vào đang bị thiếu.")

    # Kiem tra gia tri khong phai so huu han
    numeric_values = input_df.select_dtypes(include=np.number)

    if not np.isfinite(numeric_values.to_numpy()).all():
        errors.append("Dữ liệu chứa giá trị không hợp lệ.")

    # Kiem tra pham vi tung dac trung
    for feature in features:

        value = input_df[feature].iloc[0]
        min_value = training_df[feature].min()
        max_value = training_df[feature].max()

        if value < min_value or value > max_value:
            errors.append(
                f"{feature}: giá trị phải nằm trong "
                f"khoảng {min_value:.4f} đến {max_value:.4f}."
            )

    return errors


# =====================================
# 5. NHAP THONG SO HOA HOC
# =====================================

st.divider()
st.subheader("1. Nhập thông số hóa học")

st.info(
    "Điều chỉnh các thông số bên dưới để thực hiện dự đoán."
)

input_values = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    with (col1 if i % 2 == 0 else col2):

        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        median_value = float(df[feature].median())

        input_values[feature] = st.slider(
            feature,
            min_value=min_value,
            max_value=max_value,
            value=median_value
        )

# Tao DataFrame dau vao
input_df = pd.DataFrame([input_values])

# =====================================
# 6. HIEN THI DU LIEU DAU VAO
# =====================================

st.divider()
st.subheader("Kiểm tra thông số đầu vào")

with st.expander("Xem chi tiết 11 thông số đã nhập"):

    st.dataframe(
        input_df,
        use_container_width=True
    )

# =====================================
# 7. DU DOAN CHAT LUONG
# =====================================

st.divider()
st.subheader("2. Dự đoán chất lượng")

if st.button(
    "🍷 DỰ ĐOÁN NGAY",
    type="primary",
    use_container_width=True
):

    errors = validate_input(input_df, df)

    if errors:

        st.error("Dữ liệu đầu vào không hợp lệ!")

        for error in errors:
            st.warning(error)

    else:

        st.success("Dữ liệu đầu vào hợp lệ!")

        try:

            prediction = float(model.predict(input_df)[0])

            st.success("Dự đoán thành công!")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Điểm chất lượng dự đoán",
                    f"{prediction:.2f}/10"
                )

            with c2:
                st.metric(
                    "Điểm chất lượng gần nhất",
                    f"{round(prediction)}"
                )

            st.write("### Đánh giá tham khảo")

            if prediction < 5:
                st.warning("Chất lượng dự đoán: Thấp")

            elif prediction < 7:
                st.info("Chất lượng dự đoán: Trung bình")

            else:
                st.success("Chất lượng dự đoán: Tốt")

            st.caption(
                "Kết quả là dự đoán của mô hình học máy, "
                "không thay thế đánh giá chất lượng rượu vang chuyên môn."
            )

        except Exception as e:

            st.error(f"Lỗi trong quá trình dự đoán: {e}")


# =====================================
# 8. THONG TIN MO HINH
# =====================================

st.divider()
st.subheader("3. Thông tin mô hình")

st.write("""
- Thuật toán: Random Forest Regression
- Dữ liệu: Wine Quality Red
- Số đặc trưng: 11
- MAE: 0.4682
- RMSE: 0.6145
- R²: 0.4670
""")

st.caption("PROJECT MACHINE LEARNING - GROUP 17")


# =====================================
# 9. FEATURE IMPORTANCE
# =====================================

st.divider()
st.subheader("4. Mức độ quan trọng của các đặc trưng")

st.write(
    "Biểu đồ thể hiện mức độ đóng góp tương đối "
    "của từng thông số hóa học trong mô hình Random Forest."
)

# Lay muc do quan trong
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

# Sap xep tu cao xuong thap
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

# Ve bieu do
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.set_xlabel("Importance")
ax.set_ylabel("Chemical Features")
ax.set_title("Random Forest Feature Importance")

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Hien thi bang chi tiet
st.write("### Bảng mức độ quan trọng")

importance_display = importance_df.sort_values(
    by="Importance",
    ascending=False
).copy()

importance_display["Importance"] = (
    importance_display["Importance"].round(4)
)

st.dataframe(
    importance_display,
    use_container_width=True,
    hide_index=True
)


# =====================================
# 10. SO SANH CAC MO HINH
# =====================================

st.divider()
st.subheader("5. So sánh các mô hình Machine Learning")

COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
TUNED_PATH = ROOT / "reports" / "tuned_model_result.csv"

try:
    comparison_df = pd.read_csv(COMPARISON_PATH)

    # Them Random Forest sau toi uu neu co ket qua
    if TUNED_PATH.exists():
        tuned_df = pd.read_csv(TUNED_PATH)

        comparison_df = pd.concat(
            [comparison_df, tuned_df],
            ignore_index=True
        )

    st.write("### Bảng so sánh hiệu suất")

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # Bieu do MAE va RMSE
    st.write("### So sánh MAE và RMSE")

    error_df = comparison_df.set_index("Model")[
        ["MAE", "RMSE"]
    ]

    st.bar_chart(error_df)

    # Bieu do R2
    st.write("### So sánh R²")

    r2_df = comparison_df.set_index("Model")[["R2"]]

    st.bar_chart(r2_df)

    # Tim mo hinh tot nhat theo RMSE
    best_index = comparison_df["RMSE"].idxmin()
    best_model_name = comparison_df.loc[
        best_index, "Model"
    ]

    best_rmse = comparison_df.loc[
        best_index, "RMSE"
    ]

    best_r2 = comparison_df.loc[
        best_index, "R2"
    ]

    st.success(
        f"Mô hình có RMSE thấp nhất: {best_model_name}"
    )

    st.metric("RMSE tốt nhất", f"{best_rmse:.4f}")
    st.metric("R² của mô hình tốt nhất", f"{best_r2:.4f}")

except FileNotFoundError:
    st.warning(
        "Không tìm thấy model_comparison.csv. "
        "Hãy kiểm tra thư mục reports."
    )

except Exception as e:
    st.error(f"Lỗi khi so sánh mô hình: {e}")
