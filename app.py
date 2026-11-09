
import streamlit as st # type: ignore
import pandas as pd
import joblib
from pathlib import Path

# Duong dan
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_PATH = ROOT / "models" / "random_forest.pkl"

# Cau hinh giao dien
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

# Tai du lieu va mo hinh
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

st.divider()
st.subheader("1. Nhập thông số hóa học")

st.info("Điều chỉnh các thông số bên dưới để thực hiện dự đoán.")

# Tao giao dien nhap lieu
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

# Du lieu dau vao
input_df = pd.DataFrame([input_values])

st.divider()

st.subheader("2. Dự đoán chất lượng")

if st.button("🍷 DỰ ĐOÁN NGAY", type="primary",
             use_container_width=True):

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
