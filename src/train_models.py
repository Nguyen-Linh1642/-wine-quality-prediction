
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# 1. Đọc và xử lý dữ liệu
df = pd.read_csv(DATA_PATH, sep=";")
df = df.drop_duplicates().copy()

X = df.drop(columns=["quality"])
y = df["quality"]

# 2. Chia dữ liệu
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Khởi tạo các mô hình
models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8, random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

# 4. Huấn luyện và đánh giá
results = []

print("=" * 65)
print("NHOM 17 - WINE QUALITY PREDICTION")
print("=" * 65)

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\nMo hinh: {name}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # Lưu mô hình
    import joblib
    joblib.dump(model, MODEL_DIR / (
        name.lower().replace(" ", "_") + ".pkl"
    ))

# 5. Lưu bảng kết quả
results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("KET QUA SO SANH CAC MO HINH")
print("=" * 65)
print(results_df.to_string(index=False))
print("\nDa luu ket qua tai reports/model_comparison.csv")
print("Da luu cac mo hinh tai thu muc models/")
