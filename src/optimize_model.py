
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Duong dan
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# 2. Doc va xu ly du lieu
df = pd.read_csv(DATA_PATH, sep=";")
df = df.drop_duplicates().copy()

X = df.drop(columns=["quality"])
y = df["quality"]

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Khai bao khong gian tham so
param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 8, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 1.0]
}

# 5. Khoi tao Random Forest
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# 6. Tim tham so tot nhat bang Cross Validation
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("=" * 60)
print("BAT DAU TOI UU RANDOM FOREST")
print("=" * 60)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nTHAM SO TOT NHAT:")
print(search.best_params_)

print("\nRMSE TOT NHAT TRONG CROSS VALIDATION:")
print(-search.best_score_)

# 7. Danh gia tren tap test
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nKET QUA TREN TAP TEST:")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# 8. Luu mo hinh toi uu
joblib.dump(
    best_model,
    MODEL_DIR / "random_forest_tuned.pkl"
)

# 9. Luu ket qua
result = pd.DataFrame([{
    "Model": "Random Forest Tuned",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "CV_RMSE": -search.best_score_
}])

result.to_csv(
    REPORT_DIR / "tuned_model_result.csv",
    index=False
)

print("\nDA LUU MO HINH: models/random_forest_tuned.pkl")
print("DA LUU KET QUA: reports/tuned_model_result.csv")
print("\nTOI UU MO HINH HOAN TAT!")
