
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Khai bao duong dan
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_PATH = ROOT / "models" / "random_forest.pkl"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# 2. Doc va xu ly du lieu giong luc huan luyen
df = pd.read_csv(DATA_PATH, sep=";")
df = df.drop_duplicates().copy()

X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Tai mo hinh Random Forest da huan luyen
model = joblib.load(MODEL_PATH)

# 4. Du doan tren tap test
y_pred = model.predict(X_test)
residuals = y_test.to_numpy() - y_pred

# 5. Tinh lai cac chi so danh gia
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("NHOM 17 - PHAN TICH MO HINH RANDOM FOREST")
print("=" * 60)
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# 6. Luu du lieu du doan va sai so
prediction_df = pd.DataFrame({
    "Actual Quality": y_test.to_numpy(),
    "Predicted Quality": y_pred,
    "Residual": residuals,
    "Absolute Error": np.abs(residuals)
})

prediction_df.to_csv(
    REPORT_DIR / "prediction_errors.csv",
    index=False
)

# 7. Tim 10 mau du doan sai nhieu nhat
worst_errors = prediction_df.sort_values(
    by="Absolute Error", ascending=False
).head(10)

worst_errors.to_csv(
    REPORT_DIR / "worst_predictions.csv",
    index=False
)

print("\n10 mau co sai so lon nhat:")
print(worst_errors.to_string(index=False))

# 8. Bieu do gia tri thuc te va du doan
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted Wine Quality")
plt.tight_layout()
plt.savefig(REPORT_DIR / "04_actual_vs_predicted.png", dpi=300)
plt.close()

# 9. Bieu do phan phoi sai so
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, edgecolor="black")
plt.axvline(0, linestyle="--")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(REPORT_DIR / "05_residual_distribution.png", dpi=300)
plt.close()

# 10. Bieu do sai so theo gia tri du doan
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Quality")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(REPORT_DIR / "06_residual_plot.png", dpi=300)
plt.close()

# 11. Muc do quan trong cua cac dac trung
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

plt.figure(figsize=(9, 6))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(REPORT_DIR / "07_feature_importance.png", dpi=300)
plt.close()

# 12. In ket qua
print("\nMUC DO QUAN TRONG CUA CAC DAC TRUNG:")
print(importance_df.to_string(index=False))

print("\n" + "=" * 60)
print("PHAN TICH MO HINH THANH CONG!")
print("Cac bieu do va bang ket qua da duoc luu trong reports/")
print("=" * 60)
