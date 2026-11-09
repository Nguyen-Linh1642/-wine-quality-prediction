
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# NHOM 17 - WINE QUALITY PREDICTION
# EXPLORATORY DATA ANALYSIS
# ==========================================

# Đường dẫn dự án
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Đọc dữ liệu
red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")

# Cấu hình biểu đồ
sns.set_theme(style="whitegrid")

print("=" * 60)
print("NHOM 17 - PHAN TICH DU LIEU RUOU VANG")
print("=" * 60)

# 1. Kích thước dữ liệu
print("\n1. KICH THUOC DU LIEU")
print("Ruou vang do:", red.shape)
print("Ruou vang trang:", white.shape)

# 2. Kiểu dữ liệu
print("\n2. KIEU DU LIEU")
print(red.dtypes)

# 3. Kiểm tra dữ liệu thiếu
print("\n3. DU LIEU THIEU")
print("Red:", red.isnull().sum().sum())
print("White:", white.isnull().sum().sum())

# 4. Kiểm tra dữ liệu trùng
print("\n4. DU LIEU TRUNG LAP")
print("Red:", red.duplicated().sum())
print("White:", white.duplicated().sum())

# 5. Thống kê mô tả
print("\n5. THONG KE MO TA RUOU VANG DO")
print(red.describe().round(3))

print("\n6. THONG KE MO TA RUOU VANG TRANG")
print(white.describe().round(3))

# 7. Phân bố điểm chất lượng
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=red, x="quality", ax=axes[0])
axes[0].set_title("Red Wine Quality Distribution")
axes[0].set_xlabel("Quality")
axes[0].set_ylabel("Number of samples")

sns.countplot(data=white, x="quality", ax=axes[1])
axes[1].set_title("White Wine Quality Distribution")
axes[1].set_xlabel("Quality")
axes[1].set_ylabel("Number of samples")

plt.tight_layout()
plt.savefig(REPORT_DIR / "01_quality_distribution.png", dpi=300)
plt.show()

# 8. Ma trận tương quan
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(
    red.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=axes[0]
)
axes[0].set_title("Red Wine Correlation")

sns.heatmap(
    white.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=axes[1]
)
axes[1].set_title("White Wine Correlation")

plt.tight_layout()
plt.savefig(REPORT_DIR / "02_correlation.png", dpi=300)
plt.show()

# 9. So sánh nồng độ cồn theo điểm chất lượng
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=red, x="quality", y="alcohol", ax=axes[0])
axes[0].set_title("Red Wine: Alcohol vs Quality")

sns.boxplot(data=white, x="quality", y="alcohol", ax=axes[1])
axes[1].set_title("White Wine: Alcohol vs Quality")

plt.tight_layout()
plt.savefig(REPORT_DIR / "03_alcohol_quality.png", dpi=300)
plt.show()

# 10. Lưu báo cáo thống kê
with open(REPORT_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("NHOM 17 - WINE QUALITY EDA\n\n")
    f.write("RED WINE\n")
    f.write(str(red.describe()))
    f.write("\n\nWHITE WINE\n")
    f.write(str(white.describe()))

print("\nHOAN THANH PHAN TICH DU LIEU!")
print("Cac bieu do da duoc luu trong thu muc reports.")
