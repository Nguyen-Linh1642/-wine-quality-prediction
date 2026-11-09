
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# NHOM 17 - WINE QUALITY PREDICTION
# DATA PREPROCESSING
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "winequality-red.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# 1. Doc du lieu
df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 55)
print("NHOM 17 - TIEN XU LY DU LIEU")
print("=" * 55)

print("\n1. Kich thuoc ban dau:", df.shape)

# 2. Kiem tra du lieu thieu
print("\n2. So du lieu thieu:")
print(df.isnull().sum().sum())

# 3. Kiem tra du lieu trung lap
duplicate_count = df.duplicated().sum()

print("\n3. So dong trung lap:", duplicate_count)

# 4. Loai bo dong trung lap hoan toan
df = df.drop_duplicates().copy()

print("Kich thuoc sau khi xu ly:", df.shape)

# 5. Tach bien dau vao va bien muc tieu
X = df.drop(columns=["quality"])
y = df["quality"]

print("\n4. SO LUONG DAC TRUNG:", X.shape[1])
print("TEN DAC TRUNG:", X.columns.tolist())

# 6. Chia du lieu 80% train - 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. In ket qua
print("\n5. KET QUA CHIA DU LIEU")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

print("\n6. PHAN BO QUALITY")

print("Train:")
print(y_train.value_counts().sort_index())

print("\nTest:")
print(y_test.value_counts().sort_index())

# 8. Luu du lieu
X_train.to_csv(REPORT_DIR / "X_train.csv", index=False)
X_test.to_csv(REPORT_DIR / "X_test.csv", index=False)

y_train.to_csv(REPORT_DIR / "y_train.csv", index=False)
y_test.to_csv(REPORT_DIR / "y_test.csv", index=False)

# 9. Luu thong tin tien xu ly
with open(
    REPORT_DIR / "preprocessing_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("NHOM 17 - PREPROCESSING SUMMARY\n")
    f.write(f"So dong ban dau: {len(df) + duplicate_count}\n")
    f.write(f"So dong trung lap da loai: {duplicate_count}\n")
    f.write(f"So dong sau xu ly: {len(df)}\n")
    f.write(f"So dac trung: {X.shape[1]}\n")
    f.write(f"So mau train: {len(X_train)}\n")
    f.write(f"So mau test: {len(X_test)}\n")

print("\n" + "=" * 55)
print("TIEN XU LY DU LIEU THANH CONG!")
print("Du lieu da duoc luu trong thu muc reports.")
print("=" * 55)
