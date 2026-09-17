
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
OUTPUT_FILE = ROOT / "CSV" / "feature_inventory.csv"


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("========================================")
print("FEATURE INVENTORY ANALYSIS")
print("========================================")

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Training dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Label"])
y = df["Label"]

print("\n========================================")
print("FEATURE / TARGET INFORMATION")
print("========================================")

print("Input features:", X.shape[1])
print("Target variable:", "Label")
print("Target classes:", y.nunique())


# ============================================================
# 3. FEATURE INVENTORY
# ============================================================

feature_inventory = pd.DataFrame({
    "Feature_Number": range(1, len(X.columns) + 1),
    "Feature": X.columns,
    "Data_Type": X.dtypes.astype(str).values,
    "Unique_Values": [X[column].nunique() for column in X.columns],
    "Missing_Values": [X[column].isna().sum() for column in X.columns],
    "Infinite_Values": [
        np.isinf(X[column].values).sum()
        for column in X.columns
    ]
})


print("\n========================================")
print("FEATURE INVENTORY")
print("========================================")

print(feature_inventory.to_string(index=False))


# ============================================================
# 4. DATA TYPE ANALYSIS
# ============================================================

print("\n========================================")
print("DATA TYPE ANALYSIS")
print("========================================")

numeric_features = X.select_dtypes(include=np.number).columns
non_numeric_features = X.select_dtypes(exclude=np.number).columns

print("Numeric features:", len(numeric_features))
print("Non-numeric features:", len(non_numeric_features))

if len(non_numeric_features) > 0:
    print("\nNon-numeric features:")
    print(list(non_numeric_features))
else:
    print("All 78 input features are numeric.")


# ============================================================
# 5. CONSTANT FEATURE ANALYSIS
# ============================================================

print("\n========================================")
print("CONSTANT FEATURE ANALYSIS")
print("========================================")

unique_counts = X.nunique()

constant_features = unique_counts[unique_counts <= 1]

print("Constant features:", len(constant_features))

if len(constant_features) > 0:
    print("\nConstant features found:")
    print(constant_features)
else:
    print("No constant features found.")


# ============================================================
# 6. ZERO-VARIANCE CHECK
# ============================================================

print("\n========================================")
print("ZERO-VARIANCE CHECK")
print("========================================")

variances = X.var()

zero_variance_features = variances[variances == 0]

print("Zero-variance features:", len(zero_variance_features))

if len(zero_variance_features) > 0:
    print("\nZero-variance features:")
    print(zero_variance_features)
else:
    print("No zero-variance features found.")


# ============================================================
# 7. FINAL DATA QUALITY CHECK
# ============================================================

print("\n========================================")
print("FINAL DATA QUALITY CHECK")
print("========================================")

total_missing = X.isna().sum().sum()
total_infinite = np.isinf(X).sum().sum()

print("Missing values in input features:", total_missing)
print("Infinite values in input features:", total_infinite)


# ============================================================
# 8. SAVE FEATURE INVENTORY
# ============================================================

feature_inventory.to_csv(OUTPUT_FILE, index=False)

print("\n========================================")
print("FEATURE INVENTORY SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Number of features recorded:", len(feature_inventory))


# ============================================================
# 9. COMPLETION
# ============================================================

print("\n========================================")
print("FEATURE INVENTORY ANALYSIS COMPLETE")
print("========================================")