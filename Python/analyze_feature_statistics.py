from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
OUTPUT_FILE = ROOT / "CSV" / "feature_statistics.csv"


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("========================================")
print("FEATURE STATISTICAL ANALYSIS")
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

print("\n========================================")
print("FEATURE INFORMATION")
print("========================================")

print("Input features:", X.shape[1])
print("Target variable:", "Label")


# ============================================================
# 3. CALCULATE FEATURE STATISTICS
# ============================================================

print("\n========================================")
print("CALCULATING FEATURE STATISTICS")
print("========================================")

statistics = pd.DataFrame({
    "Feature": X.columns,
    "Mean": X.mean(),
    "Std": X.std(),
    "Min": X.min(),
    "Median": X.median(),
    "Max": X.max(),
    "Skewness": X.skew(),
    "Unique_Values": X.nunique()
})

statistics = statistics.reset_index(drop=True)

print("Feature statistics calculated successfully.")


# ============================================================
# 4. HIGHLY SKEWED FEATURES
# ============================================================

print("\n========================================")
print("SKEWNESS ANALYSIS")
print("========================================")

skew_threshold = 10

highly_skewed = statistics[
    statistics["Skewness"].abs() >= skew_threshold
].sort_values(
    by="Skewness",
    key=lambda x: x.abs(),
    ascending=False
)

print("Skewness threshold: |skewness| >= 10")
print("Highly skewed features:", len(highly_skewed))

if len(highly_skewed) > 0:
    print("\nHighly skewed features:")
    print(
        highly_skewed[
            ["Feature", "Skewness", "Min", "Median", "Max"]
        ].to_string(index=False)
    )


# ============================================================
# 5. FEATURE RANGE ANALYSIS
# ============================================================

print("\n========================================")
print("FEATURE RANGE ANALYSIS")
print("========================================")

statistics["Range"] = statistics["Max"] - statistics["Min"]

largest_ranges = statistics.sort_values(
    by="Range",
    ascending=False
).head(10)

print("Top 10 features by numerical range:")

print(
    largest_ranges[
        ["Feature", "Min", "Max", "Range"]
    ].to_string(index=False)
)


# ============================================================
# 6. ZERO-VARIANCE CONFIRMATION
# ============================================================

print("\n========================================")
print("ZERO-VARIANCE CONFIRMATION")
print("========================================")

zero_variance = statistics[
    statistics["Std"] == 0
]

print("Zero-variance features:", len(zero_variance))

if len(zero_variance) > 0:
    print("\nZero-variance features:")
    print(zero_variance["Feature"].to_string(index=False))


# ============================================================
# 7. FINAL STATISTICAL SUMMARY
# ============================================================

print("\n========================================")
print("STATISTICAL ANALYSIS SUMMARY")
print("========================================")

print("Total input features:", len(X.columns))
print("Highly skewed features:", len(highly_skewed))
print("Zero-variance features:", len(zero_variance))

print(
    "Features with negative values:",
    (X.min() < 0).sum()
)

print(
    "Features with maximum value > 1,000,000:",
    (X.max() > 1_000_000).sum()
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

statistics.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Features recorded:", len(statistics))


# ============================================================
# 9. COMPLETION
# ============================================================

print("\n========================================")
print("FEATURE STATISTICAL ANALYSIS COMPLETE")
print("========================================")