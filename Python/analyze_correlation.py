
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
OUTPUT_FILE = ROOT / "CSV" / "feature_correlation.csv"


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("=" * 40)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 40)

print("\nLoading training dataset...")

df = pd.read_csv(TRAIN_FILE, low_memory=False)

print("Training dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Shape: {df.shape}")


# Separate features and target
X = df.drop(columns=["Label"])


print("\n" + "=" * 40)
print("FEATURE / TARGET INFORMATION")
print("=" * 40)

print(f"Input features: {X.shape[1]}")
print("Target variable: Label")


# Calculate correlation matrix
print("\nCalculating Pearson correlation matrix...")

correlation_matrix = X.corr(method="pearson")

print("Correlation matrix calculated successfully.")


# Find highly correlated feature pairs
print("\n" + "=" * 40)
print("HIGH CORRELATION ANALYSIS")
print("=" * 40)

threshold = 0.90

correlation_pairs = []

features = correlation_matrix.columns

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= threshold:
            correlation_pairs.append({
                "Feature_1": features[i],
                "Feature_2": features[j],
                "Correlation": correlation,
                "Absolute_Correlation": abs(correlation)
            })


high_corr_df = pd.DataFrame(correlation_pairs)


if len(high_corr_df) > 0:
    high_corr_df = high_corr_df.sort_values(
        by="Absolute_Correlation",
        ascending=False
    )

    print(f"Correlation threshold: |r| >= {threshold}")
    print(f"Highly correlated feature pairs: {len(high_corr_df)}")
    print("\nHighly correlated feature pairs:")
    print(high_corr_df.to_string(index=False))

else:
    print(f"Correlation threshold: |r| >= {threshold}")
    print("No highly correlated feature pairs found.")


# Save high-correlation results
high_corr_df.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 40)
print("CORRELATION ANALYSIS SUMMARY")
print("=" * 40)

print(f"Total input features analyzed: {X.shape[1]}")
print(f"Correlation threshold: |r| >= {threshold}")
print(f"Highly correlated pairs: {len(high_corr_df)}")


print("\n" + "=" * 40)
print("OUTPUT SAVED")
print("=" * 40)

print(f"Output file: {OUTPUT_FILE}")


print("\n" + "=" * 40)
print("FEATURE CORRELATION ANALYSIS COMPLETE")
print("=" * 40)