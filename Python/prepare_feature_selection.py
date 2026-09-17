from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"

INVENTORY_FILE = ROOT / "CSV" / "feature_inventory.csv"
CORRELATION_FILE = ROOT / "CSV" / "feature_correlation.csv"

OUTPUT_FILE = ROOT / "CSV" / "feature_selection_candidates.csv"


# ============================================================
# 1. LOAD FEATURE INFORMATION
# ============================================================

print("========================================")
print("FEATURE SELECTION PREPARATION")
print("========================================")

print("\nLoading feature inventory...")
inventory = pd.read_csv(INVENTORY_FILE)

print("Feature inventory loaded successfully.")
print("Features:", len(inventory))


print("\nLoading correlation analysis...")
correlation = pd.read_csv(CORRELATION_FILE)

print("Correlation analysis loaded successfully.")
print("Highly correlated pairs:", len(correlation))


# ============================================================
# 2. IDENTIFY CONSTANT FEATURES
# ============================================================

print("\n========================================")
print("CONSTANT FEATURE IDENTIFICATION")
print("========================================")

constant_features = inventory[
    inventory["Unique_Values"] <= 1
]["Feature"].tolist()

print("Constant features:", len(constant_features))

for feature in constant_features:
    print("-", feature)


# ============================================================
# 3. IDENTIFY HIGHLY CORRELATED FEATURES
# ============================================================

print("\n========================================")
print("HIGH CORRELATION IDENTIFICATION")
print("========================================")

correlated_features = set()

for _, row in correlation.iterrows():
    correlated_features.add(row["Feature_1"])
    correlated_features.add(row["Feature_2"])

correlated_features = sorted(correlated_features)

print("Features involved in high-correlation pairs:",
      len(correlated_features))


# ============================================================
# 4. CREATE FEATURE CANDIDATE REPORT
# ============================================================

print("\n========================================")
print("CREATING FEATURE CANDIDATE REPORT")
print("========================================")

candidate_report = inventory[
    inventory["Feature"].isin(
        set(constant_features) | set(correlated_features)
    )
].copy()


candidate_report["Constant_Feature"] = (
    candidate_report["Feature"].isin(constant_features)
)

candidate_report["Highly_Correlated"] = (
    candidate_report["Feature"].isin(correlated_features)
)


candidate_report["Selection_Reason"] = np.where(
    candidate_report["Constant_Feature"],
    "Zero variance / constant feature",
    "Part of highly correlated feature pair"
)


# ============================================================
# 5. DISPLAY SUMMARY
# ============================================================

print("\n========================================")
print("FEATURE SELECTION CANDIDATES")
print("========================================")

print(
    candidate_report[
        [
            "Feature",
            "Unique_Values",
            "Constant_Feature",
            "Highly_Correlated",
            "Selection_Reason"
        ]
    ].to_string(index=False)
)


# ============================================================
# 6. SAVE REPORT
# ============================================================

candidate_report.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Candidate features recorded:",
      len(candidate_report))


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("PREPARATION SUMMARY")
print("========================================")

print("Total input features:", len(inventory))
print("Constant features:", len(constant_features))
print("Features in high-correlation pairs:",
      len(correlated_features))
print("Total candidate features:",
      len(candidate_report))

print("\n========================================")
print("FEATURE SELECTION PREPARATION COMPLETE")
print("========================================")