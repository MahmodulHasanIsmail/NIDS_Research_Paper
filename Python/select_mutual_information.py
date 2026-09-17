from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
BASELINE_FILE = ROOT / "CSV" / "baseline_features.csv"
OUTPUT_FILE = ROOT / "CSV" / "mutual_information_scores.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("========================================")
print("MUTUAL INFORMATION FEATURE SELECTION")
print("========================================")

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Training dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. LOAD BASELINE FEATURES
# ============================================================

baseline = pd.read_csv(BASELINE_FILE)

baseline_features = baseline["Feature"].tolist()

print("\n========================================")
print("BASELINE FEATURE INFORMATION")
print("========================================")

print("Baseline features:", len(baseline_features))


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df[baseline_features]
y = df["Label"]

print("\nTarget variable:", "Label")
print("Target classes:", y.nunique())


# ============================================================
# 4. HANDLE NUMERICAL DATA
# ============================================================

print("\n========================================")
print("DATA PREPARATION")
print("========================================")

X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().sum().sum() > 0:
    print("Missing/infinite values detected.")
    print("Replacing with median values...")

    X = X.fillna(X.median())

else:
    print("No missing or infinite values detected.")


# ============================================================
# 5. ENCODE TARGET
# ============================================================

print("\nEncoding target classes...")

target_mapping = {
    label: index
    for index, label in enumerate(sorted(y.unique()))
}

y_encoded = y.map(target_mapping)

print("Target encoding completed.")
print("Classes encoded:", len(target_mapping))


# ============================================================
# 6. CALCULATE MUTUAL INFORMATION
# ============================================================

print("\n========================================")
print("CALCULATING MUTUAL INFORMATION")
print("========================================")

print("This may take some time because the dataset")
print("contains over 2 million training samples.")

mi_scores = mutual_info_classif(
    X,
    y_encoded,
    random_state=42
)

print("Mutual Information calculation completed.")


# ============================================================
# 7. CREATE FEATURE RANKING
# ============================================================

mi_results = pd.DataFrame({
    "Feature": baseline_features,
    "Mutual_Information": mi_scores
})

mi_results = mi_results.sort_values(
    by="Mutual_Information",
    ascending=False
).reset_index(drop=True)

mi_results["Rank"] = (
    mi_results.index + 1
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MUTUAL INFORMATION FEATURE RANKING")
print("========================================")

print(
    mi_results[
        [
            "Rank",
            "Feature",
            "Mutual_Information"
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

mi_results[
    [
        "Rank",
        "Feature",
        "Mutual_Information"
    ]
].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Features ranked:", len(mi_results))

print("\n========================================")
print("MUTUAL INFORMATION ANALYSIS COMPLETE")
print("========================================")