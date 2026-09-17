from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_selection import chi2


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
BASELINE_FILE = ROOT / "CSV" / "baseline_features.csv"
OUTPUT_FILE = ROOT / "CSV" / "chi_square_scores.csv"


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("========================================")
print("CHI-SQUARE FEATURE SELECTION")
print("========================================")

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Training dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Shape:", df.shape)


# ============================================================
# 2. LOAD BASELINE FEATURES
# ============================================================

print("\n========================================")
print("BASELINE FEATURE INFORMATION")
print("========================================")

baseline = pd.read_csv(BASELINE_FILE)

features = baseline["Feature"].tolist()

print("Baseline features:", len(features))
print("Target variable:", "Label")
print("Target classes:", df["Label"].nunique())


# ============================================================
# 3. PREPARE DATA
# ============================================================

print("\n========================================")
print("DATA PREPARATION")
print("========================================")

X = df[features].copy()
y = df["Label"].copy()

print("Missing values:", X.isna().sum().sum())
print("Infinite values:", np.isinf(X).sum().sum())

if X.isna().sum().sum() > 0 or np.isinf(X).sum().sum() > 0:
    print("ERROR: Missing or infinite values detected.")
    exit()

print("No missing or infinite values detected.")


# ============================================================
# 4. HANDLE NEGATIVE VALUES
# ============================================================

print("\nChecking feature values...")

negative_features = []

for column in X.columns:
    if (X[column] < 0).any():
        negative_features.append(column)

print("Features containing negative values:", len(negative_features))

if len(negative_features) > 0:
    print("\nNegative-value features:")
    for feature in negative_features:
        print("-", feature)


# ============================================================
# 5. SHIFT FEATURES TO NON-NEGATIVE RANGE
# ============================================================

print("\nPreparing features for Chi-Square analysis...")

X_chi = X.copy()

for column in X_chi.columns:
    minimum = X_chi[column].min()

    if minimum < 0:
        X_chi[column] = X_chi[column] - minimum

print("All feature values are now non-negative.")


# ============================================================
# 6. ENCODE TARGET
# ============================================================

print("\nEncoding target classes...")

target_mapping = {
    label: index
    for index, label in enumerate(y.unique())
}

y_encoded = y.map(target_mapping)

print("Target encoding completed.")
print("Classes encoded:", y_encoded.nunique())


# ============================================================
# 7. CALCULATE CHI-SQUARE SCORES
# ============================================================

print("\n========================================")
print("CALCULATING CHI-SQUARE SCORES")
print("========================================")

print("This may take some time because the dataset")
print("contains over 2 million training samples.")

chi_scores, p_values = chi2(X_chi, y_encoded)

print("Chi-Square calculation completed.")


# ============================================================
# 8. CREATE FEATURE RANKING
# ============================================================

chi_results = pd.DataFrame({
    "Feature": features,
    "Chi_Square_Score": chi_scores,
    "P_Value": p_values
})

chi_results = chi_results.sort_values(
    by="Chi_Square_Score",
    ascending=False
).reset_index(drop=True)

chi_results.insert(
    0,
    "Rank",
    range(1, len(chi_results) + 1)
)


# ============================================================
# 9. DISPLAY RANKING
# ============================================================

print("\n========================================")
print("CHI-SQUARE FEATURE RANKING")
print("========================================")

print(chi_results.to_string(index=False))


# ============================================================
# 10. SAVE RESULTS
# ============================================================

chi_results.to_csv(OUTPUT_FILE, index=False)

print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Features ranked:", len(chi_results))


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n========================================")
print("CHI-SQUARE FEATURE SELECTION COMPLETE")
print("========================================")