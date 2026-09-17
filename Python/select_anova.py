from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "Processed_Data" / "train_CICIDS2017.csv"
BASELINE_FILE = Path(__file__).resolve().parent.parent / "CSV" / "baseline_features.csv"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "CSV" / "anova_scores.csv"


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("========================================")
print("ANOVA F-SCORE FEATURE SELECTION")
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

baseline_features = baseline["Feature"].tolist()

print("Baseline features:", len(baseline_features))

X = df[baseline_features]
y = df["Label"]

print("Target variable:", "Label")
print("Target classes:", y.nunique())


# ============================================================
# 3. DATA PREPARATION
# ============================================================

print("\n========================================")
print("DATA PREPARATION")
print("========================================")

missing_values = X.isna().sum().sum()
infinite_values = np.isinf(X).sum().sum()

print("Missing values:", missing_values)
print("Infinite values:", infinite_values)

if missing_values > 0 or infinite_values > 0:
    print("\nERROR: Missing or infinite values detected.")
    print("ANOVA analysis cannot continue.")
    exit()

print("No missing or infinite values detected.")

print("\nEncoding target classes...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Target encoding completed.")
print("Classes encoded:", len(label_encoder.classes_))


# ============================================================
# 4. CALCULATE ANOVA F-SCORES
# ============================================================

print("\n========================================")
print("CALCULATING ANOVA F-SCORES")
print("========================================")

print("This may take some time because the dataset")
print("contains over 2 million training samples.")

f_scores, p_values = f_classif(X, y_encoded)

print("ANOVA F-score calculation completed.")


# ============================================================
# 5. CREATE RANKING
# ============================================================

anova_results = pd.DataFrame({
    "Feature": baseline_features,
    "F_Score": f_scores,
    "P_Value": p_values
})

anova_results = anova_results.sort_values(
    by="F_Score",
    ascending=False
).reset_index(drop=True)

anova_results.insert(
    0,
    "Rank",
    range(1, len(anova_results) + 1)
)


# ============================================================
# 6. DISPLAY FEATURE RANKING
# ============================================================

print("\n========================================")
print("ANOVA F-SCORE FEATURE RANKING")
print("========================================")

print(
    anova_results.to_string(index=False)
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

anova_results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Features ranked:", len(anova_results))


# ============================================================
# 8. COMPLETION
# ============================================================

print("\n========================================")
print("ANOVA F-SCORE ANALYSIS COMPLETE")
print("========================================")