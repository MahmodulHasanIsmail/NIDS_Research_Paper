import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "train_CICIDS2017.csv"
BASELINE_FILE = "baseline_features.csv"
OUTPUT_FILE = "rf_ranking.csv"

N_FEATURES_TO_SELECT = 35


# ============================================================
# HEADER
# ============================================================

print("========================================")
print("RANDOM FOREST FEATURE SELECTION")
print("========================================")


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

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

baseline_df = pd.read_csv(BASELINE_FILE)

baseline_features = baseline_df["Feature"].tolist()

print("Baseline features:", len(baseline_features))
print("Target variable:", "Label")


# ============================================================
# 3. PREPARE DATA
# ============================================================

print("\n========================================")
print("DATA PREPARATION")
print("========================================")

X = df[baseline_features].copy()
y = df["Label"].copy()

print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Target classes:", y.nunique())


# ============================================================
# 4. DATA QUALITY CHECK
# ============================================================

print("\n========================================")
print("DATA QUALITY CHECK")
print("========================================")

missing_values = X.isna().sum().sum()
infinite_values = np.isinf(X).sum().sum()

print("Missing values:", missing_values)
print("Infinite values:", infinite_values)

if missing_values > 0:
    raise ValueError("Dataset contains missing values.")

if infinite_values > 0:
    raise ValueError("Dataset contains infinite values.")

print("Data quality check passed.")


# ============================================================
# 5. TARGET ENCODING
# ============================================================

print("\n========================================")
print("TARGET ENCODING")
print("========================================")

y_encoded, class_names = pd.factorize(y)

print("Classes encoded:", len(class_names))

print("\nTarget classes:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# 6. RANDOM FOREST CONFIGURATION
# ============================================================

print("\n========================================")
print("RANDOM FOREST CONFIGURATION")
print("========================================")

print("Estimator: Random Forest Classifier")
print("Number of trees: 100")
print("Criterion: Gini")
print("Features before selection:", X.shape[1])
print("Target features:", N_FEATURES_TO_SELECT)
print("Parallel jobs: -1")
print("Random state: 42")

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

print("Random Forest model created successfully.")


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

print("\n========================================")
print("TRAINING RANDOM FOREST")
print("========================================")

print("Training on the complete training dataset.")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", y.nunique())

print("\nThis may take considerable time.")
print("CPU usage may be high.")
print("Memory usage may also be high.")
print("Please do not interrupt the process.")

model.fit(X, y_encoded)

print("\nRandom Forest training completed.")


# ============================================================
# 8. CALCULATE FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("CALCULATING FEATURE IMPORTANCE")
print("========================================")

feature_importance = model.feature_importances_

print("Feature importance calculated.")
print("Importance values:", len(feature_importance))


# ============================================================
# 9. CREATE FEATURE RANKING
# ============================================================

print("\n========================================")
print("CREATING RANDOM FOREST FEATURE RANKING")
print("========================================")

rf_ranking = pd.DataFrame({
    "Feature": baseline_features,
    "RF_Importance": feature_importance
})

rf_ranking = rf_ranking.sort_values(
    by=["RF_Importance", "Feature"],
    ascending=[False, True]
).reset_index(drop=True)

rf_ranking.insert(
    0,
    "Rank",
    range(1, len(rf_ranking) + 1)
)


# ============================================================
# 10. SELECT TOP FEATURES
# ============================================================

rf_ranking["Selected"] = False

rf_ranking.loc[
    rf_ranking.index < N_FEATURES_TO_SELECT,
    "Selected"
] = True


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("RANDOM FOREST FEATURE RANKING")
print("========================================")

print(
    rf_ranking.to_string(index=False)
)


# ============================================================
# 12. DISPLAY SELECTED FEATURES
# ============================================================

selected_features = rf_ranking[
    rf_ranking["Selected"] == True
]["Feature"].tolist()

print("\n========================================")
print("SELECTED FEATURES")
print("========================================")

print("Selected features:", len(selected_features))

for i, feature in enumerate(selected_features, 1):
    print(f"{i:2d}. {feature}")


# ============================================================
# 13. SAVE RESULTS
# ============================================================

rf_ranking.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Features ranked:", len(rf_ranking))
print("Features selected:", len(selected_features))


# ============================================================
# 14. SUMMARY
# ============================================================

print("\n========================================")
print("RANDOM FOREST FEATURE SELECTION SUMMARY")
print("========================================")

print("Original features:", 78)
print("Constant features removed:", 8)
print("Baseline features:", len(baseline_features))
print("RF selected features:", len(selected_features))

print(
    "Features removed by RF:",
    len(baseline_features) - len(selected_features)
)

print("Target classes:", y.nunique())
print("Number of trees:", 100)
print("Criterion:", "Gini")


print("\n========================================")
print("RANDOM FOREST FEATURE SELECTION COMPLETE")
print("========================================")