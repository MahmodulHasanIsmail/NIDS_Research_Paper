from pathlib import Path
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
BASELINE_FILE = ROOT / "CSV" / "baseline_features.csv"
OUTPUT_FILE = ROOT / "CSV" / "l1_ranking.csv"

N_FEATURES_TO_SELECT = 35

RANDOM_STATE = 42
C_VALUE = 1.0
MAX_ITER = 1000

# CPU control
# Use -1 for maximum CPU usage.
# Use 4, 6, etc. if you want to multitask.
N_JOBS = -1


# ============================================================
# HEADER
# ============================================================

print("========================================")
print("L1 FEATURE SELECTION")
print("========================================")


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

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

baseline_df = pd.read_csv(
    BASELINE_FILE
)

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
    raise ValueError(
        "Dataset contains missing values."
    )

if infinite_values > 0:
    raise ValueError(
        "Dataset contains infinite values."
    )

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
# 6. FEATURE STANDARDIZATION
# ============================================================

print("\n========================================")
print("FEATURE STANDARDIZATION")
print("========================================")

print("Standardizing features...")
print("This may take some time because the dataset")
print("contains over 2 million samples.")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature standardization completed.")


# ============================================================
# 7. L1 MODEL CONFIGURATION
# ============================================================

print("\n========================================")
print("L1 MODEL CONFIGURATION")
print("========================================")

print("Estimator: Logistic Regression")
print("Regularization: L1")
print("Multiclass strategy: One-vs-Rest")
print("Solver: saga")
print("C:", C_VALUE)
print("Maximum iterations:", MAX_ITER)
print("Parallel jobs:", N_JOBS)
print("Features before selection:", X.shape[1])
print("Target features:", N_FEATURES_TO_SELECT)


# ============================================================
# 8. CREATE BASE L1 MODEL
# ============================================================

print("\nCreating L1 Logistic Regression estimator...")

base_model = LogisticRegression(
    solver="saga",
    penalty="l1",
    C=C_VALUE,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE
)


# ============================================================
# 9. ONE-VS-REST CLASSIFIER
# ============================================================

print("Creating One-vs-Rest classifier...")

model = OneVsRestClassifier(
    estimator=base_model,
    n_jobs=N_JOBS
)

print("One-vs-Rest classifier created successfully.")


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n========================================")
print("TRAINING L1 LOGISTIC REGRESSION")
print("========================================")

print("Training on the complete training dataset.")
print("Samples:", X_scaled.shape[0])
print("Features:", X_scaled.shape[1])
print("Classes:", len(class_names))

print("\n15 binary L1 classifiers will be trained")
print("using the One-vs-Rest strategy.")

print("\nThis may take considerable time.")
print("CPU usage may be high.")
print("Memory usage may also be high.")

print("\nPlease do not interrupt the process.")


model.fit(
    X_scaled,
    y_encoded
)

print("\nL1 Logistic Regression training completed.")


# ============================================================
# 11. EXTRACT COEFFICIENTS
# ============================================================

print("\n========================================")
print("EXTRACTING MODEL COEFFICIENTS")
print("========================================")

coefficients = np.array([
    estimator.coef_[0]
    for estimator in model.estimators_
])

print("Coefficient matrix shape:", coefficients.shape)

print(
    "Expected shape:",
    (len(class_names), len(baseline_features))
)


# ============================================================
# 12. CALCULATE FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("CALCULATING FEATURE IMPORTANCE")
print("========================================")

absolute_coefficients = np.abs(
    coefficients
)

# Maximum absolute coefficient across
# all 15 One-vs-Rest classifiers.

max_importance = np.max(
    absolute_coefficients,
    axis=0
)

# Mean absolute coefficient across
# all 15 classifiers.

mean_importance = np.mean(
    absolute_coefficients,
    axis=0
)

# Number of classifiers in which the feature
# received a non-zero coefficient.

nonzero_counts = np.sum(
    absolute_coefficients > 1e-10,
    axis=0
)

print("Feature importance calculated.")


# ============================================================
# 13. CREATE FEATURE RANKING
# ============================================================

print("\n========================================")
print("CREATING L1 FEATURE RANKING")
print("========================================")

l1_ranking = pd.DataFrame({
    "Feature": baseline_features,
    "L1_Max_Importance": max_importance,
    "L1_Mean_Importance": mean_importance,
    "Nonzero_Class_Count": nonzero_counts
})


# Rank primarily by maximum absolute coefficient.
#
# Mean importance and non-zero class count are retained
# as additional information for research analysis.

l1_ranking = l1_ranking.sort_values(
    by=[
        "L1_Max_Importance",
        "L1_Mean_Importance",
        "Nonzero_Class_Count",
        "Feature"
    ],
    ascending=[
        False,
        False,
        False,
        True
    ]
).reset_index(drop=True)


# ============================================================
# 14. ADD RANK
# ============================================================

l1_ranking.insert(
    0,
    "Rank",
    range(
        1,
        len(l1_ranking) + 1
    )
)


# ============================================================
# 15. SELECT TOP 35 FEATURES
# ============================================================

l1_ranking["Selected"] = False

l1_ranking.loc[
    :N_FEATURES_TO_SELECT - 1,
    "Selected"
] = True


# ============================================================
# 16. DISPLAY RANKING
# ============================================================

print("\n========================================")
print("L1 FEATURE RANKING")
print("========================================")

print(
    l1_ranking.to_string(
        index=False
    )
)


# ============================================================
# 17. DISPLAY SELECTED FEATURES
# ============================================================

selected_features = l1_ranking[
    l1_ranking["Selected"]
]["Feature"].tolist()

print("\n========================================")
print("SELECTED FEATURES")
print("========================================")

print(
    "Selected features:",
    len(selected_features)
)

for i, feature in enumerate(
    selected_features,
    1
):
    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# 18. SAVE RESULTS
# ============================================================

l1_ranking.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print(
    "Output file:",
    OUTPUT_FILE
)

print(
    "Features ranked:",
    len(l1_ranking)
)

print(
    "Features selected:",
    len(selected_features)
)


# ============================================================
# 19. SUMMARY
# ============================================================

print("\n========================================")
print("L1 FEATURE SELECTION SUMMARY")
print("========================================")

print("Original features:", 78)
print("Constant features removed:", 8)
print(
    "Baseline features:",
    len(baseline_features)
)
print(
    "L1 selected features:",
    len(selected_features)
)
print(
    "Features removed by L1:",
    len(baseline_features)
    - len(selected_features)
)
print(
    "Target classes:",
    len(class_names)
)
print(
    "Multiclass strategy:",
    "One-vs-Rest"
)
print(
    "Regularization:",
    "L1"
)
print(
    "Solver:",
    "saga"
)
print(
    "C:",
    C_VALUE
)

print("\n========================================")
print("L1 FEATURE SELECTION COMPLETE")
print("========================================")