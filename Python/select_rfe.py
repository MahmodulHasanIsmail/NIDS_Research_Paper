import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "train_CICIDS2017.csv"
BASELINE_FILE = "baseline_features.csv"
OUTPUT_FILE = "rfe_ranking.csv"

N_FEATURES_TO_SELECT = 35


print("========================================")
print("RFE FEATURE SELECTION")
print("========================================")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Training dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Shape:", df.shape)


baseline_df = pd.read_csv(BASELINE_FILE)

baseline_features = baseline_df["Feature"].tolist()

print("\nBaseline features:", len(baseline_features))
print("Target variable:", "Label")


X = df[baseline_features]
y = df["Label"]

print("\nSamples:", X.shape[0])
print("Features:", X.shape[1])
print("Target classes:", y.nunique())


missing_values = X.isna().sum().sum()
infinite_values = np.isinf(X).sum().sum()

print("\nMissing values:", missing_values)
print("Infinite values:", infinite_values)

if missing_values > 0:
    raise ValueError("Dataset contains missing values.")

if infinite_values > 0:
    raise ValueError("Dataset contains infinite values.")

print("Data quality check passed.")


y_encoded, class_labels = pd.factorize(y)

print("\nClasses encoded:", len(np.unique(y_encoded)))

for i, label in enumerate(class_labels):
    print(f"{i}: {label}")


scaler = StandardScaler()

print("\nStandardizing features...")

X_scaled = scaler.fit_transform(X)

print("Feature standardization completed.")


print("\n========================================")
print("RFE CONFIGURATION")
print("========================================")

print("Estimator: Logistic Regression")
print("Regularization: L2")
print("Solver: lbfgs")
print("Features before RFE:", X.shape[1])
print("Features to select:", N_FEATURES_TO_SELECT)
print("RFE step: 1")


estimator = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)


print("\n========================================")
print("RUNNING RFE")
print("========================================")

rfe = RFE(
    estimator=estimator,
    n_features_to_select=N_FEATURES_TO_SELECT,
    step=1
)

rfe.fit(X_scaled, y_encoded)

print("RFE completed successfully.")


rfe_ranking = pd.DataFrame({
    "Feature": baseline_features,
    "RFE_Ranking": rfe.ranking_,
    "Selected": rfe.support_
})


rfe_ranking = rfe_ranking.sort_values(
    by=["RFE_Ranking", "Feature"],
    ascending=[True, True]
).reset_index(drop=True)


rfe_ranking.insert(
    0,
    "Rank",
    range(1, len(rfe_ranking) + 1)
)


print("\n========================================")
print("RFE FEATURE RANKING")
print("========================================")

print(rfe_ranking.to_string(index=False))


selected_features = rfe_ranking[
    rfe_ranking["Selected"] == True
]["Feature"].tolist()


print("\n========================================")
print("SELECTED FEATURES")
print("========================================")

print("Selected features:", len(selected_features))

for i, feature in enumerate(selected_features, 1):
    print(f"{i:2d}. {feature}")


rfe_ranking.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("OUTPUT SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Features ranked:", len(rfe_ranking))
print("Features selected:", len(selected_features))


print("\n========================================")
print("RFE ANALYSIS SUMMARY")
print("========================================")

print("Original features:", 78)
print("Constant features removed:", 8)
print("Baseline features:", len(baseline_features))
print("RFE selected features:", len(selected_features))
print(
    "Features removed by RFE:",
    len(baseline_features) - len(selected_features)
)

print("\n========================================")
print("RFE FEATURE SELECTION COMPLETE")
print("========================================")