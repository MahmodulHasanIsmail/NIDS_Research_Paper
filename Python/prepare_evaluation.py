from pathlib import Path
import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
TEST_FILE = ROOT / "Processed_Data" / "test_CICIDS2017.csv"

BASELINE_FILE = ROOT / "CSV" / "baseline_features.csv"

MI_FILE = ROOT / "CSV" / "mutual_information_scores.csv"
ANOVA_FILE = ROOT / "CSV" / "anova_scores.csv"
CHI2_FILE = ROOT / "CSV" / "chi_square_scores.csv"
RFE_FILE = ROOT / "CSV" / "rfe_ranking.csv"
L1_FILE = ROOT / "CSV" / "l1_ranking.csv"
RF_FILE = ROOT / "CSV" / "rf_ranking.csv"

OUTPUT_FILE = ROOT / "CSV" / "evaluation_feature_sets.csv"

N_FEATURES = 35


# ============================================================
# HEADER
# ============================================================

print("========================================")
print("EVALUATION SETUP")
print("========================================")


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    TRAIN_FILE,
    TEST_FILE,
    BASELINE_FILE,
    MI_FILE,
    ANOVA_FILE,
    CHI2_FILE,
    RFE_FILE,
    L1_FILE,
    RF_FILE
]

for file in required_files:

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Required file not found: {file}"
        )

    print(f"FOUND: {file}")

print("\nAll required files found.")


# ============================================================
# 2. LOAD DATASET HEADERS ONLY
# ============================================================

print("\n========================================")
print("CHECKING DATASET FEATURES")
print("========================================")

train_columns = pd.read_csv(
    TRAIN_FILE,
    nrows=0
).columns.tolist()

test_columns = pd.read_csv(
    TEST_FILE,
    nrows=0
).columns.tolist()

print("Training dataset columns:", len(train_columns))
print("Testing dataset columns:", len(test_columns))

if train_columns != test_columns:
    raise ValueError(
        "Training and testing dataset columns do not match."
    )

print("Training and testing feature structures match.")


# ============================================================
# 3. LOAD BASELINE FEATURES
# ============================================================

print("\n========================================")
print("LOADING FEATURE SETS")
print("========================================")

baseline_df = pd.read_csv(BASELINE_FILE)

baseline_features = baseline_df["Feature"].tolist()

print(
    "Baseline:",
    len(baseline_features),
    "features"
)


# ============================================================
# 4. FUNCTION TO LOAD RANKING FILE
# ============================================================

def load_selected_features(
    filename,
    method_name
):

    df = pd.read_csv(filename)

    if "Feature" not in df.columns:
        raise ValueError(
            f"'Feature' column not found in {filename}"
        )

    # If Selected column exists, use it
    if "Selected" in df.columns:

        selected = df[
            df["Selected"] == True
        ]["Feature"].tolist()

    else:

        # Otherwise take the top 35 rows
        selected = df[
            "Feature"
        ].head(N_FEATURES).tolist()

    if len(selected) != N_FEATURES:

        raise ValueError(
            f"{method_name} has "
            f"{len(selected)} selected features. "
            f"Expected {N_FEATURES}."
        )

    print(
        f"{method_name}:",
        len(selected),
        "features"
    )

    return selected


# ============================================================
# 5. LOAD ALL FEATURE-SELECTION RESULTS
# ============================================================

mi_features = load_selected_features(
    MI_FILE,
    "Mutual Information"
)

anova_features = load_selected_features(
    ANOVA_FILE,
    "ANOVA"
)

chi2_features = load_selected_features(
    CHI2_FILE,
    "Chi-Square"
)

rfe_features = load_selected_features(
    RFE_FILE,
    "RFE"
)

l1_features = load_selected_features(
    L1_FILE,
    "L1"
)

rf_features = load_selected_features(
    RF_FILE,
    "Random Forest"
)


# ============================================================
# 6. VERIFY FEATURES EXIST IN DATASET
# ============================================================

print("\n========================================")
print("FEATURE VALIDATION")
print("========================================")


feature_sets = {

    "Baseline": baseline_features,

    "Mutual Information": mi_features,

    "ANOVA": anova_features,

    "Chi-Square": chi2_features,

    "RFE": rfe_features,

    "L1": l1_features,

    "Random Forest": rf_features
}


for method, features in feature_sets.items():

    missing = [
        feature
        for feature in features
        if feature not in train_columns
    ]

    if missing:

        print(
            f"\nERROR: {method} contains "
            f"missing features:"
        )

        for feature in missing:
            print(" -", feature)

        raise ValueError(
            f"{method} contains invalid features."
        )

    print(
        f"{method}: all features validated."
    )


# ============================================================
# 7. CREATE MASTER FEATURE SET FILE
# ============================================================

print("\n========================================")
print("CREATING MASTER FEATURE SET")
print("========================================")

rows = []

for method, features in feature_sets.items():

    for rank, feature in enumerate(
        features,
        start=1
    ):

        rows.append({

            "Feature_Set": method,

            "Rank": rank,

            "Feature": feature,

            "Selected": True

        })


evaluation_df = pd.DataFrame(rows)


evaluation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n========================================")
print("EVALUATION FEATURE SET SUMMARY")
print("========================================")

for method, features in feature_sets.items():

    print(
        f"{method:20s}: "
        f"{len(features)} features"
    )


print("\nTotal feature sets:",
      len(feature_sets))

print(
    "Output file:",
    OUTPUT_FILE
)


# ============================================================
# 9. FINAL CHECK
# ============================================================

print("\n========================================")
print("EVALUATION SETUP COMPLETE")
print("========================================")

print(
    "Baseline features:",
    len(baseline_features)
)

print(
    "Features per selected method:",
    N_FEATURES
)

print(
    "Number of selection methods:",
    len(feature_sets) - 1
)

print(
    "Master feature file:",
    OUTPUT_FILE
)

print("\nReady for ML model evaluation.")