from pathlib import Path
import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
TEST_FILE = ROOT / "Processed_Data" / "test_CICIDS2017.csv"
FEATURE_FILE = ROOT / "CSV" / "evaluation_feature_sets.csv"
RESULT_FILE = ROOT / "CSV" / "evaluation_results.csv"


# ============================================================
# START
# ============================================================

print("=" * 50)
print("ML MODEL EVALUATION")
print("=" * 50)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training shape: {train_df.shape}")
print(f"Testing shape : {test_df.shape}")

TARGET = "Label"


# ============================================================
# TARGET ENCODING
# ============================================================

print("\nEncoding target labels...")

encoder = LabelEncoder()

y_train = encoder.fit_transform(train_df[TARGET])
y_test = encoder.transform(test_df[TARGET])

print(f"Classes: {len(encoder.classes_)}")


# ============================================================
# LOAD FEATURE SETS
# ============================================================

print("\nLoading feature sets...")

feature_df = pd.read_csv(FEATURE_FILE)

feature_sets = {}

for feature_set in feature_df["Feature_Set"].unique():

    features = feature_df[
        feature_df["Feature_Set"] == feature_set
    ]["Feature"].tolist()

    feature_sets[feature_set] = features

    print(f"{feature_set}: {len(features)} features")


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        l1_ratio=0,
        solver="saga",
        max_iter=500,
        n_jobs=-1,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=42,
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )
}


# ============================================================
# RESUME SUPPORT
# ============================================================

if os.path.exists(RESULT_FILE):

    results_df = pd.read_csv(RESULT_FILE)

    completed = set(
        zip(
            results_df["Feature_Set"],
            results_df["Model"]
        )
    )

    print(
        f"\nExisting results found: "
        f"{len(results_df)} experiments"
    )

else:

    results_df = pd.DataFrame()

    completed = set()


# ============================================================
# EVALUATION
# ============================================================

total_experiments = len(feature_sets) * len(models)

experiment_number = len(completed)

print("\n" + "=" * 50)
print("STARTING EVALUATION")
print("=" * 50)

print(f"Total experiments: {total_experiments}")


for feature_set_name, features in feature_sets.items():

    print("\n" + "=" * 50)
    print(f"FEATURE SET: {feature_set_name}")
    print(f"Number of features: {len(features)}")
    print("=" * 50)

    X_train = train_df[features]
    X_test = test_df[features]


    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print("Standardizing features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    for model_name, model in models.items():

        if (feature_set_name, model_name) in completed:

            print(
                f"\nSKIPPING: "
                f"{feature_set_name} + {model_name}"
            )

            continue

        experiment_number += 1

        print("\n" + "-" * 50)
        print(
            f"EXPERIMENT {experiment_number}/"
            f"{total_experiments}"
        )
        print(f"Feature Set: {feature_set_name}")
        print(f"Model      : {model_name}")
        print("-" * 50)


        # ====================================================
        # SELECT DATA
        # ====================================================

        if model_name in [
            "Logistic Regression",
            "Linear SVM",
            "KNN"
        ]:

            Xtr = X_train_scaled
            Xte = X_test_scaled

        else:

            Xtr = X_train
            Xte = X_test


        # ====================================================
        # TRAINING
        # ====================================================

        print("Training...")

        start_train = time.time()

        model.fit(Xtr, y_train)

        train_time = time.time() - start_train

        print(
            f"Training completed: "
            f"{train_time:.2f} seconds"
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        print("Predicting...")

        start_predict = time.time()

        y_pred = model.predict(Xte)

        predict_time = time.time() - start_predict

        print(
            f"Prediction completed: "
            f"{predict_time:.2f} seconds"
        )


        # ====================================================
        # METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        print("\nRESULTS")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


        # ====================================================
        # SAVE RESULT IMMEDIATELY
        # ====================================================

        new_result = pd.DataFrame([{

            "Feature_Set": feature_set_name,
            "Feature_Count": len(features),
            "Model": model_name,

            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,

            "Training_Time_Seconds": train_time,
            "Prediction_Time_Seconds": predict_time

        }])

        results_df = pd.concat(
            [results_df, new_result],
            ignore_index=True
        )

        results_df.to_csv(
            RESULT_FILE,
            index=False
        )

        print(
            f"\nSaved to {RESULT_FILE}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("ALL AVAILABLE EXPERIMENTS COMPLETE")
print("=" * 50)

print("\nResults saved to:")
print(RESULT_FILE)

print("\nFinal results:")
print(results_df.to_string(index=False))

print("\n" + "=" * 50)
print("EVALUATION COMPLETE")
print("=" * 50)