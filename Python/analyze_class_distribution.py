
from pathlib import Path
import pandas as pd
import os

# CONFIGURATION
ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
TEST_FILE = ROOT / "Processed_Data" / "test_CICIDS2017.csv"

OUTPUT_DIR = ROOT / "CSV"

# 1. LOAD DATASETS
print("========================================")
print("CLASS DISTRIBUTION ANALYSIS")
print("========================================")

print("\nLoading training dataset...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)

print("Training dataset loaded.")
print("Training shape:", train_df.shape)


print("\nLoading testing dataset...")

test_df = pd.read_csv(TEST_FILE, low_memory=False)

print("Testing dataset loaded.")
print("Testing shape:", test_df.shape)

# 2. TRAINING CLASS DISTRIBUTION
print("\n========================================")
print("TRAINING CLASS DISTRIBUTION")
print("========================================")

train_counts = train_df["Label"].value_counts()
train_percentages = train_df["Label"].value_counts(normalize=True) * 100

train_distribution = pd.DataFrame({
    "Class": train_counts.index,
    "Count": train_counts.values,
    "Percentage": train_percentages.values
})

print(train_distribution.to_string(index=False))

# 3. TESTING CLASS DISTRIBUTION
print("\n========================================")
print("TESTING CLASS DISTRIBUTION")
print("========================================")

test_counts = test_df["Label"].value_counts()
test_percentages = test_df["Label"].value_counts(normalize=True) * 100

test_distribution = pd.DataFrame({
    "Class": test_counts.index,
    "Count": test_counts.values,
    "Percentage": test_percentages.values
})

print(test_distribution.to_string(index=False))

# 4. CHECK CLASS PRESENCE
print("\n========================================")
print("CLASS PRESENCE CHECK")
print("========================================")

train_classes = set(train_df["Label"].unique())
test_classes = set(test_df["Label"].unique())

print("Classes in training set:", len(train_classes))
print("Classes in testing set:", len(test_classes))

missing_from_train = test_classes - train_classes
missing_from_test = train_classes - test_classes

print("\nClasses missing from training set:")
print(missing_from_train if missing_from_train else "None")

print("\nClasses missing from testing set:")
print(missing_from_test if missing_from_test else "None")

# 5. IMBALANCE RATIO
print("\n========================================")
print("CLASS IMBALANCE ANALYSIS")
print("========================================")

largest_class = train_counts.max()
smallest_class = train_counts.min()

largest_class_name = train_counts.idxmax()
smallest_class_name = train_counts.idxmin()

imbalance_ratio = largest_class / smallest_class

print("Largest training class:", largest_class_name)
print("Largest class count:", largest_class)

print("\nSmallest training class:", smallest_class_name)
print("Smallest class count:", smallest_class)

print("\nImbalance ratio:")
print(f"{imbalance_ratio:.2f}:1")

# 6. BENIGN VS ATTACK DISTRIBUTION
print("\n========================================")
print("BENIGN VS ATTACK DISTRIBUTION")
print("========================================")

train_benign = (train_df["Label"] == "BENIGN").sum()
train_attack = len(train_df) - train_benign

test_benign = (test_df["Label"] == "BENIGN").sum()
test_attack = len(test_df) - test_benign

print("\nTRAINING SET")
print("BENIGN:", train_benign)
print("ATTACK:", train_attack)

print("\nTesting set")
print("BENIGN:", test_benign)
print("ATTACK:", test_attack)


train_benign_percentage = train_benign / len(train_df) * 100
train_attack_percentage = train_attack / len(train_df) * 100

test_benign_percentage = test_benign / len(test_df) * 100
test_attack_percentage = test_attack / len(test_df) * 100

print("\nTRAINING PERCENTAGES")
print(f"BENIGN: {train_benign_percentage:.4f}%")
print(f"ATTACK: {train_attack_percentage:.4f}%")

print("\nTESTING PERCENTAGES")
print(f"BENIGN: {test_benign_percentage:.4f}%")
print(f"ATTACK: {test_attack_percentage:.4f}%")

# 7. SAVE DISTRIBUTION TABLES
train_distribution.to_csv(
    OUTPUT_DIR / "training_class_distribution.csv",
    index=False
)

test_distribution.to_csv(
    OUTPUT_DIR / "testing_class_distribution.csv",
    index=False
)

# 8. FINAL SUMMARY
print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print("Training distribution saved as:")
print(OUTPUT_DIR / "training_class_distribution.csv")

print("\nTesting distribution saved as:")
print(OUTPUT_DIR / "testing_class_distribution.csv")
