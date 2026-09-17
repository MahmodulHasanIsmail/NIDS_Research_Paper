import pandas as pd
import os
from sklearn.model_selection import train_test_split


INPUT_FILE = "cleaned_CICIDS2017.csv"

TRAIN_FILE = "train_CICIDS2017.csv"
TEST_FILE = "test_CICIDS2017.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


print("========================================")
print("DATASET SPLITTING")
print("========================================")

if not os.path.exists(INPUT_FILE):
    print("ERROR: Input dataset not found.")
    print("Expected:", INPUT_FILE)
    raise SystemExit

print("Input dataset:", INPUT_FILE)


print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Shape:", df.shape)


X = df.drop(columns=["Label"])
y = df["Label"]

print("\n========================================")
print("FEATURE / TARGET SEPARATION")
print("========================================")

print("Input features:", X.shape[1])
print("Target variable: Label")
print("Target classes:", y.nunique())


print("\n========================================")
print("ORIGINAL CLASS DISTRIBUTION")
print("========================================")

print(y.value_counts())


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Training proportion: 80%")
print("Testing proportion: 20%")
print("Random state:", RANDOM_STATE)
print("Stratification: Label")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


train_df = X_train.copy()
train_df["Label"] = y_train

test_df = X_test.copy()
test_df["Label"] = y_test

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


print("\n========================================")
print("SPLIT RESULTS")
print("========================================")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("Training columns:", len(train_df.columns))
print("Testing columns:", len(test_df.columns))

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


print("\n========================================")
print("TRAINING CLASS DISTRIBUTION")
print("========================================")

print(train_df["Label"].value_counts())


print("\n========================================")
print("TESTING CLASS DISTRIBUTION")
print("========================================")

print(test_df["Label"].value_counts())


print("\n========================================")
print("SAVING TRAINING DATASET")
print("========================================")

train_df.to_csv(TRAIN_FILE, index=False)

print("Training dataset saved as:", TRAIN_FILE)


print("\n========================================")
print("SAVING TESTING DATASET")
print("========================================")

test_df.to_csv(TEST_FILE, index=False)

print("Testing dataset saved as:", TEST_FILE)


print("\n========================================")
print("FILE VERIFICATION")
print("========================================")

print(
    "Training file exists:",
    os.path.exists(TRAIN_FILE)
)

print(
    "Testing file exists:",
    os.path.exists(TEST_FILE)
)


print("\n========================================")
print("FINAL VERIFICATION")
print("========================================")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Total rows:", len(train_df) + len(test_df))

print("Expected total rows:", len(df))

print(
    "Row count preserved:",
    len(train_df) + len(test_df) == len(df)
)

print(
    "Training missing values:",
    train_df.isna().sum().sum()
)

print(
    "Testing missing values:",
    test_df.isna().sum().sum()
)

print("\n========================================")
print("DATASET SPLITTING COMPLETE")
print("========================================")