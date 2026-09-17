from pathlib import Path

import pandas as pd
import numpy as np
import glob
import os


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "MachineLearningCVE"

OUTPUT_FILE = ROOT / "Processed_Data" / "cleaned_CICIDS2017.csv"


# ============================================================
# 1. FIND ALL CSV FILES
# ============================================================

files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))

print("Number of CSV files found:", len(files))

for file in files:
    print(os.path.basename(file))


# ============================================================
# 2. LOAD AND COMBINE ALL CSV FILES
# ============================================================

dataframes = []

for file in files:
    print(f"\nLoading: {os.path.basename(file)}")

    df = pd.read_csv(file, low_memory=False)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    dataframes.append(df)


combined_df = pd.concat(dataframes, ignore_index=True)


print("\n========================================")
print("COMBINED DATASET")
print("========================================")
print("Rows:", len(combined_df))
print("Columns:", len(combined_df.columns))
print("Shape:", combined_df.shape)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

combined_df.columns = combined_df.columns.str.strip()


# ============================================================
# 4. CLEAN LABEL VALUES
# ============================================================

label_fix_map = {
    "Web Attack � Brute Force": "Web Attack-Brute Force",
    "Web Attack � XSS": "Web Attack-XSS",
    "Web Attack � Sql Injection": "Web Attack-SQL Injection"
}

combined_df["Label"] = combined_df["Label"].replace(label_fix_map)

combined_df["Label"] = (
    combined_df["Label"]
    .astype(str)
    .str.strip()
)


print("\n========================================")
print("LABEL DISTRIBUTION AFTER CLEANING")
print("========================================")

print(combined_df["Label"].value_counts())


# ============================================================
# 5. REMOVE EXACT DUPLICATE ROWS
# ============================================================

before_duplicates = len(combined_df)

combined_df = combined_df.drop_duplicates().reset_index(drop=True)

after_duplicates = len(combined_df)


print("\n========================================")
print("DUPLICATE REMOVAL")
print("========================================")

print(
    "Rows before duplicate removal:",
    before_duplicates
)

print(
    "Duplicate rows removed:",
    before_duplicates - after_duplicates
)

print(
    "Rows after duplicate removal:",
    after_duplicates
)


# ============================================================
# 6. CHECK INFINITE AND MISSING VALUES
# ============================================================

print("\n========================================")
print("INFINITE AND MISSING VALUE CHECK")
print("========================================")


# Select numeric columns
numeric_columns = combined_df.select_dtypes(
    include="number"
).columns


# Count infinite values
infinite_mask = np.isinf(
    combined_df[numeric_columns]
)

print(
    "Total infinite values:",
    infinite_mask.sum().sum()
)


# Replace positive and negative infinity with NaN
combined_df[numeric_columns] = combined_df[
    numeric_columns
].replace(
    [np.inf, -np.inf],
    np.nan
)


# Count missing values
missing_values = combined_df.isna().sum()


print("\nColumns containing missing values:")
print(
    missing_values[
        missing_values > 0
    ]
)

print(
    "\nTotal missing values:",
    missing_values.sum()
)


# ============================================================
# 7. ANALYZE ROWS CONTAINING MISSING VALUES
# ============================================================

rows_with_missing = combined_df.isna().any(axis=1).sum()

print(
    "\nRows containing at least one missing value:",
    rows_with_missing
)


print("\nMissing-value rows by column:")

for column in combined_df.columns:

    count = combined_df[column].isna().sum()

    if count > 0:
        print(
            column,
            "->",
            count,
            "rows"
        )


# Check whether both affected columns are missing

flow_bytes_missing = combined_df[
    "Flow Bytes/s"
].isna()

flow_packets_missing = combined_df[
    "Flow Packets/s"
].isna()


both_missing = (
    flow_bytes_missing &
    flow_packets_missing
).sum()


only_flow_bytes = (
    flow_bytes_missing &
    ~flow_packets_missing
).sum()


only_flow_packets = (
    flow_packets_missing &
    ~flow_bytes_missing
).sum()


print("\n========================================")
print("MISSING VALUE ROW ANALYSIS")
print("========================================")

print(
    "Both Flow Bytes/s and Flow Packets/s missing:",
    both_missing
)

print(
    "Only Flow Bytes/s missing:",
    only_flow_bytes
)

print(
    "Only Flow Packets/s missing:",
    only_flow_packets
)


# ============================================================
# 8. ANALYZE LABELS OF MISSING-VALUE ROWS
# ============================================================

missing_rows = combined_df[
    combined_df["Flow Bytes/s"].isna() &
    combined_df["Flow Packets/s"].isna()
]


print("\n========================================")
print("LABEL DISTRIBUTION OF MISSING-VALUE ROWS")
print("========================================")

print(
    "Number of affected rows:",
    len(missing_rows)
)

print()

print(
    missing_rows["Label"].value_counts()
)


# ============================================================
# 9. REMOVE ROWS CONTAINING MISSING VALUES
# ============================================================

before_missing_removal = len(combined_df)

combined_df = combined_df.dropna().reset_index(drop=True)

after_missing_removal = len(combined_df)


print("\n========================================")
print("MISSING VALUE REMOVAL")
print("========================================")

print(
    "Rows before missing-value removal:",
    before_missing_removal
)

print(
    "Rows removed:",
    before_missing_removal -
    after_missing_removal
)

print(
    "Rows after missing-value removal:",
    after_missing_removal
)


# ============================================================
# 10. FINAL MISSING VALUE CHECK
# ============================================================

print("\n========================================")
print("FINAL MISSING VALUE CHECK")
print("========================================")

print(
    "Total missing values remaining:",
    combined_df.isna().sum().sum()
)


# ============================================================
# 11. FINAL DATASET SHAPE
# ============================================================

print("\n========================================")
print("FINAL CLEANED DATASET")
print("========================================")

print(
    "Rows:",
    len(combined_df)
)

print(
    "Columns:",
    len(combined_df.columns)
)

print(
    "Shape:",
    combined_df.shape
)


# ============================================================
# 12. FEATURE VALIDATION
# ============================================================

print("\n========================================")
print("FEATURE VALIDATION")
print("========================================")


# Separate target variable
X = combined_df.drop(
    columns=["Label"]
)

y = combined_df["Label"]


print(
    "Number of input features:",
    X.shape[1]
)

print(
    "Target variable:",
    "Label"
)


# Check numeric and non-numeric columns

numeric_features = X.select_dtypes(
    include=np.number
).columns

non_numeric_features = X.select_dtypes(
    exclude=np.number
).columns


print(
    "\nNumeric features:",
    len(numeric_features)
)

print(
    "Non-numeric features:",
    len(non_numeric_features)
)


if len(non_numeric_features) > 0:

    print("\nNon-numeric columns:")

    print(
        list(non_numeric_features)
    )

else:

    print(
        "All input features are numeric."
    )


# Final missing-value check

print(
    "\nMissing values in input features:",
    X.isna().sum().sum()
)


# Final infinite-value check

print(
    "Infinite values in input features:",
    np.isinf(X).sum().sum()
)


# Check target variable

print(
    "\nNumber of target classes:",
    y.nunique()
)


print(
    "\nTarget classes:"
)

print(
    sorted(y.unique())
)


print("\n========================================")
print("FEATURE VALIDATION COMPLETE")
print("========================================")


# ============================================================
# 13. SAVE FINAL CLEANED DATASET
# ============================================================

print("\n========================================")
print("SAVING CLEANED DATASET")
print("========================================")


combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Cleaned dataset saved as:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(combined_df)
)

print(
    "Columns:",
    len(combined_df.columns)
)

print(
    "File saved successfully."
)


# ============================================================
# 14. FINAL DATASET VERIFICATION
# ============================================================

print("\n========================================")
print("FINAL DATASET VERIFICATION")
print("========================================")


print(
    "File exists:",
    os.path.exists(OUTPUT_FILE)
)

print(
    "Final rows:",
    len(combined_df)
)

print(
    "Final columns:",
    len(combined_df.columns)
)

print(
    "Final shape:",
    combined_df.shape
)

print(
    "Missing values:",
    combined_df.isna().sum().sum()
)

print(
    "Infinite values:",
    np.isinf(
        combined_df.select_dtypes(
            include=np.number
        )
    ).sum().sum()
)


print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")