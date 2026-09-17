
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "Processed_Data" / "train_CICIDS2017.csv"
INVENTORY_FILE = ROOT / "CSV" / "feature_inventory.csv"
OUTPUT_FILE = ROOT / "CSV" / "baseline_features.csv"


# ============================================================
# 1. LOAD FEATURE INVENTORY
# ============================================================

print("========================================")
print("BASELINE FEATURE SET CREATION")
print("========================================")

print("\nLoading feature inventory...")

inventory = pd.read_csv(INVENTORY_FILE)

print("Feature inventory loaded successfully.")
print("Original input features:", len(inventory))


# ============================================================
# 2. IDENTIFY CONSTANT FEATURES
# ============================================================

constant_features = inventory[
    inventory["Unique_Values"] <= 1
]["Feature"].tolist()

print("\n========================================")
print("REMOVING CONSTANT FEATURES")
print("========================================")

print("Constant features:", len(constant_features))

for feature in constant_features:
    print("-", feature)


# ============================================================
# 3. CREATE BASELINE FEATURE SET
# ============================================================

baseline = inventory[
    ~inventory["Feature"].isin(constant_features)
].copy()

baseline["Baseline_Status"] = "Retained"


# ============================================================
# 4. DISPLAY BASELINE SUMMARY
# ============================================================

print("\n========================================")
print("BASELINE FEATURE SET")
print("========================================")

print("Original features:", len(inventory))
print("Constant features removed:", len(constant_features))
print("Baseline features retained:", len(baseline))

print("\nBaseline features:")

for i, feature in enumerate(baseline["Feature"], start=1):
    print(f"{i:2}. {feature}")


# ============================================================
# 5. SAVE BASELINE FEATURE LIST
# ============================================================

baseline[
    [
        "Feature_Number",
        "Feature",
        "Data_Type",
        "Unique_Values",
        "Baseline_Status"
    ]
].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("BASELINE FEATURE SET SAVED")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Baseline features:", len(baseline))

print("\n========================================")
print("BASELINE CREATION COMPLETE")
print("========================================")