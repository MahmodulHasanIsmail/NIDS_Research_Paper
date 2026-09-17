
from pathlib import Path

import pandas as pd
import glob
import os


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "MachineLearningCVE"


# ============================================================
# CROSS-FILE DUPLICATE ANALYSIS
# ============================================================

files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))

seen_hashes = set()
total_cross_duplicates = 0

print("Cross-file duplicate analysis")
print("=" * 60)

for file in files:
    filename = os.path.basename(file)
    file_cross_duplicates = 0
    file_rows = 0

    file_hashes = set()

    for chunk in pd.read_csv(file, chunksize=50000):
        file_rows += len(chunk)

        row_hashes = pd.util.hash_pandas_object(
            chunk,
            index=False
        )

        for h in row_hashes:
            if h in seen_hashes:
                file_cross_duplicates += 1

        file_hashes.update(row_hashes)

    total_cross_duplicates += file_cross_duplicates

    print(
        f"{filename} -> "
        f"Rows: {file_rows:,} | "
        f"Cross-file duplicates: {file_cross_duplicates:,}"
    )

    seen_hashes.update(file_hashes)

print("=" * 60)
print(f"TOTAL cross-file duplicates: {total_cross_duplicates:,}")