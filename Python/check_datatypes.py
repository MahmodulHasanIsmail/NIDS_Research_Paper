
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
# FIND CSV FILES
# ============================================================

files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))

reference_file = files[0]

reference_df = pd.read_csv(reference_file, nrows=1000)
reference_types = reference_df.dtypes.astype(str)

print("Reference:", os.path.basename(reference_file))


# ============================================================
# COMPARE DATATYPES
# ============================================================

for file in files[1:]:
    current_df = pd.read_csv(file, nrows=1000)
    current_types = current_df.dtypes.astype(str)

    differences = pd.DataFrame({
        "Reference": reference_types,
        "Current": current_types
    })

    differences = differences[
        differences["Reference"] != differences["Current"]
    ]

    print("\n" + os.path.basename(file))

    if differences.empty:
        print("No datatype differences")
    else:
        print(differences)