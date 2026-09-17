
from pathlib import Path

import pandas as pd
import glob
import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "MachineLearningCVE"


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))

for file in files:
    df = pd.read_csv(file)

    numeric_df = df.select_dtypes(include="number")
    infinite_counts = np.isinf(numeric_df).sum()

    infinite_counts = infinite_counts[infinite_counts > 0]

    print("\n" + os.path.basename(file))

    if infinite_counts.empty:
        print("No infinite values")
    else:
        print(infinite_counts)