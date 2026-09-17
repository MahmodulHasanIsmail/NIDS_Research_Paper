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

    infinite_count = np.isinf(
        df.select_dtypes(include="number")
    ).sum().sum()

    print(
        os.path.basename(file),
        "-> Infinite values:",
        infinite_count
    )