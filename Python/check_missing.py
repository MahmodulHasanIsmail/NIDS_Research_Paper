from pathlib import Path
import pandas as pd
import glob
import os

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "MachineLearningCVE"

files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))

for file in files:
    print("\n" + os.path.basename(file))

    df = pd.read_csv(file)
    missing = df.isna().sum()

    print(missing[missing > 0])