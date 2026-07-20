import pandas as pd
import numpy as np
from pathlib import Path

#CONFIG 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "jobs_clean.csv"

def load_data():
    df = pd.read_csv(CLEAN_PATH)
    df["posted_date"] = pd.to_datetime(df["posted_date"], utc=True, errors="coerce")
    print(f"Loaded {len(df)} rows for analysis\n")
    return df

if __name__ == "__main__":
    print("Job Market Intel — EDA & Analysis")

    df = load_data()
    