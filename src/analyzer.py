import pandas as pd
import numpy as np
from pathlib import Path

#CONFIG 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "jobs_clean.csv"

if __name__ == "__main__":
    print("Job Market Intel — EDA & Analysis")
