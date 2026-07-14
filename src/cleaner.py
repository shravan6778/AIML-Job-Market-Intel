import pandas as pd
import numpy as np
import re
from datetime import datetime

# Path Configuration 
RAW_PATH   = "data/raw/jobs_raw.csv"
CLEAN_PATH = "data/cleaned/jobs_clean.csv"

if __name__ == "__main__":
    print("Job Market Intel — Data Cleaning Pipeline")

    df = pd.read_csv(RAW_PATH)
