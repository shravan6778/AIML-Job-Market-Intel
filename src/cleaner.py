import pandas as pd
import numpy as np
import re
from datetime import datetime

# Path Configuration 
RAW_PATH   = "data/raw/jobs_raw.csv"
CLEAN_PATH = "data/cleaned/jobs_clean.csv"

#Step-1:Load and Basic CleanUP
def load_and_clean(df):
    print("Check whether all the data is loaded...\nLoaded {len(df)} rows")
    # Strip 'jsearch:' prefix from source_platform
    df["source_platform"] = df["source_platform"].str.replace(
        r"^jsearch:", "", regex=True
    ).str.strip()
    

if __name__ == "__main__":
    print("Job Market Intel — Data Cleaning Pipeline")

    df = pd.read_csv(RAW_PATH)

    df = load_and_clean(df)