import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path

# Path Configuration 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_raw.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "jobs_clean.csv"

#Step-1:Load and Basic CleanUP
def load_and_clean(df):
    print("Check whether all the data is loaded...\nLoaded {len(df)} rows")
    # Strip 'jsearch:' prefix from source_platform
    df["source_platform"] = df["source_platform"].str.replace(
        r"^jsearch:", "", regex=True
    ).str.strip()
    def clean_title(title):
        if pd.isna(title):
            return title
        title = str(title).strip()
        # Comprehensive list compiled from job market data
        keywords = ["Junior ML Engineer", "GenAI Engineer", "AI Engineer",
                    "LLM Engineer", "Data Scientist", "ML Engineer"]
        
        for kw in keywords:
            if title.startswith(kw) and len(title) > len(kw):
                cleaned = title[len(kw):].strip().lstrip("-–— ").strip()
                return cleaned if len(cleaned) > 3 else title
        return title

    df["job_title"] = df["job_title"].apply(clean_title)
    df["job_title"] = df["job_title"].str.strip()
    
    before = len(df)
    df = df.drop_duplicates(subset=["job_url"], keep="first")
    print(f"Removed {before - len(df)} duplicate URLs → {len(df)} rows")

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip() if c.dtype == "object" else c)

    return df

if __name__ == "__main__":
    print("Job Market Intel — Data Cleaning Pipeline")

    df = pd.read_csv(RAW_PATH)

    df = load_and_clean(df)