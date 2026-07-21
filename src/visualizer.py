import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "jobs_clean.csv"
OUTPUT_PATH = PROJECT_ROOT/ "outputs"/"report_figures"/"jm_intelligence_report.png"

YOUR_SKILLS = [
    "python", "numpy", "pandas", "matplotlib",
    "fastapi", "git", "sql",
]

# PALETTE
C_TEAL      = "#1D9E75"
C_TEAL_LIGHT= "#9FE1CB"
C_AMBER     = "#BA7517"
C_AMBER_LIGHT="#FAC775"
C_CORAL     = "#D85A30"
C_PURPLE    = "#7F77DD"
C_BLUE      = "#378ADD"
C_GRAY      = "#888780"
C_GRAY_LIGHT= "#D3D1C7"
BG          = "#F8F7F4"
CARD_BG     = "#FFFFFF"
TEXT_DARK   = "#2C2C2A"
TEXT_MID    = "#5F5E5A"

# LOAD & PREP 
def load_and_prep():
    df = pd.read_csv(CLEAN_PATH)
    df["posted_date"] = pd.to_datetime(df["posted_date"], utc=True, errors="coerce")
    return df

if __name__ == "__main__":
    print("Job Market Intel — Visualization")
    df = load_and_prep()