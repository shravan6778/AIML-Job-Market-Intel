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

# ANALYSIS FUNCTIONS (inline, mirroring analyzer.py)
def get_top_skills(df, n=15):
    has_skills = df[df["skills_extracted"] != "Not specified"]
    series = (
        has_skills["skills_extracted"]
        .str.split(",").explode()
        .str.strip().str.lower()
    )
    counts = series.value_counts().head(n)
    return pd.DataFrame({"skill": counts.index, "count": counts.values})

def get_fresher_companies(df, n=12):
    mask = df["experience_level"].isin(["Fresher (0-2 yrs)", "Internship"])
    counts = df[mask]["company"].value_counts()
    counts = counts[counts.index != "Unknown Company"].head(n)
    return pd.DataFrame({"company": counts.index, "count": counts.values})

def get_role_dist(df):
    counts = df[df["role_category"] != "Other"]["role_category"].value_counts()
    return pd.DataFrame({"role": counts.index, "count": counts.values})

def get_weekly_trend(df):
    dated = df[df["posted_date"].notna()].copy()
    # In load_and_prep(), add after parsing dates:
    df = df[df["posted_date"].dt.year >= 2025]
    dated["week"] = dated["posted_date"].dt.to_period("W").dt.start_time
    weekly = dated.groupby("week").size().reset_index(name="count").sort_values("week")
    weekly["rolling"] = np.convolve(weekly["count"].values, np.ones(3)/3, mode="same")
    return weekly

def get_cooccurrence(df, top_n=10):
    has = df[df["skills_extracted"] != "Not specified"].copy()
    has["skills_list"] = has["skills_extracted"].str.split(",").apply(
        lambda l: [s.strip().lower() for s in l]
    )
    top_skills = (
        has["skills_list"].explode().value_counts().head(top_n).index.tolist()
    )
    idx = {s: i for i, s in enumerate(top_skills)}
    mat = np.zeros((top_n, top_n), dtype=int)
    for skills in has["skills_list"]:
        present = [s for s in skills if s in idx]
        for i, s1 in enumerate(present):
            for s2 in present[i:]:
                r, c = idx[s1], idx[s2]
                mat[r][c] += 1
                if r != c:
                    mat[c][r] += 1
    np.fill_diagonal(mat, 0)   # zero diagonal for cleaner heatmap
    return mat, top_skills

def get_skill_gap(df, n=12):
    has = df[df["skills_extracted"] != "Not specified"]
    series = (
        has["skills_extracted"].str.split(",").explode()
        .str.strip().str.lower()
    )
    top = series.value_counts().head(n)
    your_lower = [s.lower() for s in YOUR_SKILLS]
    return pd.DataFrame({
        "skill":   top.index,
        "count":   top.values,
        "you_have": [s in your_lower for s in top.index],
    })
    
# FIGURE
def build_report(df):
    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        "Hyderabad AI/ML Job Market Intelligence Report  ·  2025–2026",
        fontsize=20, fontweight="bold", color=TEXT_DARK, y=0.98
    )
    fig.text(
        0.5, 0.955,
        f"Based on {len(df):,} real job postings · {df['company'].nunique()} companies · "
        f"Scraped July 2026",
        ha="center", fontsize=11, color=TEXT_MID
    )

    gs = GridSpec(
        2, 3,
        figure=fig,
        left=0.06, right=0.97,
        top=0.92, bottom=0.07,
        hspace=0.42, wspace=0.35
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRAY_LIGHT)
            spine.set_linewidth(0.8)
            
if __name__ == "__main__":
    print("Job Market Intel — Visualization")
    df = load_and_prep()