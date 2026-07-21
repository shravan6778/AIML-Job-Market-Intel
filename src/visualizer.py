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
    # PANEL 1: TOP 15 SKILLS 
    skills_df = get_top_skills(df, 15)
    skills_df = skills_df.sort_values("count")          # ascending for barh

    colors_p1 = [C_TEAL if c >= 20 else C_TEAL_LIGHT for c in skills_df["count"]]
    bars = ax1.barh(skills_df["skill"], skills_df["count"],
                    color=colors_p1, height=0.65, edgecolor="none")

    for bar, val in zip(bars, skills_df["count"]):
        ax1.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", ha="left",
                 fontsize=8.5, color=TEXT_MID)

    ax1.set_title("Top 15 In-Demand Skills", fontsize=12,
                  fontweight="bold", color=TEXT_DARK, pad=10)
    ax1.set_xlabel("No. of job postings", fontsize=9, color=TEXT_MID)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.tick_params(axis="x", labelsize=8, colors=TEXT_MID)
    ax1.set_xlim(0, skills_df["count"].max() * 1.2)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4, color=C_GRAY_LIGHT)
    ax1.set_axisbelow(True)
    high_patch = mpatches.Patch(color=C_TEAL, label="High demand (≥20 jobs)")
    ax1.legend(handles=[high_patch], fontsize=8, loc="lower right",
               framealpha=0.8, edgecolor=C_GRAY_LIGHT)
    
    # PANEL 2: ROLE DISTRIBUTION
    role_df = get_role_dist(df).sort_values("count")
    role_colors = [C_PURPLE, C_BLUE, C_TEAL, C_AMBER, C_CORAL,
                   C_GRAY, C_GRAY_LIGHT, C_TEAL_LIGHT][:len(role_df)]

    bars2 = ax2.barh(role_df["role"], role_df["count"],
                     color=role_colors[::-1], height=0.6, edgecolor="none")

    total_roles = role_df["count"].sum()
    for bar, val in zip(bars2, role_df["count"]):
        pct = val / total_roles * 100
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val}  ({pct:.0f}%)", va="center", ha="left",
                 fontsize=8.5, color=TEXT_MID)

    ax2.set_title("Role Category Distribution", fontsize=12,
                  fontweight="bold", color=TEXT_DARK, pad=10)
    ax2.set_xlabel("No. of job postings", fontsize=9, color=TEXT_MID)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.tick_params(axis="x", labelsize=8, colors=TEXT_MID)
    ax2.set_xlim(0, role_df["count"].max() * 1.35)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4, color=C_GRAY_LIGHT)
    ax2.set_axisbelow(True)
    ax2.text(0.98, 0.02, f"n = {total_roles} (excl. untagged)",
             transform=ax2.transAxes, ha="right", fontsize=8, color=TEXT_MID)
    
    # PANEL 3: TOP COMPANIES HIRING FRESHERS
    comp_df = get_fresher_companies(df, 12).sort_values("count")

    bars3 = ax3.barh(comp_df["company"], comp_df["count"],
                     color=C_AMBER_LIGHT, height=0.6, edgecolor="none")
    # Highlight top 3
    for i, bar in enumerate(bars3):
        if i >= len(comp_df) - 3:
            bar.set_color(C_AMBER)

    for bar, val in zip(bars3, comp_df["count"]):
        ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", ha="left",
                 fontsize=8.5, color=TEXT_MID)

    ax3.set_title("Top Companies Hiring Freshers", fontsize=12,
                  fontweight="bold", color=TEXT_DARK, pad=10)
    ax3.set_xlabel("Fresher / Internship postings", fontsize=9, color=TEXT_MID)
    ax3.tick_params(axis="y", labelsize=8.5)
    ax3.tick_params(axis="x", labelsize=8, colors=TEXT_MID)
    ax3.set_xlim(0, comp_df["count"].max() * 1.25)
    ax3.xaxis.grid(True, linestyle="--", alpha=0.4, color=C_GRAY_LIGHT)
    ax3.set_axisbelow(True)
    total_fresher = df[df["experience_level"].isin(
        ["Fresher (0-2 yrs)", "Internship"])].shape[0]
    ax3.text(0.98, 0.02, f"Total fresher roles: {total_fresher}",
             transform=ax3.transAxes, ha="right", fontsize=8, color=TEXT_MID)
    
    # PANEL 4: WEEKLY POSTING TREND
    weekly = get_weekly_trend(df)

    ax4.fill_between(weekly["week"], weekly["count"],
                     alpha=0.18, color=C_BLUE)
    ax4.plot(weekly["week"], weekly["count"],
             color=C_BLUE, linewidth=1.5, marker="o",
             markersize=4, label="Weekly postings")
    ax4.plot(weekly["week"], weekly["rolling"],
             color=C_CORAL, linewidth=2, linestyle="--",
             label="3-week rolling avg")

    # Annotate the peak
    peak_idx = weekly["count"].idxmax()
    peak_row = weekly.loc[peak_idx]
    ax4.annotate(
        f"Peak: {int(peak_row['count'])} jobs",
        xy=(peak_row["week"], peak_row["count"]),
        xytext=(0, 12), textcoords="offset points",
        ha="center", fontsize=8.5, color=C_CORAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.2)
    )

    ax4.set_title("Weekly Job Posting Trend (2025–2026)", fontsize=12,
                  fontweight="bold", color=TEXT_DARK, pad=10)
    ax4.set_ylabel("Job postings per week", fontsize=9, color=TEXT_MID)
    ax4.tick_params(axis="x", rotation=35, labelsize=8)
    ax4.tick_params(axis="y", labelsize=8)
    ax4.legend(fontsize=8.5, framealpha=0.8, edgecolor=C_GRAY_LIGHT)
    ax4.yaxis.grid(True, linestyle="--", alpha=0.4, color=C_GRAY_LIGHT)
    ax4.set_axisbelow(True)
    ax4.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%b %d")
    )
    plt.setp(ax4.xaxis.get_majorticklabels(), ha="right")
    
    # PANEL 5: SKILL CO-OCCURRENCE HEATMAP
    mat, top_skills = get_cooccurrence(df, 10)

    # Normalize by diagonal (self-count before we zeroed it) for % view
    # Use raw counts, displayed with imshow
    im = ax5.imshow(mat, cmap="YlOrRd", aspect="auto")

    ax5.set_xticks(range(len(top_skills)))
    ax5.set_yticks(range(len(top_skills)))
    ax5.set_xticklabels(top_skills, rotation=45, ha="right", fontsize=8)
    ax5.set_yticklabels(top_skills, fontsize=8)

    # Annotate cells with count
    for i in range(len(top_skills)):
        for j in range(len(top_skills)):
            val = mat[i, j]
            if val > 0:
                text_color = "white" if val > mat.max() * 0.55 else TEXT_DARK
                ax5.text(j, i, str(val), ha="center", va="center",
                         fontsize=7, color=text_color)

    ax5.set_title("Skill Co-occurrence Matrix", fontsize=12,
                  fontweight="bold", color=TEXT_DARK, pad=10)
    ax5.text(0.5, -0.22,
             "Cell value = jobs where both skills appear together",
             transform=ax5.transAxes, ha="center", fontsize=8, color=TEXT_MID)
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    
if __name__ == "__main__":
    print("Job Market Intel — Visualization")
    df = load_and_prep()