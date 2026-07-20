import pandas as pd
import numpy as np
from pathlib import Path

#CONFIG 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "jobs_clean.csv"

# Your current skill set 
YOUR_SKILLS = [
    "python", "numpy", "pandas", "matplotlib",
    "fastapi", "git", "sql",
]

def load_data():
    df = pd.read_csv(CLEAN_PATH)
    df["posted_date"] = pd.to_datetime(df["posted_date"], utc=True, errors="coerce")
    print(f"Loaded {len(df)} rows for analysis\n")
    return df

#Q1: TOP 20 IN-DEMAND SKILLS
def q1_top_skills(df):
    # Keep only rows that have skills
    has_skills = df[df["skills_extracted"] != "Not specified"].copy()

    # Split comma-separated skills into lists, explode into rows
    skill_series = (
        has_skills["skills_extracted"]
        .str.split(",")
        .explode()
        .str.strip()
        .str.lower()
    )
    
    # Count and sort
    skill_counts = skill_series.value_counts()

    # Total jobs with skills (denominator for % calculation)
    total = len(has_skills)

    # Build result DataFrame
    result = pd.DataFrame({
        "skill":   skill_counts.index,
        "count":   skill_counts.values,
        "pct_jobs": np.round(skill_counts.values / total * 100, 1),
    }).head(20)
    
    print("── Q1: Top 20 Skills ──")
    print(result.to_string(index=False))
    return result

# Q2: TOP COMPANIES HIRING FRESHERS 
def q2_fresher_companies(df):
    fresher_mask = df["experience_level"].isin(["Fresher (0-2 yrs)", "Internship"])
    freshers = df[fresher_mask].copy()

    company_counts = (
        freshers["company"]
        .value_counts()
        .reset_index()
    )
    company_counts.columns = ["company", "job_count"]

    # Add company size tag
    company_counts = company_counts.merge(
        df[["company", "company_size"]].drop_duplicates("company"),
        on="company", how="left"
    )

    result = company_counts.head(15)

    print("\n── Q2: Top Companies Hiring Freshers ──")
    print(result.to_string(index=False))
    return result

# Q3: ROLE CATEGORY DISTRIBUTION 
def q3_role_distribution(df):
    role_counts = (
        df[df["role_category"] != "Other"]["role_category"]
        .value_counts()
        .reset_index()
    )
    role_counts.columns = ["role", "count"]
    role_counts["pct"] = np.round(
        role_counts["count"] / role_counts["count"].sum() * 100, 1
    )

    print("\n── Q3: Role Distribution ──")
    print(role_counts.to_string(index=False))
    return role_counts

# Q4: WEEKLY POSTING TREND
def q4_weekly_trend(df):
    dated = df[df["posted_date"].notna()].copy()

    # Filter to recent data only (2025 onward)
    dated = dated[dated["posted_date"].dt.year >= 2025]

    dated["week"] = dated["posted_date"].dt.to_period("W").dt.start_time

    weekly = (
        dated.groupby("week")
        .size()
        .reset_index(name="job_count")
        .sort_values("week")
    )

    # 3-week rolling average (NumPy)
    weekly["rolling_avg"] = np.convolve(
        weekly["job_count"].values,
        np.ones(3) / 3,
        mode="same"
    )

    print(f"\n── Q4: Weekly Trend ({len(weekly)} weeks) ──")
    print(weekly.tail(8).to_string(index=False))
    return weekly

# Q5: SKILL CO-OCCURRENCE MATRIX 
def q5_skill_cooccurrence(df, top_n=12):
    has_skills = df[df["skills_extracted"] != "Not specified"].copy()
    has_skills["skills_list"] = (
        has_skills["skills_extracted"]
        .str.split(",")
        .apply(lambda lst: [s.strip().lower() for s in lst])
    )

    # Get top N skills by frequency
    all_skills = has_skills["skills_list"].explode()
    top_skills = all_skills.value_counts().head(top_n).index.tolist()

    # Build co-occurrence matrix using NumPy
    n = len(top_skills)
    skill_idx = {s: i for i, s in enumerate(top_skills)}
    matrix = np.zeros((n, n), dtype=int)

    for skills in has_skills["skills_list"]:
        # Only top skills present in this posting
        present = [s for s in skills if s in skill_idx]
        for i, s1 in enumerate(present):
            for s2 in present[i:]:   # upper triangle only
                r, c = skill_idx[s1], skill_idx[s2]
                matrix[r][c] += 1
                if r != c:
                    matrix[c][r] += 1  # mirror

    cooc_df = pd.DataFrame(matrix, index=top_skills, columns=top_skills)

    print(f"\n── Q5: Co-occurrence Matrix ({top_n}×{top_n}) ──")
    print(cooc_df.to_string())
    return cooc_df, top_skills

# Q6: YOUR SKILL COVERAGE VS MARKET
def q6_skill_gap(df, top_n=15):

    has_skills = df[df["skills_extracted"] != "Not specified"].copy()
    skill_series = (
        has_skills["skills_extracted"]
        .str.split(",")
        .explode()
        .str.strip()
        .str.lower()
    )
    top_skills = skill_series.value_counts().head(top_n)

    your_lower = [s.lower() for s in YOUR_SKILLS]

    result = pd.DataFrame({
        "skill":        top_skills.index,
        "market_count": top_skills.values,
        "you_have":     [s in your_lower for s in top_skills.index],
    })

    coverage = result["you_have"].sum()
    print(f"\n── Q6: Skill Gap Analysis ──")
    print(f"  You cover {coverage}/{top_n} top market skills")
    print(result.to_string(index=False))
    return result

# SUMMARY STATS 
def summary_stats(df):
    print("SUMMARY STATS")
    print(f"Total job postings analysed : {len(df)}")
    print(f"Unique companies : {df['company'].nunique()}")
    print(f"Date range : {df['posted_date'].min()} → {df['posted_date'].max()}")
    print(f"Fresher-friendly roles : {(df['experience_level']=='Fresher (0-2 yrs)').sum()}")
    print(f"Internships : {(df['experience_level']=='Internship').sum()}")
    print(f"Remote/WFH in titles : {df['job_title'].str.contains('remote|wfh|work from home', case=False, na=False).sum()}")
    print(f"Salary disclosed : {(df['salary_label']!='Not disclosed').sum()}")
    print(f"Most common role : {df['role_category'].value_counts().idxmax()}")
    print(f"Top hiring company : {df['company'].value_counts().idxmax()}")

if __name__ == "__main__":
    print("Job Market Intel — EDA & Analysis")

    df = load_data()
    
    q1 = q1_top_skills(df)
    
    q2 = q2_fresher_companies(df)
    
    q3 = q3_role_distribution(df)
    
    q4 = q4_weekly_trend(df)
    
    q5 = q5_skill_cooccurrence(df)
    
    q6 = q6_skill_gap(df)
    
    summary_stats(df)
    
    