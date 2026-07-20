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

if __name__ == "__main__":
    print("Job Market Intel — EDA & Analysis")

    df = load_data()
    
    q1 = q1_top_skills(df)
    
    q2 = q2_fresher_companies(df)
    
    q3 = q3_role_distribution(df)
    