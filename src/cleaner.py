import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path

# Path Configuration 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_raw.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "jobs_clean.csv"

# Skills we care about for Hyderabad AI/ML market
SKILL_KEYWORDS = [
    "python", "sql", "numpy", "pandas", "matplotlib", "scikit-learn", "sklearn",
    "tensorflow", "keras", "pytorch", "opencv", "nlp", "spacy", "nltk",
    "transformers", "huggingface", "langchain", "langgraph", "llamaindex",
    "rag", "vector database", "pinecone", "weaviate", "chromadb", "faiss",
    "llm", "openai", "gemini", "claude", "mistral", "ollama",
    "mlflow", "dvc", "airflow", "fastapi", "flask", "docker", "kubernetes",
    "aws", "azure", "gcp", "spark", "hadoop", "pyspark",
    "git", "linux", "rest api", "mongodb", "postgresql", "mysql",
    "agentic ai", "multi-agent", "prompt engineering", "fine-tuning",
    "computer vision", "object detection", "yolo", "cnn", "rnn", "lstm",
    "bert", "gpt", "stable diffusion", "generative ai", "genai",
    "power bi", "tableau", "excel", "r", "java", "c++",
]

# Role category mapping
ROLE_MAP = {
    "ml engineer":    "ML Engineer",
    "machine learning": "ML Engineer",
    "ai engineer":    "AI Engineer",
    "ai programmer":  "AI Engineer",
    "data scientist": "Data Scientist",
    "data analyst":   "Data Analyst",
    "genai":          "GenAI/LLM Engineer",
    "llm":            "GenAI/LLM Engineer",
    "generative ai":  "GenAI/LLM Engineer",
    "nlp":            "NLP Engineer",
    "computer vision":"Computer Vision Engineer",
    "deep learning":  "ML Engineer",
    "data engineer":  "Data Engineer",
}

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

def fix_dates_format(df):
    def parse_date(val):
        if pd.isna(val) or str(val).strip() == "":
            return pd.NaT
        val = str(val).strip()
        try:
            return pd.to_datetime(val, utc=True).normalize()
        except Exception:
            return pd.NaT

    df["posted_date"] = df["posted_date_raw"].apply(parse_date)
    df["posted_date"] = pd.to_datetime(df["posted_date"], utc=True)

    # Days since posted (from scrape date)
    df["scraped_at_dt"] = pd.to_datetime(df["scraped_at"], utc=True, errors="coerce")
    df["days_since_posted"] = (
        df["scraped_at_dt"] - df["posted_date"]
    ).dt.days

    # Month label for trend analysis
    df["posted_month"] = df["posted_date"].dt.to_period("M").astype(str)
    df["posted_month"] = df["posted_month"].fillna("Unknown")

    null_dates = df["posted_date"].isna().sum()
    print(f"  {null_dates} rows with no date → marked as NaT")
    print(f"  Date range: {df['posted_date'].min()} → {df['posted_date'].max()}")

    return df

def extract_from_description(df):
    #SKILLS EXTRACTION
    def extract_skills(row):
        text_sources = [
            str(row.get("description_snippet", "") or ""),
            str(row.get("job_title", "") or ""),
            str(row.get("search_keyword", "") or ""),
            str(row.get("job_url", "") or ""),
        ]
        combined = " ".join(text_sources).lower()

        found = []
        for skill in SKILL_KEYWORDS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, combined):
                found.append(skill)
        return ", ".join(found) if found else "Not specified"
    df["skills_extracted"] = df.apply(extract_skills, axis=1)
    
    #EXPERIENCE EXTRACTION
    def extract_experience(row):
        text_sources = [
            str(row.get("description_snippet", "") or ""),
            str(row.get("job_title", "") or ""),
            str(row.get("search_keyword", "") or ""),
        ]
        combined = " ".join(text_sources).lower()

        # Fresher / entry-level signals
        if any(x in combined for x in ["fresher", "0-1 year", "0 - 1 year",
                                         "entry level", "entry-level", "0-2 year",
                                         "recent graduate", "junior"]):
            return "Fresher (0-2 yrs)"

        # Mid-level signals
        if any(x in combined for x in ["2-4 year", "2 - 4 year", "3-5 year",
                                         "2+ year", "3+ year"]):
            return "Mid (2-5 yrs)"

        # Senior signals
        if any(x in combined for x in ["5+ year", "5-8 year", "senior",
                                         "lead", "principal"]):
            return "Senior (5+ yrs)"

        # Internship
        if any(x in combined for x in ["intern", "internship", "stipend",
                                         "trainee"]):
            return "Internship"

        return "Not specified"
    df["experience_extracted"] = df.apply(extract_experience, axis=1)

    #SALARY EXTRACTION
    def extract_salary(val, description):
        # First try the salary column
        for text in [str(val or ""), str(description or "")]:
            text_lower = text.lower()
            # Match patterns like "4-7 LPA", "6 LPA", "8-12 LPA"
            match = re.search(r'(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*lpa', text_lower)
            if match:
                low, high = float(match.group(1)), float(match.group(2))
                return f"{low}-{high} LPA", round((low + high) / 2, 1)
            # Single value like "6 LPA"
            match = re.search(r'(\d+(?:\.\d+)?)\s*lpa', text_lower)
            if match:
                val_num = float(match.group(1))
                return f"{val_num} LPA", val_num
            # Per month stipend
            match = re.search(r'(\d+(?:,\d+)?)\s*(?:per month|/month|pm)', text_lower)
            if match:
                monthly = float(match.group(1).replace(",", ""))
                annual = round(monthly * 12 / 100000, 1)
                return f"~{annual} LPA (stipend)", annual
        return "Not disclosed", np.nan

    df[["salary_label", "salary_mid_lpa"]] = df.apply(
        lambda r: pd.Series(extract_salary(r["salary"], r["description_snippet"])),
        axis=1
    )

    skills_found = (df["skills_extracted"] != "Not specified").sum()
    exp_found    = (df["experience_extracted"] != "Not specified").sum()
    sal_found    = (df["salary_label"] != "Not disclosed").sum()
    print(f"Skills extracted: {skills_found}/{len(df)} rows")
    print(f"Experience extracted: {exp_found}/{len(df)} rows")
    print(f"Salary found: {sal_found}/{len(df)} rows")

    return df

def standardize(df):
    #EMPLOYMENT TYPE
    def clean_employment(val):
        if pd.isna(val):
            return "Not specified"
        val = str(val).lower()
        val = val.replace("–", "-").replace("—", "-")  # fix em-dashes
        if "full" in val:
            return "Full-time"
        if "intern" in val or "trainee" in val:
            return "Internship"
        if "part" in val:
            return "Part-time"
        if "contract" in val or "freelance" in val:
            return "Contract"
        return "Not specified"

    df["employment_type_clean"] = df["employment_type"].apply(clean_employment)
    
    #ROLE CATEGORY 
    def categorize_role(title):
        if pd.isna(title):
            return "Other"
        title_lower = str(title).lower()
        for keyword, category in ROLE_MAP.items():
            if keyword in title_lower:
                return category
        return "Other"

    df["role_category"] = df["job_title"].apply(categorize_role)
    
    #COMPANY SIZE (heuristic from known names)
    MNC_NAMES = ["tcs", "infosys", "wipro", "accenture", "ibm", "microsoft",
                 "google", "amazon", "cognizant", "capgemini", "dxc", "hcl",
                 "tech mahindra", "mphasis", "oracle", "sap"]
    STARTUP_SIGNALS = ["kore.ai", "yellow.ai", "caw", "tessell", "nisum",
                       "talent500", "facilio", "unbxd", "darwinbox"]

    def company_size(company):
        if pd.isna(company):
            return "Unknown"
        c = str(company).lower()
        if any(m in c for m in MNC_NAMES):
            return "MNC / Large"
        if any(s in c for s in STARTUP_SIGNALS):
            return "Startup"
        return "Mid-size / Unknown"

    df["company_size"] = df["company"].apply(company_size)

    #FILL REMAINING NULLS
    df["company"]  = df["company"].fillna("Unknown Company")
    df["location"] = df["location"].fillna("Hyderabad (assumed)")

    # Category counts
    print("  Role categories:\n",
          df["role_category"].value_counts().to_string())
    print("  Employment types:\n",
          df["employment_type_clean"].value_counts().to_string())

    return df

if __name__ == "__main__":
    print("Job Market Intel — Data Cleaning Pipeline")

    df = pd.read_csv(RAW_PATH)

    df = load_and_clean(df)
    
    df = fix_dates_format(df)
    
    df = extract_from_description(df)