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

if __name__ == "__main__":
    print("Job Market Intel — Data Cleaning Pipeline")

    df = pd.read_csv(RAW_PATH)

    df = load_and_clean(df)
    
    df = fix_dates_format(df)
    
    df = extract_from_description(df)