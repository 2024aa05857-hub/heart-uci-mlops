from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLS = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal","target"
]

def download_raw(out_path: str = "data/raw/heart.csv") -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(UCI_URL, header=None, names=COLS)
    df.to_csv(out, index=False)
    return str(out)

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace("?", pd.NA)

    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["target"] = (df["target"] > 0).astype(int)

    num_cols = [c for c in COLS if c != "target"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
    return df

def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
