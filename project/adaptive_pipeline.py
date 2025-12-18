import pandas as pd

def clean_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean(numeric_only=True)).drop_duplicates().reset_index(drop=True)

def clean_median(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True)).drop_duplicates().reset_index(drop=True)

def clean_dropna(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().drop_duplicates().reset_index(drop=True)

def clean_mean_zscore(df: pd.DataFrame, target: str = "quality") -> pd.DataFrame:
    tmp = clean_mean(df)
    X = tmp.drop(columns=[target])
    z = (X - X.mean()) / X.std(ddof=0)
    mask = (z.abs() < 3).all(axis=1)
    return tmp.loc[mask].reset_index(drop=True)

def clean_median_zscore(df: pd.DataFrame, target: str = "quality") -> pd.DataFrame:
    tmp = clean_median(df)
    X = tmp.drop(columns=[target])
    z = (X - X.mean()) / X.std(ddof=0)
    mask = (z.abs() < 3).all(axis=1)
    return tmp.loc[mask].reset_index(drop=True)

def get_strategies(target: str = "quality"):
    return {
        "mean+dedup": clean_mean,
        "median+dedup": clean_median,
        "dropna+dedup": clean_dropna,
        "mean+dedup+zscore": lambda d: clean_mean_zscore(d, target=target),
        "median+dedup+zscore": lambda d: clean_median_zscore(d, target=target),
    }

