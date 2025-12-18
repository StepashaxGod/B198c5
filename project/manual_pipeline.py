import pandas as pd


def clean_manual_fixed(df: pd.DataFrame, target: str = "quality") -> pd.DataFrame:
    """
    Manual fixed cleaning:
    1) mean imputation (features only)
    2) drop duplicates (rows)
    3) Z-score outlier removal (|z| < 3 across all features)
    """
    features = df.drop(columns=[target])
    filled = features.fillna(features.mean(numeric_only=True))

    tmp = pd.concat([filled, df[target]], axis=1).drop_duplicates().reset_index(drop=True)

    X = tmp.drop(columns=[target])
    z = (X - X.mean()) / X.std(ddof=0)
    mask = (z.abs() < 3).all(axis=1)

    return tmp.loc[mask].reset_index(drop=True)

