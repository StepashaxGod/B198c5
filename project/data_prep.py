import numpy as np
import pandas as pd


def load_wine_dataset(path: str) -> pd.DataFrame:
    "Load wine quality dataset from CSV file."
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)

    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")

    return df

# noise is not directly evaluated
def make_dirty_dataset(
    df: pd.DataFrame,
    target: str,
    missing_rate: float = 0.05,
    noise_std: float = 0.02,
    n_duplicates: int = 50,
    outlier_rate: float = 0.03,
    outlier_multiplier: float = 6.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Creates a 'dirty' version of the dataset by injecting:
    -- missing values
    -- Gaussian noise
    -- duplicated rows
    -- outliers""",
    rng = np.random.default_rng(seed)
    dirty = df.copy()

    feature_cols = [c for c in dirty.columns if c != target]

    n_cells = dirty[feature_cols].size
    n_missing = int(n_cells * missing_rate)

    rows = rng.integers(0, dirty.shape[0], n_missing)
    cols = rng.integers(0, len(feature_cols), n_missing)

    for r, c in zip(rows, cols):
        dirty.at[r, feature_cols[c]] = np.nan

    noise = rng.normal(0, noise_std, size=dirty[feature_cols].shape)
    dirty[feature_cols] = dirty[feature_cols] + noise

    if n_duplicates > 0:
        dup_idx = rng.choice(dirty.index, size=n_duplicates, replace=True)
        duplicates = dirty.loc[dup_idx]
        dirty = pd.concat([dirty, duplicates], ignore_index=True)

    n_outliers = int(len(dirty) * outlier_rate)
    outlier_rows = rng.choice(dirty.index, size=n_outliers, replace=False)
    outlier_cols = rng.choice(feature_cols, size=n_outliers, replace=True)

    for r, c in zip(outlier_rows, outlier_cols):
        dirty.at[r, c] *= outlier_multiplier

    return dirty.reset_index(drop=True)