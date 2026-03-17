import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the Metro Interstate Traffic Volume dataset.
    Validates file existence and basic structure.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    print(f"✅ Data loaded successfully!")
    print(f"   Shape      : {df.shape}")
    print(f"   Columns    : {list(df.columns)}")
    print(f"   Date range : {df['date_time'].min()} → {df['date_time'].max()}")
    print(f"   Missing    : {df.isnull().sum().sum()} total missing values")

    return df


def get_basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset statistics."""
    print("\n── Dataset Info ──────────────────────")
    print(df.dtypes)
    print("\n── First 5 Rows ──────────────────────")
    print(df.head())
    print("\n── Basic Statistics ──────────────────")
    print(df.describe())