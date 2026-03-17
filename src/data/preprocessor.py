import pandas as pd
import numpy as np


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date_time column to datetime and extract components."""
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['hour']        = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek   # 0=Mon, 6=Sun
    df['month']       = df['date_time'].dt.month
    df['year']        = df['date_time'].dt.year
    df['day_of_year'] = df['date_time'].dt.dayofyear
    df['quarter']     = df['date_time'].dt.quarter

    print("✅ Datetime parsed and components extracted.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    df = df.copy()

    missing_before = df.isnull().sum().sum()

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    missing_after = df.isnull().sum().sum()
    print(f"✅ Missing values handled: {missing_before} → {missing_after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"✅ Duplicates removed: {before - after} rows dropped.")
    return df


def remove_outliers(df: pd.DataFrame, col: str = 'traffic_volume') -> pd.DataFrame:
    """Remove outliers using IQR method."""
    df = df.copy()
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    before = len(df)
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    after = len(df)
    print(f"✅ Outliers removed from '{col}': {before - after} rows dropped.")
    return df


def encode_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Encode weather_main and holiday columns."""
    df = df.copy()

    # Encode weather_main
    df['weather_encoded'] = pd.Categorical(df['weather_main']).codes

    # Encode holiday — 'None' means no holiday
    df['is_holiday'] = (df['holiday'] != 'None').astype(int)

    print("✅ Weather and holiday columns encoded.")
    return df


def add_congestion_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add congestion level classification label.
    Bins based on traffic volume distribution.
    """
    df = df.copy()
    bins   = [0, 1000, 3000, 5000, 8000]
    labels = ['Low', 'Moderate', 'High', 'Critical']
    df['congestion_level'] = pd.cut(
        df['traffic_volume'], bins=bins, labels=labels, include_lowest=True
    )
    dist = df['congestion_level'].value_counts(normalize=True) * 100
    print("✅ Congestion labels added:")
    for lvl, pct in dist.items():
        print(f"   {lvl:<10} {pct:.1f}%")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing pipeline."""
    print("\n── Running Preprocessing Pipeline ───")
    df = parse_datetime(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = encode_weather(df)
    df = add_congestion_label(df)
    print(f"\n✅ Preprocessing complete. Final shape: {df.shape}")
    return df