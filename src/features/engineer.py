import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encoding and time-based flags."""
    df = df.copy()

    # Cyclical encoding — preserves continuity (e.g. 23:00 → 00:00)
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']    = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']    = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Time-based flags
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_am']  = ((df['hour'] >= 7)  & (df['hour'] <= 9)  & (df['is_weekend'] == 0)).astype(int)
    df['is_rush_pm']  = ((df['hour'] >= 16) & (df['hour'] <= 18) & (df['is_weekend'] == 0)).astype(int)
    df['is_night']    = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_morning']  = ((df['hour'] >= 6)  & (df['hour'] <= 11)).astype(int)
    df['is_midday']   = ((df['hour'] >= 12) & (df['hour'] <= 15)).astype(int)

    # Season
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3:  'Spring', 4: 'Spring', 5: 'Spring',
        6:  'Summer', 7: 'Summer', 8: 'Summer',
        9:  'Autumn', 10:'Autumn', 11:'Autumn'
    })
    df['season_encoded'] = pd.Categorical(df['season']).codes

    print("✅ Time features added.")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features."""
    df = df.copy()

    df['temp_c']       = df['temp'] - 273.15   # Kelvin → Celsius
    df['is_raining']   = (df['rain_1h'] > 0).astype(int)
    df['is_snowing']   = (df['snow_1h'] > 0).astype(int)
    df['heavy_rain']   = (df['rain_1h'] > 10).astype(int)
    df['heavy_snow']   = (df['snow_1h'] > 5).astype(int)
    df['log_rain']     = np.log1p(df['rain_1h'])
    df['log_snow']     = np.log1p(df['snow_1h'])
    df['bad_weather']  = df['weather_main'].isin(
        ['Snow', 'Thunderstorm', 'Fog', 'Squall']
    ).astype(int)

    # Comfort index — high temp + no rain = more drivers
    df['comfort_index'] = (
        (df['temp_c'].between(10, 25)).astype(int) &
        (df['is_raining'] == 0) &
        (df['is_snowing'] == 0)
    ).astype(int)

    print("✅ Weather features added.")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features — previous hour and day traffic."""
    df = df.copy()
    df = df.sort_values('date_time').reset_index(drop=True)

    df['volume_lag_1h']  = df['traffic_volume'].shift(1)   # 1 hour ago
    df['volume_lag_2h']  = df['traffic_volume'].shift(2)   # 2 hours ago
    df['volume_lag_24h'] = df['traffic_volume'].shift(24)  # same hour yesterday
    df['volume_lag_168h']= df['traffic_volume'].shift(168) # same hour last week

    # Rolling averages
    df['rolling_mean_3h']  = df['traffic_volume'].shift(1).rolling(3).mean()
    df['rolling_mean_24h'] = df['traffic_volume'].shift(1).rolling(24).mean()

    # Drop rows with NaN from lag features
    df = df.dropna(subset=['volume_lag_1h', 'volume_lag_24h'])

    print(f"✅ Lag features added. Shape after lag: {df.shape}")
    return df


def get_feature_columns() -> list:
    """Return the final list of features used for modeling."""
    return [
        # Time features
        'hour', 'day_of_week', 'month', 'quarter',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'dow_sin', 'dow_cos',
        'is_weekend', 'is_holiday', 'is_rush_am', 'is_rush_pm',
        'is_night', 'is_morning', 'is_midday', 'season_encoded',

        # Weather features
        'temp_c', 'rain_1h', 'snow_1h', 'clouds_all',
        'is_raining', 'is_snowing', 'heavy_rain', 'heavy_snow',
        'log_rain', 'log_snow', 'bad_weather', 'comfort_index',
        'weather_encoded',

        # Lag features
        'volume_lag_1h', 'volume_lag_2h', 'volume_lag_24h', 'volume_lag_168h',
        'rolling_mean_3h', 'rolling_mean_24h',
    ]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    print("\n── Running Feature Engineering ───────")
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_lag_features(df)
    print(f"\n✅ Feature engineering complete. Total features: {len(get_feature_columns())}")
    return df