import json
import joblib
import numpy as np
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess
from src.features.engineer import engineer_features, get_feature_columns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

df = load_raw_data('data/raw/Metro_Interstate_Traffic_Volume.csv')
df = preprocess(df)
df = engineer_features(df)

X = df[get_feature_columns()].fillna(df[get_feature_columns()].median())
y = df['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_models = {
    'linear_regression': LinearRegression(),
    'random_forest':     RandomForestRegressor(
                            n_estimators=100, max_depth=15,
                            min_samples_leaf=4, n_jobs=-1, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(
                            n_estimators=100, max_depth=5,
                            learning_rate=0.1, random_state=42),
    'xgboost':           XGBRegressor(
                            n_estimators=100, max_depth=6,
                            learning_rate=0.1, n_jobs=-1,
                            random_state=42, verbosity=0),
}

metrics = {}
print('Training regression models...')
for name, model in reg_models.items():
    print(f'  Training {name}...')
    model.fit(X_train, y_train)
    joblib.dump(model, f'outputs/models/{name}.pkl')

    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    mape = np.mean(np.abs((y_test - pred) / (y_test + 1))) * 100

    # Cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3,
                                scoring='r2', n_jobs=-1)

    metrics[name] = {
        'mae':     round(mae, 1),
        'rmse':    round(rmse, 1),
        'r2':      round(r2, 4),
        'mape':    round(mape, 2),
        'cv_mean': round(cv_scores.mean(), 4),
        'cv_std':  round(cv_scores.std(), 4),
    }

    print(f'  done — MAE={metrics[name]["mae"]}  RMSE={metrics[name]["rmse"]}  R²={metrics[name]["r2"]}  CV={metrics[name]["cv_mean"]}±{metrics[name]["cv_std"]}')

with open('outputs/reg_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print('\n── Final Regression Metrics ──────────────')
for k, v in metrics.items():
    print(f'  {k}:')
    print(f'    MAE  : {v["mae"]}')
    print(f'    RMSE : {v["rmse"]}')
    print(f'    R²   : {v["r2"]}')
    print(f'    MAPE : {v["mape"]}%')
    print(f'    CV   : {v["cv_mean"]} ± {v["cv_std"]}')
print('\n✅ All regression models saved to outputs/models/')
print('✅ Metrics saved to outputs/reg_model_metrics.json')
