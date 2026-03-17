import json
import joblib
import numpy as np
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess
from src.features.engineer import engineer_features, get_feature_columns
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

df = load_raw_data('data/raw/Metro_Interstate_Traffic_Volume.csv')
df = preprocess(df)
df = engineer_features(df)

X = df[get_feature_columns()].fillna(df[get_feature_columns()].median())
y_reg = df['traffic_volume']
le = LabelEncoder()
y_cls = le.fit_transform(df['congestion_level'].astype(str))

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

cls_models = {
    'random_forest':     RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'xgboost':           XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0, eval_metric='mlogloss'),
}

metrics = {}
for name in ['random_forest', 'gradient_boosting', 'xgboost']:
    reg_model = joblib.load(f'outputs/models/{name}.pkl')
    reg_pred  = reg_model.predict(X_test)
    metrics[name] = {
        'r2':  round(r2_score(y_reg_test, reg_pred), 4),
        'mae': round(mean_absolute_error(y_reg_test, reg_pred), 1),
    }

print('Training classification models...')
for name, model in cls_models.items():
    print(f'  Training {name}...')
    model.fit(X_train, y_cls_train)
    joblib.dump(model, f'outputs/models/{name}_cls.pkl')
    pred = model.predict(X_test)
    metrics[name]['f1']        = round(f1_score(y_cls_test, pred, average='weighted'), 4)
    metrics[name]['accuracy']  = round(accuracy_score(y_cls_test, pred), 4)
    metrics[name]['precision'] = round(precision_score(y_cls_test, pred, average='weighted', zero_division=0), 4)
    metrics[name]['recall']    = round(recall_score(y_cls_test, pred, average='weighted', zero_division=0), 4)
    print(f'  done — F1={metrics[name]["f1"]}  Acc={metrics[name]["accuracy"]}')

with open('outputs/model_metrics.json', 'w') as f:
    json.dump(metrics, f)

print('\nFinal metrics:')
for k, v in metrics.items():
    print(f'  {k}: {v}')

