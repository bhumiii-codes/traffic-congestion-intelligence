import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder


def split_data(df, feature_cols, target_reg='traffic_volume', target_cls='congestion_level'):
    """Split data into train, validation and test sets (70/15/15)."""

    X = df[feature_cols].fillna(df[feature_cols].median())

    # Regression target
    y_reg = df[target_reg]

    # Classification target
    le = LabelEncoder()
    y_cls = le.fit_transform(df[target_cls].astype(str))

    # First split — 70% train, 30% temp
    X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = train_test_split(
        X, y_reg, y_cls, test_size=0.30, random_state=42, shuffle=False
    )

    # Second split — 50% of temp = 15% val, 15% test
    X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
        X_temp, y_reg_temp, y_cls_temp, test_size=0.50, random_state=42, shuffle=False
    )

    print(f"✅ Data split complete:")
    print(f"   Train : {X_train.shape[0]:,} rows")
    print(f"   Val   : {X_val.shape[0]:,} rows")
    print(f"   Test  : {X_test.shape[0]:,} rows")

    return (X_train, X_val, X_test,
            y_reg_train, y_reg_val, y_reg_test,
            y_cls_train, y_cls_val, y_cls_test,
            le)


def train_regression_models(X_train, y_train):
    """Train all regression models."""
    print("\n── Training Regression Models ────────")

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15,
            min_samples_leaf=4, n_jobs=-1, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, n_jobs=-1,
            random_state=42, verbosity=0
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"   ✅ {name} done.")

    return trained


def train_classification_models(X_train, y_train):
    """Train all classification models."""
    print("\n── Training Classification Models ────")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15,
            min_samples_leaf=4, n_jobs=-1, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, n_jobs=-1,
            random_state=42, verbosity=0,
            eval_metric='mlogloss'
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"   ✅ {name} done.")

    return trained


def cross_validate_models(models, X_train, y_train, task='regression', cv=5):
    """Run K-Fold cross validation on all models."""
    print(f"\n── {cv}-Fold Cross Validation ────────────")

    scoring = 'r2' if task == 'regression' else 'f1_weighted'
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            'mean': scores.mean(),
            'std':  scores.std(),
            'scores': scores
        }
        print(f"   {name:<25} {scoring}= {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


def save_models(models, output_dir='outputs/models'):
    """Save all trained models to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(output_dir, filename)
        joblib.dump(model, filepath)
    print(f"\n✅ All models saved to {output_dir}/")


def load_model(name, output_dir='outputs/models'):
    """Load a saved model from disk."""
    filename = name.lower().replace(' ', '_') + '.pkl'
    filepath = os.path.join(output_dir, filename)
    return joblib.load(filepath)