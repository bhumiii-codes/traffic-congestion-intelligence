"""
Urban Traffic Congestion Intelligence System
============================================
Main pipeline — runs end to end:
  1. Load data
  2. Preprocess
  3. Feature engineering
  4. Train models
  5. Evaluate (regression + classification)
  6. Visualize
"""

import os
import warnings
warnings.filterwarnings('ignore')

from src.data.loader       import load_raw_data, get_basic_info
from src.data.preprocessor import preprocess
from src.features.engineer import engineer_features, get_feature_columns
from src.models.trainer    import (split_data, train_regression_models,
                                   train_classification_models,
                                   cross_validate_models, save_models)
from src.models.evaluator  import (evaluate_regression, evaluate_classification,
                                   plot_regression_comparison,
                                   plot_classification_comparison,
                                   plot_confusion_matrix,
                                   plot_predictions_vs_actual,
                                   plot_feature_importance,
                                   plot_learning_curves)
from src.visualization.plots import run_all_plots

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = 'data/raw/Metro_Interstate_Traffic_Volume.csv'

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 1 — LOADING DATA")
print("="*55)
df = load_raw_data(DATA_PATH)
get_basic_info(df)

# ─────────────────────────────────────────────
# STEP 2 — PREPROCESS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 2 — PREPROCESSING")
print("="*55)
df = preprocess(df)

# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 3 — FEATURE ENGINEERING")
print("="*55)
df = engineer_features(df)
FEATURES = get_feature_columns()

# ─────────────────────────────────────────────
# STEP 4 — SPLIT DATA
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 4 — SPLITTING DATA (70/15/15)")
print("="*55)
(X_train, X_val, X_test,
 y_reg_train, y_reg_val, y_reg_test,
 y_cls_train, y_cls_val, y_cls_test,
 label_encoder) = split_data(df, FEATURES)

# ─────────────────────────────────────────────
# STEP 5 — TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 5 — TRAINING MODELS")
print("="*55)
reg_models = train_regression_models(X_train, y_reg_train)
cls_models = train_classification_models(X_train, y_cls_train)

# ─────────────────────────────────────────────
# STEP 6 — CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 6 — CROSS VALIDATION")
print("="*55)
reg_cv = cross_validate_models(reg_models, X_train, y_reg_train, task='regression')
cls_cv = cross_validate_models(cls_models, X_train, y_cls_train, task='classification')

# ─────────────────────────────────────────────
# STEP 7 — EVALUATE
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 7 — EVALUATION")
print("="*55)
reg_metrics = evaluate_regression(reg_models, X_test, y_reg_test)
cls_metrics = evaluate_classification(cls_models, X_test, y_cls_test, label_encoder)

print("\n── Regression Metrics Table ──────────────")
print(reg_metrics.to_string(index=False))

print("\n── Classification Metrics Table ──────────")
print(cls_metrics.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 8 — EVALUATION PLOTS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 8 — EVALUATION PLOTS")
print("="*55)
plot_regression_comparison(reg_metrics)
plot_classification_comparison(cls_metrics)
plot_confusion_matrix(cls_models, X_test, y_cls_test, label_encoder)
plot_predictions_vs_actual(reg_models, X_test, y_reg_test)
plot_feature_importance(reg_models, FEATURES)
plot_learning_curves(reg_models, X_train, y_reg_train, task='regression')

# ─────────────────────────────────────────────
# STEP 9 — EDA VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 9 — EDA VISUALIZATIONS")
print("="*55)
run_all_plots(df)

# ─────────────────────────────────────────────
# STEP 10 — SAVE MODELS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 10 — SAVING MODELS")
print("="*55)
save_models(reg_models, 'outputs/models')

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL SUMMARY")
print("="*55)
best_reg = reg_metrics.loc[reg_metrics['R²'].idxmax(), 'Model']
best_cls = cls_metrics.loc[cls_metrics['F1 Score'].idxmax(), 'Model']

print(f"\n  Best Regression Model  : {best_reg}")
print(f"  Best R²                : {reg_metrics['R²'].max():.4f}")
print(f"\n  Best Classification    : {best_cls}")
print(f"  Best F1 Score          : {cls_metrics['F1 Score'].max():.4f}")
print(f"\n  All figures saved to   : outputs/figures/")
print(f"  All models saved to    : outputs/models/")
print("\n" + "="*55)
print("  ✅ Pipeline complete!")
print("="*55)