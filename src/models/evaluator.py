import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


OUTPUT_DIR = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    'bg':     '#0D1117',
    'panel':  '#161B22',
    'border': '#30363D',
    'text':   '#E6EDF3',
    'sub':    '#8B949E',
    'green':  '#3FB950',
    'blue':   '#58A6FF',
    'orange': '#F78166',
    'yellow': '#E3B341',
    'purple': '#BC8CFF',
    'red':    '#FF6E6E',
}


def evaluate_regression(models, X_test, y_test):
    """Evaluate all regression models and return metrics dataframe."""
    print("\n── Regression Evaluation ─────────────")
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        mape   = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

        results.append({
            'Model': name,
            'MAE':   round(mae, 2),
            'RMSE':  round(rmse, 2),
            'R²':    round(r2, 4),
            'MAPE%': round(mape, 2)
        })

        print(f"   {name:<25} MAE={mae:.0f}  RMSE={rmse:.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    return pd.DataFrame(results)


def evaluate_classification(models, X_test, y_test, label_encoder):
    """Evaluate all classification models and return metrics dataframe."""
    print("\n── Classification Evaluation ─────────")
    results = []
    class_names = label_encoder.classes_

    for name, model in models.items():
        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model':      name,
            'Accuracy':   round(acc, 4),
            'F1 Score':   round(f1, 4),
            'Precision':  round(precision, 4),
            'Recall':     round(recall, 4),
        })

        print(f"   {name:<25} Acc={acc:.4f}  F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

        # Full classification report
        print(f"\n   Classification Report — {name}:")
        print(classification_report(y_test, y_pred, target_names=class_names))

    return pd.DataFrame(results)


def plot_regression_comparison(reg_metrics, save=True):
    """Bar chart comparing regression models across metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Regression Model Comparison', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    metrics = ['MAE', 'RMSE', 'R²', 'MAPE%']
    colors  = [PALETTE['blue'], PALETTE['purple'], PALETTE['green'], PALETTE['orange']]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.set_facecolor(PALETTE['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE['border'])
        bars = ax.bar(reg_metrics['Model'], reg_metrics[metric],
                      color=color, edgecolor=PALETTE['border'], width=0.5)
        ax.set_title(metric, color=PALETTE['text'], fontweight='bold')
        ax.tick_params(colors=PALETTE['sub'])
        ax.set_xticklabels(reg_metrics['Model'], rotation=20, ha='right',
                           fontsize=8, color=PALETTE['sub'])
        for bar, val in zip(bars, reg_metrics[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    str(val), ha='center', va='bottom',
                    fontsize=8, color=PALETTE['text'])

    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/regression_comparison.png',
                    dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved: regression_comparison.png")
    plt.close()


def plot_classification_comparison(cls_metrics, save=True):
    """Bar chart comparing classification models across metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Classification Model Comparison', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    colors  = [PALETTE['green'], PALETTE['blue'], PALETTE['purple'], PALETTE['orange']]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.set_facecolor(PALETTE['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE['border'])
        bars = ax.bar(cls_metrics['Model'], cls_metrics[metric],
                      color=color, edgecolor=PALETTE['border'], width=0.5)
        ax.set_title(metric, color=PALETTE['text'], fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors=PALETTE['sub'])
        ax.set_xticklabels(cls_metrics['Model'], rotation=20, ha='right',
                           fontsize=8, color=PALETTE['sub'])
        for bar, val in zip(bars, cls_metrics[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    str(val), ha='center', va='bottom',
                    fontsize=8, color=PALETTE['text'])

    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/classification_comparison.png',
                    dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved: classification_comparison.png")
    plt.close()


def plot_confusion_matrix(models, X_test, y_test, label_encoder, save=True):
    """Plot confusion matrix for all classification models."""
    class_names = label_encoder.classes_
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Confusion Matrices — All Models', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues', cbar=True,
                    annot_kws={'size': 10, 'color': 'white'})
        ax.set_facecolor(PALETTE['panel'])
        ax.set_title(name, color=PALETTE['text'], fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', color=PALETTE['sub'])
        ax.set_ylabel('Actual', color=PALETTE['sub'])
        ax.tick_params(colors=PALETTE['sub'])

    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png',
                    dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved: confusion_matrices.png")
    plt.close()


def plot_predictions_vs_actual(models, X_test, y_test, save=True):
    """Plot actual vs predicted for regression models."""
    n   = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Actual vs Predicted — Regression Models', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    if n == 1:
        axes = [axes]

    sample = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        ax.set_facecolor(PALETTE['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE['border'])
        ax.scatter(y_test.values[sample], y_pred[sample],
                   alpha=0.3, s=8, color=PALETTE['blue'], rasterized=True)
        lim = [0, max(y_test.max(), y_pred.max())]
        ax.plot(lim, lim, color=PALETTE['orange'], lw=2, ls='--', label='Perfect fit')
        r2 = r2_score(y_test, y_pred)
        ax.set_title(f'{name}\nR²={r2:.4f}', color=PALETTE['text'], fontweight='bold')
        ax.set_xlabel('Actual Volume', color=PALETTE['sub'])
        ax.set_ylabel('Predicted Volume', color=PALETTE['sub'])
        ax.tick_params(colors=PALETTE['sub'])
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/predictions_vs_actual.png',
                    dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved: predictions_vs_actual.png")
    plt.close()


def plot_feature_importance(models, feature_cols, top_n=15, save=True):
    """Plot feature importance for tree-based models."""
    tree_models = {k: v for k, v in models.items()
                   if hasattr(v, 'feature_importances_')}

    if not tree_models:
        print("No tree-based models found for feature importance.")
        return

    n   = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 8), facecolor=PALETTE['bg'])
    fig.suptitle('Feature Importance — Tree Models', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        fi = pd.Series(model.feature_importances_, index=feature_cols)
        fi = fi.nlargest(top_n).sort_values()

        ax.set_facecolor(PALETTE['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE['border'])
        colors = [PALETTE['blue'] if i >= len(fi)-3 else
                  PALETTE['purple'] if i >= len(fi)-8 else
                  PALETTE['teal'] if 'teal' in PALETTE else PALETTE['green']
                  for i in range(len(fi))]
        ax.barh(fi.index, fi.values, color=colors, edgecolor=PALETTE['border'])
        ax.set_title(name, color=PALETTE['text'], fontweight='bold')
        ax.set_xlabel('Importance Score', color=PALETTE['sub'])
        ax.tick_params(colors=PALETTE['sub'], labelsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/feature_importance.png',
                    dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved: feature_importance.png")
    plt.close()


def plot_learning_curves(models, X_train, y_train, task='regression', save=True):
    """Plot learning curves to check for overfitting."""
    from sklearn.model_selection import learning_curve

    scoring = 'r2' if task == 'regression' else 'f1_weighted'
    n       = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), facecolor=PALETTE['bg'])
    fig.suptitle(f'Learning Curves ({scoring})', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    if n == 1:
        axes = [axes]

    train_sizes = np.linspace(0.1, 1.0, 8)

    for ax, (name, model) in zip(axes, models.items()):
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes, cv=3,
            scoring=scoring, n_jobs=-1
        )
        ax.set_facecolor(PALETTE['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE['border'])

        ax.plot(sizes, train_scores.mean(axis=1),
                color=PALETTE['blue'], lw=2, marker='o', label='Train')
        ax.fill_between(sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.15, color=PALETTE['blue'])
        ax.plot(sizes, val_scores.mean(axis=1),
                color=PALETTE['orange'], lw=2, marker='s', label='Validation')
        ax.fill_between(sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.15, color=PALETTE['orange'])

        ax.set_title(name, color=PALETTE['text'], fontweight='bold')
        ax.set_xlabel('Training Size', color=PALETTE['sub'])
        ax.set_ylabel(scoring, color=PALETTE['sub'])
        ax.tick_params(colors=PALETTE['sub'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/learning_curves.png',
                    dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved: learning_curves.png")
    plt.close()