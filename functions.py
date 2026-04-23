# ============================================================
# funciones.py
# Helper functions for the technical development of the thesis
# ============================================================

# ============================================================
# IMPORTS
# ============================================================
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from numpy.linalg import inv, LinAlgError
from scipy.stats import norm, loguniform

# sklearn — metrics and evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, make_scorer,
    silhouette_score, auc
)

# sklearn — models and preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (
    GridSearchCV, train_test_split, cross_val_score,
    RandomizedSearchCV, StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import FitFailedWarning

# imbalanced-learn
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, KMeansSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours

warnings.simplefilter('ignore', FitFailedWarning)
warnings.simplefilter('ignore', UserWarning)


# ============================================================
# DATA UTILITIES
# ============================================================

PALETA = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def _roc_multiclass(model, X_te, Y_te, title, path):
    classes = model.classes_
    Y_bin   = label_binarize(Y_te, classes=classes)
    prob    = model.predict_proba(X_te)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], prob[:, i])
        ax.plot(fpr, tpr, lw=2, color=PALETA[i % len(PALETA)],
                label=f"{cls}  (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), title=title,
           xlabel="False Positive Rate (FPR)",
           ylabel="True Positive Rate (TPR)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(linestyle='--', alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{path}.png", dpi=150, bbox_inches='tight')
    plt.show()


def _roc_binary(model, X_te, Y_te, title, path):
    prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(Y_te, prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color=PALETA[0],
            label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), title=title,
           xlabel="False Positive Rate (FPR)",
           ylabel="True Positive Rate (TPR)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle='--', alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{path}.png", dpi=150, bbox_inches='tight')
    plt.show()

def _counts_in_order(y: pd.Series) -> pd.Series:
    # Respect category order if categorical; otherwise sort by label
    if pd.api.types.is_categorical_dtype(y):
        return y.value_counts().reindex(y.cat.categories, fill_value=0)
    else:
        return y.value_counts().sort_index()

def _wrap_labels(labels, width=14):
    return ['\n'.join(textwrap.wrap(str(lbl), width=width)) for lbl in labels]

def _nice_bars(ax, x, h, labels, title):
    # bars with border and slight transparency
    bars = ax.bar(x, h, width=0.85, linewidth=0.8, edgecolor='black', alpha=0.9)

    # subtle grid
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    # very faint panel background
    ax.set_facecolor((0, 0, 0, 0.02))

    # cleaner spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(_wrap_labels(labels), rotation=25, ha='right')
    ax.set_title(title, pad=10, fontsize=12, weight='bold')

    # headroom so value labels fit above bars
    ymax = (h.max() * 1.15) if len(h) else 1.0
    ax.set_ylim(0, ymax)

    return bars

def _annotate_counts_percent(ax, bars, counts, total):
    for rect, c in zip(bars, counts):
        y = rect.get_height()
        pct = (c / total * 100.0) if total else 0.0
        ax.text(rect.get_x() + rect.get_width()/2, y,
                f'{c}\n{pct:.1f}%',
                ha='center', va='bottom', fontsize=9)

def plot_class_histograms(Y1, Y2, Y3, titles=('BMI WHO Classification','Glucose Alteration','HTA Classification')):
    mpl.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    Ys = [Y1, Y2, Y3]
    for ax, y, ttl in zip(axes, Ys, titles):
        counts_s = _counts_in_order(pd.Series(y))
        labels = list(counts_s.index)
        counts = counts_s.to_numpy()
        total = counts.sum()
        x = np.arange(len(labels))

        bars = _nice_bars(ax, x, counts, labels, ttl)
        _annotate_counts_percent(ax, bars, counts, total)

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')

        for xi in x:
            ax.axvline(xi - 0.5, color=(0,0,0,0.05), linewidth=0.8)

    plt.show()


def dummify(df, cols, drop_first=False, dummy_na=False, dtype='int8', prefix_sep='__'):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return pd.get_dummies(df, columns=cols, drop_first=drop_first,
                          dummy_na=dummy_na, dtype=dtype, prefix_sep=prefix_sep)


def dqr(data):
    cols     = pd.DataFrame(list(data.columns.values), columns=['Name'], index=list(data.columns.values))
    dtyp     = pd.DataFrame(data.dtypes, columns=['Type'])
    misval   = pd.DataFrame(data.isnull().sum(), columns=['N/A value'])
    presval  = pd.DataFrame(data.count(), columns=['Count values'])
    unival   = pd.DataFrame(columns=['Unique values'])
    minval   = pd.DataFrame(columns=['Min'])
    maxval   = pd.DataFrame(columns=['Max'])
    mean     = pd.DataFrame(data.mean(), columns=['Mean'])
    Std      = pd.DataFrame(data.std(), columns=['Std'])
    Var      = pd.DataFrame(data.var(), columns=['Var'])
    median   = pd.DataFrame(data.median(), columns=['Median'])
    skewness = pd.DataFrame(data.skew(), columns=['Skewness'])
    kurtosis = pd.DataFrame(data.kurtosis(), columns=['Kurtosis'])

    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except Exception:
            pass

    return (cols.join(dtyp).join(misval).join(presval).join(unival)
               .join(minval).join(maxval).join(mean).join(Std)
               .join(Var).join(median).join(skewness).join(kurtosis))


def drop_outliers_iqr(df, col, k=1.5):
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = df[col].between(Q1 - k * IQR, Q3 + k * IQR)
    return df[mask]


# ============================================================
# EVALUATION AND METRICS
# ============================================================

def best_threshold_roc_corner(y_true, y_prob):
    """
    Returns the threshold at the ROC curve point closest to the (0, 1) corner.

    Parameters
    ----------
    y_true  : true labels (shape [n_samples]).
    y_prob  : predicted probabilities for the positive class (shape [n_samples]).

    Returns
    -------
    threshold, fpr[idx], tpr[idx]
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    dist = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    idx  = np.argmin(dist)
    return float(thr[idx]), float(fpr[idx]), float(tpr[idx])


def best_thresholds_roc_corner_ovr(y_true, prob_matrix, classes):
    """
    Computes one threshold per class using the OvR (One-vs-Rest) strategy.
    The threshold is the ROC curve point closest to the (0, 1) corner.

    Parameters
    ----------
    y_true      : true labels (array-like).
    prob_matrix : predict_proba(X) -> shape [n_samples, n_classes] in the order of 'classes'.
    classes     : array with the class order from the model.

    Returns
    -------
    thresholds : np.array of shape [n_classes].
    """
    thresholds = np.zeros(len(classes), dtype=float)
    for j, cls in enumerate(classes):
        y_bin        = (y_true == cls).astype(int)
        fpr, tpr, thr = roc_curve(y_bin, prob_matrix[:, j])
        dist          = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
        k             = np.argmin(dist)
        thresholds[j] = thr[k] if len(thr) else 0.5
    return thresholds


def predict_with_thresholds_ovr(estimator, X, classes, thresholds):
    """If any class exceeds its threshold, pick the one with the largest (p_j - τ_j); otherwise argmax."""
    P        = estimator.predict_proba(X)
    margins  = P - thresholds[None, :]
    has_any  = (P >= thresholds[None, :]).any(axis=1)
    pred     = np.argmax(P, axis=1)
    pred_thr = np.argmax(margins, axis=1)
    pred[has_any] = pred_thr[has_any]
    return classes[pred]


def confusion_matrix_binary(model, X_test, Y_test, X_val, Y_val,
                             title="Confusion Matrix (ROC-corner threshold)"):
    # --- Threshold optimization ---
    probs_val  = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thr = roc_curve(Y_val, probs_val)
    dist       = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    thr_best   = thr[np.argmin(dist)]

    # --- Predictions ---
    probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_thr = (probs_test >= thr_best).astype(int)

    # --- Plot ---
    cm   = confusion_matrix(Y_test, y_pred_thr)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Altered"])
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(labelsize=10)

    for text in ax.texts:
        text.set_fontsize(13)
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.show()
    print(f"  Optimal threshold (ROC corner): {thr_best:.4f}")


def confusion_matrix_mult(model, Y_train, X_test, Y_test, X_val, Y_val,
                           title="Confusion Matrix (ROC-corner threshold)"):
    # --- Resolve classes ---
    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model, "named_steps") and "svc" in model.named_steps:
        classes = model.named_steps["svc"].classes_
    else:
        classes = np.unique(Y_train)

    # --- Threshold optimization ---
    P_val   = model.predict_proba(X_val)
    thr_ovr = best_thresholds_roc_corner_ovr(Y_val, P_val, classes)

    for name, delta in {"bajo": -0.03, "sobrepeso": -0.03, "obesidad": -0.03}.items():
        if name in classes:
            j = np.where(classes == name)[0][0]
            thr_ovr[j] = np.clip(thr_ovr[j] + delta, 0.0, 1.0)

    # --- Predictions ---
    y_pred_thr = predict_with_thresholds_ovr(model, X_test, classes, thr_ovr)

    # --- Plot ---
    cm   = confusion_matrix(Y_test, y_pred_thr, labels=classes)
    n    = len(classes)
    fig, ax = plt.subplots(figsize=(4 + n, 3 + n))  # escala con número de clases
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    for text in ax.texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.show()
    print("  Optimal thresholds per class (ROC corner):")
    for cls, t in zip(classes, thr_ovr):
        print(f"    {cls}: {t:.4f}")


def metrics(model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y):
    """
    Computes metrics using THRESHOLD(S) from the ROC 'good corner' (0,1) derived from VALIDATION.
    - Binary    : single threshold (OvR on the positive class).
    - Multiclass: OvR threshold per class (one per column of predict_proba/decision_function).

    Returns:
      metrics_df    : DataFrame with Accuracy, Precision, Recall (threshold-based) and ROC AUC (score-based).
      roc_corner_df : DataFrame with threshold(s) + (FPR, TPR, distance) per class.
    """
    classes   = np.array(sorted(np.unique(Y)))
    n_classes = len(classes)
    is_binary = (n_classes == 2)
    avg       = 'binary' if is_binary else 'macro'

    # --- internal helpers ---
    def _scores(m, X):
        if hasattr(m, "predict_proba"):
            P = m.predict_proba(X)
            if is_binary:
                pos_label = 1 if 1 in classes else classes[-1]
                pos_idx   = list(m.classes_).index(pos_label) if hasattr(m, "classes_") else list(classes).index(pos_label)
                return P[:, pos_idx]
            return P
        elif hasattr(m, "decision_function"):
            return m.decision_function(X)
        return m.predict(X)

    def _roc_corner_threshold(y_true_bin, scores_1d):
        fpr, tpr, thr = roc_curve(y_true_bin, scores_1d)
        dist = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
        idx  = np.argmin(dist)
        return float(thr[idx]), float(fpr[idx]), float(tpr[idx]), float(dist[idx])

    def _predict_with_thresholds_ovr(scores, cls_order, thresholds):
        margins  = scores - thresholds[None, :]
        has_any  = (scores >= thresholds[None, :]).any(axis=1)
        pred     = np.argmax(scores, axis=1)
        pred_thr = np.argmax(margins, axis=1)
        pred[has_any] = pred_thr[has_any]
        return np.array(cls_order)[pred]

    # --- scores per split ---
    score_tr = _scores(model, X_train)
    score_te = _scores(model, X_test)
    score_va = _scores(model, X_val)

    # --- threshold(s) derived from VALIDATION ---
    if is_binary:
        pos_label = 1 if 1 in classes else classes[-1]
        y_val_bin = (Y_val == pos_label).astype(int)
        thr, fpr_b, tpr_b, dist_b = _roc_corner_threshold(y_val_bin, score_va)
        roc_corner_df = pd.DataFrame({
            "class": [pos_label], "threshold": [thr],
            "FPR": [fpr_b], "TPR": [tpr_b], "distance": [dist_b],
        })
        yhat_tr = (score_tr >= thr).astype(int)
        yhat_te = (score_te >= thr).astype(int)
        yhat_va = (score_va >= thr).astype(int)
    else:
        if hasattr(model, "classes_"):
            cls_model = np.array(model.classes_)
        elif hasattr(model, "named_steps") and "svc" in model.named_steps:
            cls_model = np.array(model.named_steps["svc"].classes_)
        else:
            cls_model = classes

        if score_va.ndim == 1:
            raise ValueError("In multiclass mode a score matrix (n, C) was expected.")

        thr_list, fpr_list, tpr_list, dist_list = [], [], [], []
        for j, cls in enumerate(cls_model):
            y_bin = (Y_val == cls).astype(int)
            thr_j, fpr_j, tpr_j, dist_j = _roc_corner_threshold(y_bin, score_va[:, j])
            thr_list.append(thr_j); fpr_list.append(fpr_j)
            tpr_list.append(tpr_j); dist_list.append(dist_j)

        thresholds = np.array(thr_list)
        roc_corner_df = pd.DataFrame({
            "class": cls_model, "threshold": thresholds,
            "FPR": fpr_list, "TPR": tpr_list, "distance": dist_list,
        })
        yhat_tr = _predict_with_thresholds_ovr(score_tr, cls_model, thresholds)
        yhat_te = _predict_with_thresholds_ovr(score_te, cls_model, thresholds)
        yhat_va = _predict_with_thresholds_ovr(score_va, cls_model, thresholds)

    # --- metrics ---
    train_accuracy = accuracy_score(Y_train, yhat_tr)
    test_accuracy  = accuracy_score(Y_test,  yhat_te)
    val_accuracy   = accuracy_score(Y_val,   yhat_va)

    if is_binary:
        pos_label       = 1 if 1 in classes else classes[-1]
        train_precision = precision_score(Y_train, yhat_tr, average=avg, pos_label=pos_label, zero_division=0)
        test_precision  = precision_score(Y_test,  yhat_te, average=avg, pos_label=pos_label, zero_division=0)
        val_precision   = precision_score(Y_val,   yhat_va, average=avg, pos_label=pos_label, zero_division=0)
        train_recall    = recall_score(Y_train, yhat_tr, average=avg, pos_label=pos_label, zero_division=0)
        test_recall     = recall_score(Y_test,  yhat_te, average=avg, pos_label=pos_label, zero_division=0)
        val_recall      = recall_score(Y_val,   yhat_va, average=avg, pos_label=pos_label, zero_division=0)
        train_roc_auc   = roc_auc_score(Y_train, score_tr)
        test_roc_auc    = roc_auc_score(Y_test,  score_te)
        val_roc_auc     = roc_auc_score(Y_val,   score_va)
    else:
        train_precision = precision_score(Y_train, yhat_tr, average='macro', zero_division=0)
        test_precision  = precision_score(Y_test,  yhat_te, average='macro', zero_division=0)
        val_precision   = precision_score(Y_val,   yhat_va, average='macro', zero_division=0)
        train_recall    = recall_score(Y_train, yhat_tr, average='macro', zero_division=0)
        test_recall     = recall_score(Y_test,  yhat_te, average='macro', zero_division=0)
        val_recall      = recall_score(Y_val,   yhat_va, average='macro', zero_division=0)
        if isinstance(score_tr, np.ndarray) and score_tr.ndim == 2:
            train_roc_auc = roc_auc_score(Y_train, score_tr, multi_class='ovr', average='macro')
            test_roc_auc  = roc_auc_score(Y_test,  score_te, multi_class='ovr', average='macro')
            val_roc_auc   = roc_auc_score(Y_val,   score_va, multi_class='ovr', average='macro')
        else:
            warnings.warn("Could not compute macro OvR AUC robustly; returning NaN.")
            train_roc_auc = test_roc_auc = val_roc_auc = np.nan

    metrics_df = pd.DataFrame({
        'Train': [train_accuracy, train_precision, train_recall, train_roc_auc],
        'Test':  [test_accuracy,  test_precision,  test_recall,  test_roc_auc],
        'Val':   [val_accuracy,   val_precision,   val_recall,   val_roc_auc],
    }, index=["Accuracy (thr*)", "Precision (thr*)", "Recall (thr*)", "ROC AUC"])
    # * metrics computed with the threshold(s) selected on the validation set

    return metrics_df, roc_corner_df


# ============================================================
# MODEL COMPARISON
# ============================================================

def _stack_y(Y_train=None, Y_test=None, Y_val=None):
    parts = [y for y in (Y_train, Y_test, Y_val) if y is not None]
    if len(parts) == 0:
        raise ValueError("At least one Y vector is required to build Y_all.")
    return np.concatenate([np.asarray(y).ravel() for y in parts], axis=0)


def _call_metrics(metrics_fn, model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_all):
    """Supports metrics_fn returning either metrics_df or (metrics_df, roc_corner_df)."""
    out = metrics_fn(model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_all)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, None


def compare_models_multi_data(model_specs, *, metrics_fn, prefer_split="Test", float_fmt="{:.3f}"):
    """
    Compares multiple models by computing metrics per split and returns styled tables.

    Parameters
    ----------
    model_specs : list of dicts with keys:
        'name', 'model', 'X_train', 'X_test', 'X_val' (optional),
        'Y_train', 'Y_test', 'Y_val' (optional).
    metrics_fn  : function with signature (model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_all)
                  returning metrics_df or (metrics_df, roc_corner_df).
    prefer_split: split name for the simple slice ('Test' by default).
    float_fmt   : number format string for styled tables.

    Returns
    -------
    panel_df, slice_df, roc_table, styled_panel, styled_slice
    """
    results        = {}
    roc_info       = {}
    all_metric_names = set()
    all_splits     = set()

    for spec in model_specs:
        name    = spec["name"]
        model   = spec["model"]
        X_train = spec["X_train"]
        X_test  = spec["X_test"]
        X_val   = spec.get("X_val", None)
        Y_train = spec["Y_train"]
        Y_test  = spec["Y_test"]
        Y_val   = spec.get("Y_val", None)
        Y_all   = _stack_y(Y_train, Y_test, Y_val)

        mdf, rocdf = _call_metrics(metrics_fn, model, X_train, X_test, X_val,
                                   Y_train, Y_test, Y_val, Y_all)
        results[name] = mdf
        if rocdf is not None:
            r = rocdf.copy()
            r.insert(0, "model", name)
            roc_info[name] = r

        all_metric_names.update(mdf.index.tolist())
        all_splits.update(mdf.columns.tolist())

    all_metric_names = list(all_metric_names)
    all_splits       = list(all_splits)

    aligned  = {name: mdf.reindex(index=all_metric_names, columns=all_splits)
                for name, mdf in results.items()}
    panel_df = pd.concat(aligned, axis=1)

    slice_df  = panel_df.xs(prefer_split, axis=1, level=1) if prefer_split in all_splits else None
    roc_table = pd.concat(roc_info.values(), axis=0, ignore_index=True) if roc_info else None

    # ── Paleta y estilos base ──────────────────────────────────
    _H_BG   = "#1e2d3d"   # header background
    _H_FG   = "#ffffff"   # header text
    _I_BG   = "#f4f7fb"   # index cell background
    _I_FG   = "#1e2d3d"   # index cell text
    _SEP    = "#b0bec5"   # separator between model groups
    _CELL_B = "#e8edf3"   # row border
    _BEST   = "#17a589"   # best-value highlight (teal)
    _BEST_F = "#ffffff"

    _base = [
        {"selector": "table",
         "props": ("border-collapse: collapse; font-family: 'Segoe UI', Arial, sans-serif; "
                   "font-size: 11.5px; width: 100%;")},
        {"selector": "thead tr th",
         "props": (f"background-color: {_H_BG}; color: {_H_FG}; font-weight: 600; "
                   "padding: 9px 14px; text-align: center; white-space: nowrap; "
                   f"border-bottom: 2px solid {_SEP};")},
        {"selector": "th.row_heading, th.blank",
         "props": (f"background-color: {_I_BG}; color: {_I_FG}; font-weight: 600; "
                   f"padding: 7px 14px; text-align: left; border-right: 1px solid {_SEP};")},
        {"selector": "td",
         "props": (f"padding: 7px 14px; text-align: center; "
                   f"border-bottom: 1px solid {_CELL_B};")},
        {"selector": "caption",
         "props": ("caption-side: top; font-size: 13px; font-weight: 700; "
                   f"color: {_I_FG}; padding: 10px 4px 8px 4px; text-align: left; "
                   "font-family: 'Segoe UI', Arial, sans-serif;")},
        {"selector": "tr:hover td",
         "props": f"background-color: rgba(30, 45, 61, 0.04);"},
    ]

    def _best_style(v):
        return (f"background-color: {_BEST} !important; color: {_BEST_F}; "
                "font-weight: 700;" if v else "")

    def _style_panel(df, *, group=3):
        styles = list(_base)
        # vertical separator between model groups
        for pos in range(group, df.shape[1], group):
            styles.append({
                "selector": (f"td:nth-child({pos + 2}), "
                             f"thead th:nth-child({pos + 2})"),
                "props": f"border-left: 2px solid {_SEP};",
            })

        sty = (df.style
               .format(float_fmt)
               .background_gradient(axis=None, cmap="Blues", vmin=0.0, vmax=1.0)
               .set_table_styles(styles))

        if isinstance(df.columns, pd.MultiIndex):
            for split in df.columns.levels[1]:
                if split in df.columns.get_level_values(1):
                    def _hi(s):
                        is_max = s == s.max()
                        return [_best_style(v) for v in is_max]
                    sty = sty.apply(_hi,
                                    subset=pd.IndexSlice[:, pd.IndexSlice[:, split]],
                                    axis=1)
        else:
            def _hi(s):
                is_max = s == s.max()
                return [_best_style(v) for v in is_max]
            sty = sty.apply(_hi, axis=1)
        return sty

    def _style_slice(df):
        if df is None:
            return None
        styles = list(_base)
        sty = (df.style
               .format(float_fmt)
               .background_gradient(axis=None, cmap="Blues", vmin=0.0, vmax=1.0)
               .set_table_styles(styles))
        def _hi(s):
            is_max = s == s.max()
            return [_best_style(v) for v in is_max]
        return sty.apply(_hi, axis=1)

    return panel_df, slice_df, roc_table, _style_panel(panel_df), _style_slice(slice_df)


# ============================================================
# COEFFICIENT IMPORTANCE — LOGISTIC REGRESSION
# ============================================================

_CLASE_COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _coef_matrix_and_classes(model: BaseEstimator):
    coef    = np.asarray(model.coef_)
    classes = np.asarray(getattr(model, "classes_", np.arange(coef.shape[0])))
    return coef, classes


def _aggregate_multiclass(coef: np.ndarray, how: str = "abs_mean") -> np.ndarray:
    if coef.ndim == 1:
        return np.abs(coef)
    return np.abs(coef).max(axis=0) if how == "abs_max" else np.abs(coef).mean(axis=0)


def _shorten(s: str, max_chars: int = 28) -> str:
    s = str(s)
    return s if len(s) <= max_chars else s[:max_chars - 1] + "..."


def plot_logreg_coefs_panel_safe(
    df_coefs: pd.DataFrame,
    model_name: str,
    top_k: int = 5,
    multiclass_agg: str = "abs_max",
    max_fig_w: float = 14.0,
    max_fig_h: float = 9.0,
    label_max_chars: int = 28,
):
    """
    Visualize logistic regression coefficient importance from a
    lr_coef_importance() DataFrame.

    Receives the already-computed DataFrame directly, avoiding any
    recalculation of coefficients. Sorting and ranking use the same
    abs_coef_std column produced by lr_coef_importance.

    Parameters
    ----------
    df_coefs     : DataFrame returned by lr_coef_importance().
    model_name   : string label for the plot title.
    top_k        : number of top features to display.
    multiclass_agg: 'abs_max' or 'abs_mean' for cross-class ranking.
    max_fig_w    : max figure width in inches.
    max_fig_h    : max figure height in inches.
    label_max_chars: max characters for y-axis labels.
    """

    sort_col     = "abs_coef_std" if "abs_coef_std" in df_coefs.columns else "z"
    is_multiclass = "class" in df_coefs.columns

    fig_h = min(max_fig_h, max(4.0, 0.45 * top_k + 1.5))
    fig, ax = plt.subplots(figsize=(max_fig_w, fig_h))

    if is_multiclass:
        # --------------------------------------------------------------
        # MULTICLASS — usa abs_coef_std de cada clase para rankear
        # --------------------------------------------------------------
        classes = df_coefs["class"].unique()

        # Ranking global: max abs_coef_std de una feature entre todas las clases
        if multiclass_agg == "abs_mean":
            global_imp = (
                df_coefs.groupby("feature")[sort_col].mean()
            )
        else:  # abs_max
            global_imp = (
                df_coefs.groupby("feature")[sort_col].max()
            )

        top_features = (
            global_imp.sort_values(ascending=False)
                       .head(top_k)
                       .index.tolist()
        )
        # bottom-to-top para barh
        top_features = top_features[::-1]

        n_cls  = len(classes)
        bar_h  = 0.8 / n_cls
        y_base = np.arange(len(top_features), dtype=float)
        x_vals_all    = []
        legend_handles = []

        for c, cls in enumerate(classes):
            df_cls = df_coefs[df_coefs["class"] == cls].set_index("feature")
            vals   = np.array([
                df_cls.loc[f, "coef_std"] if f in df_cls.index else 0.0
                for f in top_features
            ])
            offset = (c - (n_cls - 1) / 2.0) * bar_h
            color  = _CLASE_COLORES[c % len(_CLASE_COLORES)]

            ax.barh(
                y_base + offset, vals,
                height=bar_h * 0.88,
                color=color,
                edgecolor="black", linewidth=0.45, alpha=0.88,
            )
            x_vals_all.extend(vals.tolist())

            legend_handles.append(
                mpatches.Patch(
                    facecolor=color, edgecolor="black", linewidth=0.8,
                    label=f"Class '{cls}':  + pushes toward '{cls}'  |  − pulls away"
                )
            )

        feats_labels = [_shorten(f, label_max_chars) for f in top_features]
        ax.set_yticks(y_base)
        ax.set_yticklabels(feats_labels, fontsize=15)

        legend = ax.legend(
            handles=legend_handles,
            title="Color = Class  (standardised coefficients)",
            title_fontsize=13,
            fontsize=12,
            loc="lower right",
            framealpha=0.92,
            edgecolor="gray",
            borderpad=1.0,
            labelspacing=0.8,
        )
        legend.get_title().set_fontweight("bold")

        x_max = max(1e-9, max(abs(v) for v in x_vals_all))
        ax.set_xlim(-x_max * 1.18, x_max * 1.18)
        ax.set_xlabel(
            "Standardised coefficient  (positive → right,  negative → left)",
            fontsize=14
        )

    else:
        df_top = df_coefs.head(top_k).iloc[::-1]  
        feats  = [_shorten(f, label_max_chars) for f in df_top["feature"]]
        vals   = df_top["coef_std"].values if "coef_std" in df_top.columns \
                 else df_top["coef"].values

        y          = np.arange(len(feats))
        colors_bar = np.where(vals >= 0, "#2e8b57", "#c0392b")
        ax.barh(y, vals, edgecolor="black", linewidth=0.7, alpha=0.9, color=colors_bar)
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=15)

        x_max = max(1e-9, np.max(np.abs(vals)))
        ax.set_xlim(-x_max * 1.28, x_max * 1.28)

        for yi, v in zip(y, vals):
            ax.text(
                v + (0.025 * x_max if v >= 0 else -0.025 * x_max),
                yi, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=14
            )

        legend_handles = [
            mpatches.Patch(facecolor="#2e8b57", edgecolor="black",
                           label="Positive → increases P(positive class)"),
            mpatches.Patch(facecolor="#c0392b", edgecolor="black",
                           label="Negative → decreases P(positive class)"),
        ]
        legend = ax.legend(
            handles=legend_handles,
            title="Color meaning  (standardised coefficients)",
            title_fontsize=13, fontsize=13,
            loc="lower right", framealpha=0.92,
            edgecolor="gray", borderpad=1.0,
        )
        legend.get_title().set_fontweight("bold")
        ax.set_xlabel(
            "Standardised coefficient  (positive → right,  negative → left)",
            fontsize=14
        )

    ax.set_title(
        f"{model_name} — Top {top_k} features by coefficient magnitude",
        fontsize=17, weight="bold", pad=14
    )
    ax.axvline(0, color="black", linewidth=1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    filename = f"importancia_coeficientes_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.show()

    
def lr_coef_importance(model, X, *, estandarizar=True, multinomial_reference="last"):
    """
    Coefficient importance and Wald significance for sklearn LogisticRegression models.
    - Binary     : observed Fisher information (X'WX)^-1 with W = p*(1-p).
    - Multinomial: observed information sum_i((x_i x_i^T) ⊗ (Diag(p_i) - p_i p_i^T)),
                   reparametrised as (beta_k - beta_ref) for k != ref to obtain interpretable
                   OR and p-values for "class k vs reference class".

    Parameters
    ----------
    model                 : fitted sklearn.linear_model.LogisticRegression.
    X                     : pd.DataFrame or np.ndarray with the SAME transformations used at training time.
    estandarizar          : bool, if True adds coef_std and or_per_sd for interpretability.
    multinomial_reference : {"last", "first", int} reference class for multinomial reparametrisation.

    Returns
    -------
    df : pd.DataFrame
        Binary      : columns [feature, coef, odds_ratio, se, z, p>|z|, (opt) coef_std, abs_coef_std, or_per_sd]
        Multinomial : row index (class, feature) with same columns per class vs reference.
    """
    if hasattr(X, "to_numpy"):
        Xn         = X.to_numpy()
        feat_names = list(X.columns)
    else:
        Xn         = np.asarray(X)
        feat_names = [f"x{i}" for i in range(Xn.shape[1])]

    coef      = np.asarray(model.coef_)
    if coef.ndim == 1:
        coef  = coef[None, :]
    intercept = np.asarray(model.intercept_).ravel()
    n_classes, p = coef.shape

    X1 = np.column_stack([np.ones(len(Xn)), Xn])
    p1 = p + 1

    # ── Binary ────────────────────────────────────────────────
    if n_classes == 1:
        beta  = np.r_[intercept[0], coef[0]]
        eta   = X1 @ beta
        p_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w     = p_hat * (1.0 - p_hat)
        XtWX  = X1.T @ (X1 * w[:, None])
        try:
            cov = inv(XtWX)
        except LinAlgError:
            cov = np.linalg.pinv(XtWX)

        se   = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        z    = beta / se
        pval = 2.0 * (1.0 - norm.cdf(np.abs(z)))

        rows = {
            "feature":    feat_names,
            "coef":       beta[1:],
            "odds_ratio": np.exp(beta[1:]),
            "se":         se[1:],
            "z":          z[1:],
            "p>|z|":      pval[1:],
        }

        if estandarizar:
            scale             = Xn.std(axis=0, ddof=0)
            scale[scale == 0] = 1.0
            coef_std          = coef[0] * scale
            rows.update({
                "coef_std":     coef_std,
                "abs_coef_std": np.abs(coef_std),
                "or_per_sd":    np.exp(coef[0] * scale),
            })

        sort_col = "abs_coef_std" if estandarizar else "z"
        df = (
            pd.DataFrame(rows)
            .sort_values(sort_col, ascending=False)
            .reset_index(drop=True)
        )

        df.attrs["intercept"]     = {"coef": float(beta[0]), "odds_ratio": float(np.exp(beta[0])),
                                     "se": float(se[0]), "z": float(z[0]),
                                     "p>|z|": float(pval[0]), "name": "const"}
        df.attrs["cov_beta"]      = cov
        df.attrs["beta_full"]     = beta
        df.attrs["feature_order"] = ["const"] + feat_names
        return df

    # ── Multinomial ───────────────────────────────────────────
    K         = n_classes
    B         = np.column_stack([intercept.reshape(-1, 1), coef])
    beta_full = B.reshape(-1)

    scores      = X1 @ B.T
    scores_clip = np.clip(scores, -30, 30)
    es          = np.exp(scores_clip)
    P           = es / es.sum(axis=1, keepdims=True)

    I = np.zeros((K * p1, K * p1), dtype=float)
    for i in range(P.shape[0]):
        pi = P[i]
        Vi = np.diag(pi) - np.outer(pi, pi)
        xi = X1[i:i + 1, :]
        I += np.kron(Vi, xi.T @ xi)

    try:
        cov_full = inv(I)
    except LinAlgError:
        cov_full = np.linalg.pinv(I)

    if multinomial_reference == "last":
        ref = K - 1
    elif multinomial_reference == "first":
        ref = 0
    else:
        ref = int(multinomial_reference)
        if not (0 <= ref < K):
            raise ValueError("multinomial_reference must be 'last', 'first', or a valid class index.")

    blocks    = []
    class_idx = []
    I_p1      = np.eye(p1)
    for k in range(K):
        if k == ref:
            continue
        row = np.zeros((p1, K * p1))
        row[:, k * p1:(k + 1) * p1]     =  I_p1
        row[:, ref * p1:(ref + 1) * p1] = -I_p1
        blocks.append(row)
        class_idx.append(k)

    T         = np.vstack(blocks)
    theta     = T @ beta_full
    cov_theta = T @ cov_full @ T.T
    se_theta  = np.sqrt(np.clip(np.diag(cov_theta), 0, np.inf))
    z_theta   = theta / se_theta
    p_theta   = 2.0 * (1.0 - norm.cdf(np.abs(z_theta)))

    if estandarizar:
        scale             = Xn.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0

    # ── Clases de contraste (K-1) ─────────────────────────────
    records = []
    for j, k in enumerate(class_idx):
        offset = j * p1
        coefs  = theta[offset + 1: offset + p1]
        ses    = se_theta[offset + 1: offset + p1]
        zs     = z_theta[offset + 1: offset + p1]
        ps     = p_theta[offset + 1: offset + p1]

        for idx_f, fname in enumerate(feat_names):
            row = {
                "class":      model.classes_[k],
                "ref_class":  model.classes_[ref],
                "feature":    fname,
                "coef":       coefs[idx_f],
                "odds_ratio": np.exp(coefs[idx_f]),
                "se":         ses[idx_f],
                "z":          zs[idx_f],
                "p>|z|":      ps[idx_f],
            }
            if estandarizar:
                cstd = coefs[idx_f] * scale[idx_f]
                row.update({
                    "coef_std":     cstd,
                    "abs_coef_std": abs(cstd),
                    "or_per_sd":    np.exp(cstd),
                })
            records.append(row)

    # ── Clase de referencia — coeficientes crudos ─────────────
    # No tiene p-values porque es el baseline del modelo,
    # pero se incluye para visualizar las 4 clases en el plot.
    for idx_f, fname in enumerate(feat_names):
        raw_coef = coef[ref, idx_f]
        row = {
            "class":      model.classes_[ref],
            "ref_class":  "baseline",
            "feature":    fname,
            "coef":       raw_coef,
            "odds_ratio": np.exp(raw_coef),
            "se":         np.nan,
            "z":          np.nan,
            "p>|z|":      np.nan,
        }
        if estandarizar:
            cstd = raw_coef * scale[idx_f]
            row.update({
                "coef_std":     cstd,
                "abs_coef_std": abs(cstd),
                "or_per_sd":    np.exp(cstd),
            })
        records.append(row)

    sort_col = "abs_coef_std" if estandarizar else "z"
    df = (
        pd.DataFrame(records)
        .sort_values(["class", sort_col], ascending=[True, False])
        .reset_index(drop=True)
    )

    intercepts = []
    for j, k in enumerate(class_idx):
        o = j * p1
        intercepts.append({
            "class": model.classes_[k], "ref_class": model.classes_[ref],
            "name": "const", "coef": float(theta[o]),
            "odds_ratio": float(np.exp(theta[o])), "se": float(se_theta[o]),
            "z": float(z_theta[o]), "p>|z|": float(p_theta[o]),
        })

    df.attrs["intercepts"]      = intercepts
    df.attrs["cov_theta"]       = cov_theta
    df.attrs["beta_rel"]        = theta
    df.attrs["feature_order"]   = ["const"] + feat_names
    df.attrs["classes"]         = list(model.classes_)
    df.attrs["reference_class"] = model.classes_[ref]
    return df


# ============================================================
# OVERSAMPLING — SMOTE
# ============================================================

def _auto_k_neighbors(y):
    c = Counter(pd.Series(y).to_numpy())
    return max(1, min(3, min(c.values()) - 1))


def _auto_m_neighbors(y):
    c = Counter(pd.Series(y).to_numpy())
    return max(3, min(10, min(c.values()) + 1))


def _cat_indices(X: pd.DataFrame):
    if isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
        return [X.columns.get_loc(c) for c in cat_cols], list(cat_cols)
    return [], []


def build_strategy_to_majority(y, ratio=0.6, max_per_class=None):
    counts = Counter(pd.Series(y).to_numpy())
    M      = max(counts.values())
    target = int(np.ceil(M * ratio))
    strategy = {}
    for cls, cnt in counts.items():
        if cnt < target:
            n_obj = min(target, max_per_class) if max_per_class is not None else target
            strategy[cls] = n_obj
    return strategy


def _make_sampler_base(X, y, sampling_strategy, sampler, k_neighbors, m_neighbors, random_state):
    k       = _auto_k_neighbors(y) if k_neighbors is None else k_neighbors
    m       = _auto_m_neighbors(y) if m_neighbors is None else m_neighbors
    cat_idx, _ = _cat_indices(X)

    if len(cat_idx) > 0:
        return SMOTENC(categorical_features=cat_idx,
                       sampling_strategy=sampling_strategy,
                       k_neighbors=k, random_state=random_state)
    if sampler == "borderline1":
        return BorderlineSMOTE(kind='borderline-1', sampling_strategy=sampling_strategy,
                               k_neighbors=k, m_neighbors=m, random_state=random_state)
    if sampler == "borderline2":
        return BorderlineSMOTE(kind='borderline-2', sampling_strategy=sampling_strategy,
                               k_neighbors=k, m_neighbors=m, random_state=random_state)
    if sampler == "kmeans":
        return KMeansSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k,
                           kmeans_estimator=5, cluster_balance_threshold=0.05,
                           density_exponent='auto', random_state=random_state)
    return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=random_state)


def smote_fit_resample(X_train, y_train, *,
                       ratio=0.6, max_per_class=None,
                       sampler="smote", clean="none",
                       k_neighbors=None, m_neighbors=None,
                       random_state=42, verbose=True):
    """
    Conservative oversampling with optional cleaning step.

    Parameters
    ----------
    sampler : 'smote' | 'borderline1' | 'borderline2' | 'kmeans'
    clean   : 'none'  | 'tomek' | 'enn'
    """
    X_df  = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y_arr = pd.Series(y_train).to_numpy()

    if verbose:
        print("Original Distribution:", Counter(y_arr))

    strategy = build_strategy_to_majority(y_arr, ratio=ratio, max_per_class=max_per_class)
    if len(strategy) == 0:
        if verbose:
            print("Oversampling not applied (empty strategy).")
        return X_df.to_numpy(), y_arr

    base = _make_sampler_base(X_df, y_arr, strategy, sampler, k_neighbors, m_neighbors, random_state)

    if clean == "none":
        sampler_pipe = base
    else:
        if sampler == "smote":
            if clean == "tomek":
                sampler_pipe = SMOTETomek(smote=base, random_state=random_state)
            else:
                sampler_pipe = SMOTEENN(smote=base, random_state=random_state)
        else:
            cleaner      = TomekLinks() if clean == "tomek" else EditedNearestNeighbours(n_neighbors=3)
            sampler_pipe = ImbPipeline(steps=[('over', base), ('clean', cleaner)])

    X_sm, y_sm = sampler_pipe.fit_resample(X_df.to_numpy(), y_arr)

    if verbose:
        used_k = k_neighbors if k_neighbors is not None else _auto_k_neighbors(y_arr)
        print(f"sampler={sampler}, clean={clean}, k_neighbors={used_k}")
        print("Strategy:", strategy)
        print("SMOTE Distribution:", Counter(y_sm))
    return X_sm, y_sm


# ============================================================
# HYPERPARAMETER SEARCH — SVM
# ============================================================

def svc_hiper_search_mult(X_train, Y_train):
    """Randomized hyperparameter search for multiclass SVC."""
    skf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = Pipeline([("svc", SVC(kernel="rbf", probability=True, random_state=42))])

    sw_bal  = compute_sample_weight(class_weight="balanced", y=Y_train)
    sw_aggr = sw_bal ** 1.4

    param_dist = {
        "svc__C":            loguniform(1e-2, 1e3),
        "svc__gamma":        loguniform(1e-4, 1e1),
        "svc__class_weight": [None, "balanced"],
        "svc__tol":          loguniform(1e-5, 1e-2),
        "svc__shrinking":    [True, False],
        "svc__cache_size":   [200, 500, 1000],
    }

    scoring = {
        "recall_macro":  "recall_macro",
        "f1_macro":      "f1_macro",
        "balanced_acc":  "balanced_accuracy",
        "auc_ovr_w":     "roc_auc_ovr_weighted",
    }

    search = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_dist,
        n_iter=100, scoring=scoring, refit="recall_macro",
        cv=skf, n_jobs=-1, verbose=1, random_state=42,
    )
    search.fit(X_train, Y_train, svc__sample_weight=sw_aggr)
    return search


def svc_hiper_search_binary(X_train, Y_train):
    """Randomized hyperparameter search for binary SVC."""
    skf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = Pipeline([("svc", SVC(kernel="rbf", probability=True, random_state=42))])

    param_dist = {
        "svc__C":            loguniform(1e-1, 1e2),
        "svc__gamma":        loguniform(1e-3, 1e0),
        "svc__class_weight": [None, "balanced"],
        "svc__tol":          loguniform(1e-5, 1e-2),
    }

    search = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_dist,
        n_iter=100, scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1", cv=skf, n_jobs=-1, verbose=1, random_state=42,
    )
    search.fit(X_train, Y_train)
    return search


# ============================================================
# CLUSTER-THEN-CLASSIFY
# ============================================================

@dataclass
class _ClusterModel:
    pure:      bool
    label_idx: Optional[int]          = None
    clf:       Optional[BaseEstimator] = None


class ClusterThenClassify(BaseEstimator, ClassifierMixin):
    """
    Cluster-then-classify:
      - Global clustering (KMeans | GMM).
      - Per cluster: if 100 % pure -> direct label assignment; otherwise -> train a local clf.
    """
    def __init__(self,
                 clusterer:      str                       = "kmeans",
                 k_range:        Tuple[int, int]           = (2, 8),
                 base_estimator: Optional[BaseEstimator]   = None,
                 random_state:   int                       = 42):
        self.clusterer      = clusterer
        self.k_range        = k_range
        self.base_estimator = base_estimator
        self.random_state   = random_state

    def _make_clusterer(self, n_clusters):
        if self.clusterer == "kmeans":
            return KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        elif self.clusterer == "gmm":
            return GaussianMixture(n_components=n_clusters, covariance_type="full",
                                   random_state=self.random_state)
        raise ValueError("clusterer must be 'kmeans' or 'gmm'")

    def _fit_clusterer(self, X, best_k):
        self.clusterer_ = self._make_clusterer(best_k).fit(X)
        return self

    def _predict_clusters(self, X):
        return self.clusterer_.predict(X)

    def _default_clf(self, n_classes):
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return SVC(kernel="rbf", probability=True, class_weight="balanced",
                   C=1.0, gamma="scale", random_state=self.random_state)

    def fit(self, X, y):
        X          = np.asarray(X)
        self.le_   = LabelEncoder().fit(y)
        y_idx      = self.le_.transform(y)
        self.classes_   = self.le_.classes_
        self.n_classes_ = len(self.classes_)

        k_min, k_max = self.k_range
        k_candidates = [k for k in range(k_min, k_max + 1) if k >= 2]
        best_k, best_score = None, -np.inf

        for k in k_candidates:
            clus   = self._make_clusterer(k)
            labels = clus.fit_predict(X) if self.clusterer == "kmeans" else clus.fit(X).predict(X)
            if len(np.unique(labels)) > 1:
                try:
                    sc = silhouette_score(X, labels)
                except Exception:
                    sc = -np.inf
            else:
                sc = -np.inf
            if sc > best_score:
                best_k, best_score = k, sc

        if best_k is None:
            best_k = 2

        self._fit_clusterer(X, best_k)
        z                = self._predict_clusters(X)
        self.n_clusters_ = best_k

        self.cluster_models_: Dict[int, _ClusterModel] = {}
        for c in range(best_k):
            idx    = np.where(z == c)[0]
            Xc, yc = X[idx], y_idx[idx]
            counts = Counter(yc)
            if len(counts) == 1:
                label_idx = next(iter(counts.keys()))
                self.cluster_models_[c] = _ClusterModel(pure=True, label_idx=label_idx)
            else:
                clf = self._default_clf(self.n_classes_)
                clf.fit(Xc, yc)
                self.cluster_models_[c] = _ClusterModel(pure=False, clf=clf)
        return self

    def predict(self, X):
        check_is_fitted(self, ["clusterer_", "cluster_models_", "le_"])
        X          = np.asarray(X)
        z          = self._predict_clusters(X)
        y_pred_idx = np.empty(len(X), dtype=int)
        for c in range(self.n_clusters_):
            mask = (z == c)
            if not np.any(mask):
                continue
            cm = self.cluster_models_[c]
            if cm.pure:
                y_pred_idx[mask] = cm.label_idx
            else:
                y_pred_idx[mask] = cm.clf.predict(X[mask])
        return self.le_.inverse_transform(y_pred_idx)

    def predict_proba(self, X):
        check_is_fitted(self, ["clusterer_", "cluster_models_", "le_"])
        X     = np.asarray(X)
        z     = self._predict_clusters(X)
        proba = np.zeros((len(X), self.n_classes_), dtype=float)
        for c in range(self.n_clusters_):
            mask = (z == c)
            if not np.any(mask):
                continue
            cm = self.cluster_models_[c]
            if cm.pure:
                proba[mask, cm.label_idx] = 1.0
            else:
                p = cm.clf.predict_proba(X[mask])
                if p.shape[1] != self.n_classes_:
                    p_full = np.zeros((p.shape[0], self.n_classes_))
                    for j, cls_idx in enumerate(cm.clf.classes_):
                        p_full[:, cls_idx] = p[:, j]
                    p = p_full
                proba[mask] = p
        return proba


def train_cluster_then_classify(X, y,
                                clusterer:      str                     = "kmeans",
                                k_range:        Tuple[int, int]         = (2, 8),
                                base_estimator: Optional[BaseEstimator] = None,
                                random_state:   int                     = 42):
    """
    Fits and returns a ClusterThenClassify model.
    X, y must already be preprocessed / scaled.
    """
    return ClusterThenClassify(
        clusterer=clusterer, k_range=k_range,
        base_estimator=base_estimator, random_state=random_state,
    ).fit(X, y)
