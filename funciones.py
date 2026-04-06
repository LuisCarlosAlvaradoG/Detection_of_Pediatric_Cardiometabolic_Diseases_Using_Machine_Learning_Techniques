import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# pip install imbalanced-learn
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score,  RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from numpy.linalg import inv, LinAlgError
from scipy.stats import norm, loguniform
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.simplefilter('ignore', FitFailedWarning)
warnings.simplefilter('ignore', UserWarning)

def dummify(df, cols, drop_first=False, dummy_na=False, dtype='int8', prefix_sep='__'):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas no encontradas: {missing}")
    return pd.get_dummies(df, columns=cols, drop_first=drop_first,
                          dummy_na=dummy_na, dtype=dtype, prefix_sep=prefix_sep)

def dqr(data):

    cols = pd.DataFrame(list(data.columns.values),
                           columns=['Name'],
                           index=list(data.columns.values))
    dtyp = pd.DataFrame(data.dtypes,columns=['Type'])
    misval = pd.DataFrame(data.isnull().sum(),
                                  columns=['N/A value'])
    presval = pd.DataFrame(data.count(),
                                  columns=['Count values'])
    unival = pd.DataFrame(columns=['Unique values'])
    minval = pd.DataFrame(columns=['Min'])
    maxval = pd.DataFrame(columns=['Max'])
    mean =pd.DataFrame(data.mean(), columns=['Mean'])
    Std =pd.DataFrame(data.std(), columns=['Std'])
    Var =pd.DataFrame(data.var(), columns=['Var'])
    median =pd.DataFrame(data.median(), columns=['Median'])

    skewness = pd.DataFrame(data.skew(), columns=['Skewness'])
    kurtosis = pd.DataFrame(data.kurtosis(), columns=['Kurtosis'])

    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except:
            pass

    # Juntar todas las tablas
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval).join(mean).join(Std).join(Var).join(median).join(skewness).join(kurtosis)

def drop_outliers_iqr(df, col, k=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = df[col].between(Q1 - k*IQR, Q3 + k*IQR)
    return df[mask]

def metrics(model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y):
    """
    Calcula métricas usando UMBRAL(ES) por 'esquina buena' de ROC tomados de VALIDACIÓN.
    - Binaria: umbral único (OvR sobre la clase positiva).
    - Multiclase: umbral OvR por clase (uno por columna de predict_proba/decision_function).

    Devuelve:
      metrics_df: DataFrame con Accuracy, Precision, Recall (usando umbrales) y ROC AUC (con scores).
      roc_corner_df: DataFrame con threshold(s) + (FPR, TPR, distancia) por clase.
    """
    classes = np.array(sorted(np.unique(Y)))
    n_classes = len(classes)
    is_binary = (n_classes == 2)
    avg = 'binary' if is_binary else 'macro'

    # --- Helpers ---
    def _scores(m, X):
        # Devuelve: binaria -> vector (n,); multiclase -> matriz (n, C)
        if hasattr(m, "predict_proba"):
            P = m.predict_proba(X)
            if is_binary:
                # encuentra índice de la clase "positiva"
                pos_label = 1 if 1 in classes else classes[-1]
                pos_idx = list(m.classes_).index(pos_label) if hasattr(m, "classes_") else list(classes).index(pos_label)
                return P[:, pos_idx]
            else:
                return P  # (n, C) en el orden m.classes_
        elif hasattr(m, "decision_function"):
            S = m.decision_function(X)
            return S  # puede ser (n,) en binaria o (n, C) en multiclase
        else:
            # Último recurso (no ideal para AUC ni ROC); usamos predicciones como scores
            return m.predict(X)

    def _roc_corner_threshold(y_true_bin, scores_1d):
        # Calcula umbral por distancia mínima a la esquina buena (0,1)
        fpr, tpr, thr = roc_curve(y_true_bin, scores_1d)
        dist = np.sqrt(fpr**2 + (1 - tpr)**2)
        idx = np.argmin(dist)
        # Si sklearn devuelve umbrales con shape len(tpr), thr tiene len(fpr/tpr); es correcto
        return (float(thr[idx]), float(fpr[idx]), float(tpr[idx]), float(dist[idx]))

    def _predict_with_thresholds_ovr(scores, cls_order, thresholds):
        """
        scores: (n, C) en el orden cls_order
        thresholds: vector (C,)
        Regla: si alguna clase supera τ_j, elegir la de mayor (p_j - τ_j); si ninguna, argmax normal.
        """
        margins = scores - thresholds[None, :]
        has_any = (scores >= thresholds[None, :]).any(axis=1)
        pred = np.argmax(scores, axis=1)           # fallback sin umbral
        pred_thr = np.argmax(margins, axis=1)
        pred[has_any] = pred_thr[has_any]
        return np.array(cls_order)[pred]

    # --- Scores en cada split ---
    score_tr = _scores(model, X_train)
    score_te = _scores(model, X_test)
    score_va = _scores(model, X_val)

    # --- Calcular umbral(es) a partir de VALIDACIÓN ---
    if is_binary:
        # binaria: score_* son vectores 1D
        # construir y binario respecto a la clase positiva
        pos_label = 1 if 1 in classes else classes[-1]
        y_val_bin = (Y_val == pos_label).astype(int)
        thr, fpr_b, tpr_b, dist_b = _roc_corner_threshold(y_val_bin, score_va)
        thresholds = np.array([thr])  # para reportar homogéneo
        roc_corner_df = pd.DataFrame({
            "class": [pos_label],
            "threshold": [thr],
            "FPR": [fpr_b],
            "TPR": [tpr_b],
            "distance": [dist_b]
        })
        # Predicciones con umbral en los 3 sets
        yhat_tr = (score_tr >= thr).astype(int)
        yhat_te = (score_te >= thr).astype(int)
        yhat_va = (score_va >= thr).astype(int)
    else:
        # multiclase: score_* deben ser (n, C)
        # Alinear orden de clases del modelo vs 'classes'
        if hasattr(model, "classes_"):
            cls_model = np.array(model.classes_)
        elif hasattr(model, "named_steps") and "svc" in model.named_steps:
            cls_model = np.array(model.named_steps["svc"].classes_)
        else:
            # intentamos deducirlo por columnas de predict_proba/decision_function
            cls_model = classes

        # asegurar que score_* sean 2D (si decision_function devolvió 1D algo va mal)
        if score_va.ndim == 1:
            raise ValueError("En multiclase se esperaba matriz de scores (n, C). Revisa predict_proba/decision_function.")

        # thresholds OvR por clase (con respecto a VALIDACIÓN)
        thr_list, fpr_list, tpr_list, dist_list = [], [], [], []
        for j, cls in enumerate(cls_model):
            y_bin = (Y_val == cls).astype(int)
            s_col = score_va[:, j]
            thr_j, fpr_j, tpr_j, dist_j = _roc_corner_threshold(y_bin, s_col)
            thr_list.append(thr_j); fpr_list.append(fpr_j); tpr_list.append(tpr_j); dist_list.append(dist_j)

        thresholds = np.array(thr_list)
        roc_corner_df = pd.DataFrame({
            "class": cls_model,
            "threshold": thresholds,
            "FPR": fpr_list,
            "TPR": tpr_list,
            "distance": dist_list
        })

        # Predicciones con umbrales OvR
        yhat_tr = _predict_with_thresholds_ovr(score_tr, cls_model, thresholds)
        yhat_te = _predict_with_thresholds_ovr(score_te, cls_model, thresholds)
        yhat_va = _predict_with_thresholds_ovr(score_va, cls_model, thresholds)

    # --- Métricas (Accuracy/Precision/Recall usando yhat con umbrales) ---
    train_accuracy = accuracy_score(Y_train, yhat_tr)
    test_accuracy  = accuracy_score(Y_test,  yhat_te)
    val_accuracy   = accuracy_score(Y_val,   yhat_va)

    if is_binary:
        pos_label = 1 if 1 in classes else classes[-1]
        train_precision = precision_score(Y_train, yhat_tr, average=avg, pos_label=pos_label, zero_division=0)
        test_precision  = precision_score(Y_test,  yhat_te, average=avg, pos_label=pos_label, zero_division=0)
        val_precision   = precision_score(Y_val,   yhat_va, average=avg, pos_label=pos_label, zero_division=0)

        train_recall = recall_score(Y_train, yhat_tr, average=avg, pos_label=pos_label, zero_division=0)
        test_recall  = recall_score(Y_test,  yhat_te, average=avg, pos_label=pos_label, zero_division=0)
        val_recall   = recall_score(Y_val,   yhat_va, average=avg, pos_label=pos_label, zero_division=0)

        # AUC con scores (no depende del umbral)
        train_roc_auc = roc_auc_score(Y_train, score_tr)
        test_roc_auc  = roc_auc_score(Y_test,  score_te)
        val_roc_auc   = roc_auc_score(Y_val,   score_va)

    else:
        train_precision = precision_score(Y_train, yhat_tr, average='macro', zero_division=0)
        test_precision  = precision_score(Y_test,  yhat_te, average='macro', zero_division=0)
        val_precision   = precision_score(Y_val,   yhat_va, average='macro', zero_division=0)

        train_recall = recall_score(Y_train, yhat_tr, average='macro', zero_division=0)
        test_recall  = recall_score(Y_test,  yhat_te, average='macro', zero_division=0)
        val_recall   = recall_score(Y_val,   yhat_va, average='macro', zero_division=0)

        # AUC macro OvR con scores (si proba disponible)
        if (isinstance(score_tr, np.ndarray) and score_tr.ndim == 2):
            train_roc_auc = roc_auc_score(Y_train, score_tr, multi_class='ovr', average='macro')
            test_roc_auc  = roc_auc_score(Y_test,  score_te, multi_class='ovr', average='macro')
            val_roc_auc   = roc_auc_score(Y_val,   score_va, multi_class='ovr', average='macro')
        else:
            warnings.warn("No pude calcular AUC macro OvR de forma robusta; devolveré NaN.")
            train_roc_auc = test_roc_auc = val_roc_auc = np.nan

    metrics_df = pd.DataFrame({
        'Train': [train_accuracy, train_precision, train_recall, train_roc_auc],
        'Test':  [test_accuracy,  test_precision,  test_recall,  test_roc_auc],
        'Val':   [val_accuracy,   val_precision,   val_recall,   val_roc_auc]
    }, index=["Accuracy (thr*)", "Precision (thr*)", "Recall (thr*)", "ROC AUC"])

    # *thr: métricas calculadas con el/los umbral(es) seleccionados en validación.
    return metrics_df, roc_corner_df


def lr_coef_importance(model, X, *, estandarizar=True, multinomial_reference="last"):
    """
    Importancia y significancia (Wald) para modelos LogisticRegression de sklearn.
    - Binaria: usa Fisher observado (X' W X)^-1 con W = p*(1-p).
    - Multinomial: usa información observada sum_i ( (x_i x_i^T) ⊗ (Diag(p_i) - p_i p_i^T) ),
      y luego reparametriza como (beta_k - beta_ref) para k != ref, para tener OR y p-values
      interpretables "clase k vs clase de referencia".

    Parámetros
    ----------
    model : sklearn.linear_model.LogisticRegression ya entrenado.
    X : pd.DataFrame o np.ndarray con las MISMAS transformaciones usadas al entrenar.
    estandarizar : bool, si True añade coef_std y or_per_sd (interpretabilidad).
    multinomial_reference : {"last", "first", int} clase de referencia para multinomial.

    Returns
    -------
    df : pd.DataFrame
        Binaria: columnas [feature, coef, odds_ratio, se, z, p>|z|, (opc) coef_std, abs_coef_std, or_per_sd]
        Multinomial: índice de filas (class, feature) y mismas columnas por clase vs referencia.
    """
    # --- preparar X y nombres ---
    if hasattr(X, "to_numpy"):
        Xn = X.to_numpy()
        feat_names = list(X.columns)
    else:
        Xn = np.asarray(X)
        feat_names = [f"x{i}" for i in range(Xn.shape[1])]

    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        coef = coef[None, :]
    intercept = np.asarray(model.intercept_).ravel()
    n_classes, p = coef.shape

    # Matriz de diseño con intercepto
    X1 = np.column_stack([np.ones(len(Xn)), Xn])   # (n, p+1)
    p1 = p + 1

    # --------- CASO BINARIO ----------
    if n_classes == 1:
        beta = np.r_[intercept[0], coef[0]]  # (p+1,)
        eta = X1 @ beta
        p_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = p_hat * (1.0 - p_hat)                    # (n,)
        Xw = X1 * w[:, None]
        XtWX = X1.T @ Xw
        try:
            cov = inv(XtWX)
        except LinAlgError:
            cov = np.linalg.pinv(XtWX)

        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        z = beta / se
        pval = 2.0 * (1.0 - norm.cdf(np.abs(z)))

        rows = {
            "feature": feat_names,
            "coef": beta[1:],
            "odds_ratio": np.exp(beta[1:]),
            "se": se[1:],
            "z": z[1:],
            "p>|z|": pval[1:],
        }

        if estandarizar:
            scale = Xn.std(axis=0, ddof=0)
            scale[scale == 0] = 1.0
            coef_std = coef[0] * scale
            rows.update({
                "coef_std": coef_std,
                "abs_coef_std": np.abs(coef_std),
                "or_per_sd": np.exp(coef[0] * scale),
            })

        df = pd.DataFrame(rows).sort_values(
            "abs_coef_std" if estandarizar else "z", ascending=False
        ).reset_index(drop=True)

        # atributos útiles
        df.attrs["intercept"] = {
            "coef": float(beta[0]),
            "odds_ratio": float(np.exp(beta[0])),
            "se": float(se[0]),
            "z": float(z[0]),
            "p>|z|": float(pval[0]),
            "name": "const",
        }
        df.attrs["cov_beta"] = cov
        df.attrs["beta_full"] = beta
        df.attrs["feature_order"] = ["const"] + feat_names
        return df

    # --------- CASO MULTINOMIAL (softmax) ----------
    # Parametrización completa (K clases) y luego reparametrizamos contra referencia
    K = n_classes

    # Construir matriz de coeficientes por clase con intercepto incluido
    B = np.column_stack([intercept.reshape(-1, 1), coef])   # (K, p+1)
    beta_full = B.reshape(-1)                               # (K*p1, )

    # Probabilidades softmax por observación
    # scores = X1 @ B.T -> (n, K)
    scores = X1 @ B.T
    scores_clip = np.clip(scores, -30, 30)                  # estabilidad numérica
    es = np.exp(scores_clip)
    P = es / es.sum(axis=1, keepdims=True)                  # (n, K)

    # Información observada: sum_i ( (x_i x_i^T) ⊗ (Diag(p_i) - p_i p_i^T) )
    # Dimensión: (K*p1, K*p1)
    I = np.zeros((K * p1, K * p1), dtype=float)
    # bucle sobre observaciones (suficientemente rápido para n~1e3, p~1e2, K pequeño)
    for i in range(P.shape[0]):
        pi = P[i]                              # (K,)
        Vi = np.diag(pi) - np.outer(pi, pi)   # (K, K)
        xi = X1[i:i+1, :]                     # (1, p1)
        G = xi.T @ xi                          # (p1, p1)
        # Kronecker: Vi ⊗ G
        I += np.kron(Vi, G)

    # Covarianza (pseudo-inversa por posible no-identificabilidad)
    try:
        cov_full = inv(I)
    except LinAlgError:
        cov_full = np.linalg.pinv(I)

    # Elegir clase de referencia y construir transformación T tal que theta = T @ beta_full,
    # con theta conteniendo (K-1) bloques (p1,) de (beta_k - beta_ref).
    if multinomial_reference == "last":
        ref = K - 1
    elif multinomial_reference == "first":
        ref = 0
    else:
        ref = int(multinomial_reference)
        if not (0 <= ref < K):
            raise ValueError("multinomial_reference debe ser 'last', 'first' o un índice de clase válido.")

    # Construir T
    blocks = []
    class_idx = []
    I_p1 = np.eye(p1)
    for k in range(K):
        if k == ref:
            continue
        row = np.zeros((p1, K * p1))
        # bloque +I en columnas del k
        row[:, k * p1:(k + 1) * p1] = I_p1
        # bloque -I en columnas del ref
        row[:, ref * p1:(ref + 1) * p1] = -I_p1
        blocks.append(row)
        class_idx.append(k)
    T = np.vstack(blocks)                               # ((K-1)*p1, K*p1)

    theta = T @ beta_full                               # ((K-1)*p1, )
    cov_theta = T @ cov_full @ T.T                      # ((K-1)*p1, (K-1)*p1)

    se_theta = np.sqrt(np.clip(np.diag(cov_theta), 0, np.inf))
    z_theta = theta / se_theta
    p_theta = 2.0 * (1.0 - norm.cdf(np.abs(z_theta)))

    # Reacomodar a tabla por clase vs referencia y por feature (omitiendo intercepto en filas)
    records = []
    # std para interpretación por 1 DE (misma para todas las clases)
    if estandarizar:
        scale = Xn.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0

    for j, k in enumerate(class_idx):  # j: bloque en theta, k: clase
        offset = j * p1
        # intercepto
        inter_coef = theta[offset + 0]
        inter_se   = se_theta[offset + 0]
        inter_z    = z_theta[offset + 0]
        inter_p    = p_theta[offset + 0]

        # características (1..p)
        coefs = theta[offset + 1: offset + p1]
        ses   = se_theta[offset + 1: offset + p1]
        zs    = z_theta[offset + 1: offset + p1]
        ps    = p_theta[offset + 1: offset + p1]

        for idx_f, fname in enumerate(feat_names):
            row = {
                "class": model.classes_[k],
                "ref_class": model.classes_[ref],
                "feature": fname,
                "coef": coefs[idx_f],
                "odds_ratio": np.exp(coefs[idx_f]),   # OR clase k vs ref por +1 en feature
                "se": ses[idx_f],
                "z": zs[idx_f],
                "p>|z|": ps[idx_f],
            }
            if estandarizar:
                cstd = coefs[idx_f] * scale[idx_f]
                row.update({
                    "coef_std": cstd,
                    "abs_coef_std": abs(cstd),
                    "or_per_sd": np.exp(cstd),
                })
            records.append(row)

    df = pd.DataFrame(records)
    sort_col = "abs_coef_std" if estandarizar else "z"
    df = df.sort_values(["class", sort_col], ascending=[True, False]).reset_index(drop=True)

    # atributos útiles
    # (guardamos también los parámetros e IC del intercepto por clase vs ref)
    intercepts = []
    for j, k in enumerate(class_idx):
        o = j * p1
        intercepts.append({
            "class": model.classes_[k],
            "ref_class": model.classes_[ref],
            "name": "const",
            "coef": float(theta[o+0]),
            "odds_ratio": float(np.exp(theta[o+0])),
            "se": float(se_theta[o+0]),
            "z": float(z_theta[o+0]),
            "p>|z|": float(p_theta[o+0]),
        })
    df.attrs["intercepts"] = intercepts
    df.attrs["cov_theta"] = cov_theta
    df.attrs["beta_rel"] = theta
    df.attrs["feature_order"] = ["const"] + feat_names
    df.attrs["classes"] = list(model.classes_)
    df.attrs["reference_class"] = model.classes_[ref]

    return df

def best_threshold_roc_corner(y_true, y_prob):
    """
    y_prob: probas de la clase positiva (shape [n_samples]).
    Devuelve: threshold, fpr[idx], tpr[idx]
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    idx = np.argmin(dist)
    return float(thr[idx]), float(fpr[idx]), float(tpr[idx])


def best_thresholds_roc_corner_ovr(y_true, prob_matrix, classes):
    """
    y_true: etiquetas reales (array-like)
    prob_matrix: predict_proba(X) -> shape [n_samples, n_classes] en el orden 'classes'
    classes: arreglo con el orden de clases del modelo
    Devuelve: thresholds (np.array de shape [n_classes])
    """
    thresholds = np.zeros(len(classes), dtype=float)
    for j, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        p = prob_matrix[:, j]
        fpr, tpr, thr = roc_curve(y_bin, p)
        dist = np.sqrt(fpr**2 + (1 - tpr)**2)
        k = np.argmin(dist)
        thresholds[j] = thr[k] if len(thr) else 0.5
    return thresholds

def best_thresholds_roc_corner_ovr(y_true, prob_matrix, classes):
    """Umbral por clase = punto (FPR,TPR) más cercano a (0,1)."""
    thresholds = np.zeros(len(classes), dtype=float)
    for j, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        fpr, tpr, thr = roc_curve(y_bin, prob_matrix[:, j])
        dist = np.sqrt(fpr**2 + (1 - tpr)**2)
        k = np.argmin(dist)
        thresholds[j] = thr[k] if len(thr) else 0.5
    return thresholds

def predict_with_thresholds_ovr(estimator, X, classes, thresholds):
    """Si alguna clase supera su umbral, elige la de mayor (p_j - τ_j); si ninguna, argmax."""
    P = estimator.predict_proba(X)             # shape (n, C)
    margins = P - thresholds[None, :]
    has_any = (P >= thresholds[None, :]).any(axis=1)
    pred = np.argmax(P, axis=1)                # fallback
    pred_thr = np.argmax(margins, axis=1)
    pred[has_any] = pred_thr[has_any]
    return classes[pred]

def confusion_matrix_mult(model, Y_train, X_test, Y_test, X_val, Y_val):
    model 
    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model, "named_steps") and "svc" in model.named_steps:
        classes = model.named_steps["svc"].classes_
    else:
        classes = np.unique(Y_train)

    # 2) Umbrales desde VALIDACIÓN (esquina buena)
    P_test = model.predict_proba(X_val)
    thr_ovr = best_thresholds_roc_corner_ovr(Y_val, P_test, classes)

    # (Opcional) favorecer minoritarias: baja su umbral un poco
    for name, delta in {"bajo": -0.03, "sobrepeso": -0.03, "obesidad": -0.03}.items():
        if name in classes:
            j = np.where(classes == name)[0][0]
            thr_ovr[j] = np.clip(thr_ovr[j] + delta, 0.0, 1.0)

    # 3) Predicciones en TEST usando esos umbrales
    y_pred_thr = predict_with_thresholds_ovr(model, X_test, classes, thr_ovr)

    # 4) Matriz de confusión (con el mismo orden de clases)
    cm = confusion_matrix(Y_test, y_pred_thr, labels=classes)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(values_format='d')
    plt.title("Matriz de confusión (umbral ROC-corner)")
    plt.show()

def confusion_matrix_binary(model, X_test, Y_test, X_val, Y_val):
    probs_val = model.predict_proba(X_val)[:, 1]  # clase positiva en la col 1 (ajusta si no)
    fpr, tpr, thr = roc_curve(Y_val, probs_val)
    dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    thr_best = thr[np.argmin(dist)]

    # 2) Predicción en TEST con ese umbral
    probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_thr = (probs_test >= thr_best).astype(int)

    # 3) Matriz de confusión
    cm = confusion_matrix(Y_test, y_pred_thr)
    ConfusionMatrixDisplay(cm).plot(values_format='d')
    plt.title("Matriz de confusión (umbral ROC-corner)")
    plt.show()


def _stack_y(Y_train=None, Y_test=None, Y_val=None):
    parts = [y for y in (Y_train, Y_test, Y_val) if y is not None]
    if len(parts) == 0:
        raise ValueError("Se requiere al menos un vector Y para construir Y_all.")
    return np.concatenate([np.asarray(y).ravel() for y in parts], axis=0)

def _call_metrics(metrics_fn, model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_all):
    """
    Soporta:
      - return metrics_df
      - return (metrics_df, roc_corner_df)
    """
    out = metrics_fn(model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_all)
    if isinstance(out, tuple) and len(out) == 2:
        metrics_df, roc_corner_df = out
    else:
        metrics_df, roc_corner_df = out, None
    return metrics_df, roc_corner_df

def compare_models_multi_data(model_specs, *, metrics_fn, prefer_split="Test", float_fmt="{:.3f}"):
    """
    model_specs: lista de dicts:
      {
        "name": "MyModel",
        "model": fitted_model,
        "X_train": Xtr, "X_test": Xte, "X_val": Xva (opcional),
        "Y_train": ytr, "Y_test": yte, "Y_val": yva (opcional)
      }
    metrics_fn: debe aceptar firma (model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_all)
                y devolver metrics_df o (metrics_df, roc_corner_df).

    Devuelve:
      panel_df      : DataFrame (multi-col) con todas las métricas por modelo y split
      slice_df      : DataFrame simple (split preferido)
      roc_table     : DataFrame con umbrales/ROC-corner por modelo (si lo provee metrics_fn)
      styled_panel  : Styler bonito del panel (degradado y mejores valores)
      styled_slice  : Styler bonito del split preferido
    """
    results = {}
    roc_info = {}   # thresholds / ROC corner por modelo
    all_metric_names = set()
    all_splits = set()

    # --- calcular métricas por modelo ---
    for spec in model_specs:
        name  = spec["name"]
        model = spec["model"]

        X_train = spec["X_train"]
        X_test  = spec["X_test"]
        X_val   = spec.get("X_val", None)

        Y_train = spec["Y_train"]
        Y_test  = spec["Y_test"]
        Y_val   = spec.get("Y_val", None)

        Y_all = _stack_y(Y_train, Y_test, Y_val)

        mdf, rocdf = _call_metrics(metrics_fn, model, X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_all)

        results[name] = mdf
        if rocdf is not None:
            # añade columna para saber a qué modelo pertenece
            r = rocdf.copy()
            r.insert(0, "model", name)
            roc_info[name] = r

        all_metric_names.update(mdf.index.tolist())
        all_splits.update(mdf.columns.tolist())

    # --- alinear y concatenar ---
    all_metric_names = list(all_metric_names)
    all_splits = list(all_splits)

    aligned = {}
    for name, mdf in results.items():
        aligned[name] = mdf.reindex(index=all_metric_names, columns=all_splits)

    panel_df = pd.concat(aligned, axis=1)  # columnas de 2 niveles: (modelo, split)

    # --- slice del split preferido ---
    slice_df = None
    if prefer_split in all_splits:
        slice_df = panel_df.xs(prefer_split, axis=1, level=1)

    # --- tabla de umbrales/ROC-corner por modelo (si existe) ---
    roc_table = None
    if len(roc_info) > 0:
        roc_table = pd.concat(roc_info.values(), axis=0, ignore_index=True)

    # --- estilo bonito: degradado + destacar máximos por fila ---
    def _style_panel(df, *, group=3, border_color="red", border_px=3):
        """
        Añade un borde izquierdo grueso cada 'group' columnas,
        compatible con columnas MultiIndex.
        """
        sty = (df.style
            .format(float_fmt)
            .background_gradient(axis=None, cmap="YlGnBu")
            .set_table_styles([
                {"selector": "th", "props": "background-color:#222; color:white;"},
                {"selector": "caption", "props": "caption-side: top; font-size:14px; font-weight:bold;"},
                # Sugerencia visual en header (puede variar con MultiIndex/colspan):
                {"selector": "thead th", "props": f"border-left: 0;"},
            ]))

        # --- Borde izquierdo grueso cada 'group' columnas en el cuerpo ---
        # Tomamos las columnas por posición (cada bloque de tamaño 'group')
        start_cols = [df.columns[i] for i in range(0, df.shape[1], group)]
        # Aplica borde a esas columnas
        sty = sty.set_properties(
            subset=pd.IndexSlice[:, start_cols],
            **{ "border-left": f"{border_px}px solid {border_color}" }
        )

        # Opcional: un borde superior en la primera fila de datos para reforzar la grilla
        sty = sty.set_table_styles(sty.table_styles + [
            {"selector": "tbody tr:first-child td",
            "props": f"border-top: {max(1, border_px-1)}px solid {border_color};"}
        ])

        # --- Tu resaltado de máximos por split se mantiene ---
        if isinstance(df.columns, pd.MultiIndex):
            for split in df.columns.levels[1]:
                if split in df.columns.get_level_values(1):
                    def _highlight_max(s):
                        is_max = s == s.max()
                        return ['font-weight: bold' if v else '' for v in is_max]
                    sty = sty.apply(_highlight_max, subset=pd.IndexSlice[:, pd.IndexSlice[:, split]], axis=1)
        else:
            sty = sty.highlight_max(axis=1)

        return sty


    def _style_slice(df):
        if df is None:
            return None
        sty = (df.style
               .format(float_fmt)
               .background_gradient(axis=None, cmap="YlOrRd")
               .set_table_styles([
                   {"selector":"th", "props":"background-color:#222; color:white;"},
                   {"selector":"caption", "props":"caption-side: top; font-size:14px; font-weight:bold;"},
               ])
               .highlight_max(axis=1))
        return sty

    styled_panel = _style_panel(panel_df)
    styled_slice = _style_slice(slice_df)

    return panel_df, slice_df, roc_table, styled_panel, styled_slice

from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, KMeansSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline

def _auto_k_neighbors(y):
    c = Counter(pd.Series(y).to_numpy())
    m = min(c.values())
    return max(1, min(3, m-1))  # conservador

def _auto_m_neighbors(y):
    c = Counter(pd.Series(y).to_numpy())
    m = min(c.values())
    return max(3, min(10, m + 1))

def _cat_indices(X: pd.DataFrame):
    if isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
        return [X.columns.get_loc(c) for c in cat_cols], list(cat_cols)
    return [], []

def build_strategy_to_majority(y, ratio=0.6, max_per_class=None):
    counts = Counter(pd.Series(y).to_numpy())
    M = max(counts.values())
    target = int(np.ceil(M * ratio))
    strategy = {}
    for cls, cnt in counts.items():
        if cnt < target:
            n_obj = min(target, max_per_class) if max_per_class is not None else target
            strategy[cls] = n_obj
    return strategy

def _make_sampler_base(X, y, sampling_strategy, sampler, k_neighbors, m_neighbors, random_state):
    k = _auto_k_neighbors(y) if k_neighbors is None else k_neighbors
    m = _auto_m_neighbors(y) if m_neighbors is None else m_neighbors
    cat_idx, _ = _cat_indices(X)

    # Si hay categóricas -> usa SMOTENC (no hay Borderline/KMeans para categóricas)
    if len(cat_idx) > 0:
        return SMOTENC(categorical_features=cat_idx,
                       sampling_strategy=sampling_strategy,
                       k_neighbors=k,
                       random_state=random_state)

    if sampler == "borderline1":
        return BorderlineSMOTE(kind='borderline-1',
                               sampling_strategy=sampling_strategy,
                               k_neighbors=k, m_neighbors=m,
                               random_state=random_state)
    if sampler == "borderline2":
        return BorderlineSMOTE(kind='borderline-2',
                               sampling_strategy=sampling_strategy,
                               k_neighbors=k, m_neighbors=m,
                               random_state=random_state)
    if sampler == "kmeans":
        return KMeansSMOTE(sampling_strategy=sampling_strategy,
                           k_neighbors=k,
                           kmeans_estimator=5,
                           cluster_balance_threshold=0.05,
                           density_exponent='auto',
                           random_state=random_state)
    # default: SMOTE clásico
    return SMOTE(sampling_strategy=sampling_strategy,
                 k_neighbors=k,
                 random_state=random_state)

def smote_fit_resample(X_train, y_train, *,
                       ratio=0.6, max_per_class=None,
                       sampler="smote", clean="none",
                       k_neighbors=None, m_neighbors=None,
                       random_state=42, verbose=True):
    """
    Oversampling conservador + limpieza opcional.
    - sampler: 'smote' | 'borderline1' | 'borderline2' | 'kmeans'
    - clean  : 'none'  | 'tomek' | 'enn'
    """
    X_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y_arr = pd.Series(y_train).to_numpy()

    if verbose:
        print("Distribución original:", Counter(y_arr))

    strategy = build_strategy_to_majority(y_arr, ratio=ratio, max_per_class=max_per_class)
    if len(strategy) == 0:
        if verbose:
            print("No se aplicó oversampling (estrategia vacía).")
        return X_df.to_numpy(), y_arr

    base = _make_sampler_base(X_df, y_arr, strategy, sampler, k_neighbors, m_neighbors, random_state)

    # --- Ensamble correcto según restricción de imblearn ---
    if clean == "none":
        sampler_pipe = base
    else:
        if sampler == "smote":
            # Aquí sí podemos usar combinadores directamente
            if clean == "tomek":
                sampler_pipe = SMOTETomek(smote=base, random_state=random_state)
            else:  # 'enn'
                sampler_pipe = SMOTEENN(smote=base, random_state=random_state)
        else:
            # Borderline/KMeans NO son válidos dentro de SMOTETomek/SMOTEENN
            # -> encadena sampler + cleaner como pasos separados
            cleaner = TomekLinks() if clean == "tomek" else EditedNearestNeighbours(n_neighbors=3)
            sampler_pipe = ImbPipeline(steps=[
                ('over', base),
                ('clean', cleaner),
            ])

    X_sm, y_sm = sampler_pipe.fit_resample(X_df.to_numpy(), y_arr)

    if verbose:
        used_k = k_neighbors if k_neighbors is not None else _auto_k_neighbors(y_arr)
        print(f"sampler={sampler}, clean={clean}, k_neighbors={used_k}")
        print("Estrategia:", strategy)
        print("Distribución SMOTE:", Counter(y_sm))
    return X_sm, y_sm


# ====== TU CASO (tres categorizaciones) ======

def svc_hiper_search_mult(X_train, Y_train):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("svc", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    # 4) Pesos de clase (opcional: además de probar 'balanced', puedes pasar un dict)
    sw_bal = compute_sample_weight(class_weight="balanced", y=Y_train)
    alpha = 1.4  # prueba 1.2–1.8
    sw_aggr = sw_bal ** alpha

    # 5) Espacio de búsqueda controlado (evita extremos)
    param_dist = {
        "svc__C":           loguniform(1e-2, 1e3),   # [0.01, 1000]
        "svc__gamma":       loguniform(1e-4, 1e1),   # [1e-4, 10]
        "svc__class_weight": [None, "balanced"],     # veremos sample_weight abajo
        "svc__tol":         loguniform(1e-5, 1e-2),
        "svc__shrinking":   [True, False],           # a veces ayuda
        "svc__cache_size":  [200, 500, 1000],        # rendimiento (no afecta métrica)
    }

    scoring = {
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "balanced_acc": "balanced_accuracy",
        "auc_ovr_w": "roc_auc_ovr_weighted",
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=100,
        scoring=scoring,
        refit="recall_macro",      # objetivo principal para refit
        cv=skf,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # (Opcional) sample_weight adicional (a veces ayuda aún más que class_weight)

    search.fit(X_train, Y_train, svc__sample_weight=sw_aggr)

    return search

def svc_hiper_search_binary(X_train, Y_train):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("svc", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    param_dist = {
        "svc__C":     loguniform(1e-1, 1e2),  # ~ [0.1, 100]
        "svc__gamma": loguniform(1e-3, 1e0),  # ~ [1e-3, 1]
        "svc__class_weight": [None, "balanced"],     # veremos sample_weight abajo
        "svc__tol":   loguniform(1e-5, 1e-2),
    }

    scoring={"f1": "f1", "roc_auc": "roc_auc"}

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=100,
        scoring=scoring,
        refit="f1",      
        cv=skf,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, Y_train)

    return search

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted


@dataclass
class _ClusterModel:
    pure: bool
    label_idx: Optional[int] = None           # si pure=True, índice de clase fija
    clf: Optional[BaseEstimator] = None       # si pure=False, clasificador entrenado


class ClusterThenClassify(BaseEstimator, ClassifierMixin):
    """
    Cluster-then-classify:
      - Clustering global (KMeans|GMM)
      - Por cluster: si es 100% puro -> asignación directa; si no -> entrena clf.
    """
    def __init__(self,
                 clusterer: str = "kmeans",        # 'kmeans' | 'gmm'
                 k_range: Tuple[int, int] = (2, 8),
                 base_estimator: Optional[BaseEstimator] = None,
                 random_state: int = 42):
        self.clusterer = clusterer
        self.k_range = k_range
        self.base_estimator = base_estimator
        self.random_state = random_state

    def _make_clusterer(self, n_clusters):
        if self.clusterer == "kmeans":
            return KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        elif self.clusterer == "gmm":
            return GaussianMixture(n_components=n_clusters, covariance_type="full",
                                   random_state=self.random_state)
        raise ValueError("clusterer debe ser 'kmeans' o 'gmm'")

    def _fit_clusterer(self, X, best_k):
        if self.clusterer == "kmeans":
            self.clusterer_ = self._make_clusterer(best_k).fit(X)
        else:  # gmm
            self.clusterer_ = self._make_clusterer(best_k).fit(X)
        return self

    def _predict_clusters(self, X):
        if isinstance(self.clusterer_, KMeans):
            return self.clusterer_.predict(X)
        else:  # GMM
            return self.clusterer_.predict(X)

    def _default_clf(self, n_classes):
        # SVC costeado y con probas par; sólido para clusters “más fáciles”
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return SVC(kernel="rbf", probability=True, class_weight="balanced",
                   C=1.0, gamma="scale", random_state=self.random_state)

    def fit(self, X, y):
        X = np.asarray(X)
        self.le_ = LabelEncoder().fit(y)
        y_idx = self.le_.transform(y)
        self.classes_ = self.le_.classes_
        self.n_classes_ = len(self.classes_)

        # ---- elegir K por silhouette ----
        k_min, k_max = self.k_range
        k_candidates = [k for k in range(k_min, k_max + 1) if k >= 2]
        best_k, best_score = None, -np.inf
        for k in k_candidates:
            clus = self._make_clusterer(k)
            labels = clus.fit_predict(X) if self.clusterer == "kmeans" else clus.fit(X).predict(X)
            # silhouette solo tiene sentido si hay >1 cluster y cada cluster tiene >=2 puntos
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
            # fallback: 2 clusters
            best_k = 2

        # ---- reentrenar clusterer con best_k ----
        self._fit_clusterer(X, best_k)
        z = self._predict_clusters(X)
        self.n_clusters_ = best_k

        # ---- construir modelos por cluster ----
        self.cluster_models_: Dict[int, _ClusterModel] = {}
        for c in range(best_k):
            idx = np.where(z == c)[0]
            Xc, yc = X[idx], y_idx[idx]
            counts = Counter(yc)
            if len(counts) == 1:
                # 100% puro
                label_idx = next(iter(counts.keys()))
                self.cluster_models_[c] = _ClusterModel(pure=True, label_idx=label_idx, clf=None)
            else:
                clf = self._default_clf(self.n_classes_)
                clf.fit(Xc, yc)
                self.cluster_models_[c] = _ClusterModel(pure=False, label_idx=None, clf=clf)

        return self

    def predict(self, X):
        check_is_fitted(self, ["clusterer_", "cluster_models_", "le_"])
        X = np.asarray(X)
        z = self._predict_clusters(X)
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
        X = np.asarray(X)
        z = self._predict_clusters(X)
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
                # asegurar orden de columnas 0..n_classes-1
                if p.shape[1] != self.n_classes_:
                    # algunos clasificadores pueden no conocer todas las clases en ese sub-set:
                    # reconstruimos matriz completa
                    p_full = np.zeros((p.shape[0], self.n_classes_))
                    for j, cls_idx in enumerate(cm.clf.classes_):
                        p_full[:, cls_idx] = p[:, j]
                    p = p_full
                proba[mask] = p
        return proba


# -------- función simple de entrenamiento (firma corta solicitada) --------
def train_cluster_then_classify(X, y,
                                clusterer: str = "kmeans",
                                k_range: Tuple[int, int] = (2, 8),
                                base_estimator: Optional[BaseEstimator] = None,
                                random_state: int = 42):
    """
    Entrena y devuelve el modelo final ClusterThenClassify.
    X, y: datos ya PREPROCESADOS/ESCALADOS.
    """
    ctc = ClusterThenClassify(clusterer=clusterer,
                              k_range=k_range,
                              base_estimator=base_estimator,
                              random_state=random_state)
    return ctc.fit(X, y)
