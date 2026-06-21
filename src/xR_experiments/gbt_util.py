# -*- coding: utf-8 -*-
"""gbt_util.py — fabrica de modelo GBT (XGBoost, com fallback HistGradientBoosting) + treino/aval.
Compartilhado pelos experimentos tabulares (1, 2, 3, 5)."""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.ensemble import HistGradientBoostingClassifier


def make_gbt():
    """GBT com tratamento nativo de missing. XGBoost se disponivel, senao HistGradientBoosting."""
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
            eval_metric='auc', early_stopping_rounds=40, n_jobs=0, tree_method='hist'), 'xgboost'
    return HistGradientBoostingClassifier(
        max_iter=600, max_depth=5, learning_rate=0.05, l2_regularization=1.0,
        validation_fraction=0.2, early_stopping=True, random_state=42), 'histgbdt'


def fit_gbt(model, kind, Xtr, ytr, Xva, yva):
    if kind == 'xgboost':
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr)          # HistGBDT faz early stopping interno
    return model


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def auc_acc(y, p):
    return float(roc_auc_score(y, p)), float(accuracy_score(y, (p >= 0.5).astype(int)))
