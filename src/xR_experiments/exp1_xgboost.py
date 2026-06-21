# -*- coding: utf-8 -*-
"""Experimento 1 — GBT/XGBoost nas features tabulares atuais.
Muda SO o modelo (arvores com boosting) vs. a logistica de contexto. Alvo y_10s, split aleatorio.
Saida: express_analysis/exp1.csv. Compara com hoje (logistica 0,541 | CNN 0,589)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba, auc_acc

OUT = 'express_analysis'; os.makedirs(OUT, exist_ok=True)
TARGET = 'y_10s'


def main():
    meta = xd.load_meta()
    X = meta[xd.TAB_FEATURES].to_numpy(np.float32)
    y = meta[TARGET].to_numpy(np.int32)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)

    model, kind = make_gbt()
    fit_gbt(model, kind, X[tr2], y[tr2], X[va], y[va])
    auc, acc = auc_acc(y[te], proba(model, X[te]))
    print('GBT (%s): AUC %.4f | acc %.4f' % (kind, auc, acc))

    # referencia interna: logistica nas MESMAS features/split (contexto)
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
    auc_lr, acc_lr = auc_acc(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1])
    print('Logistica (ref): AUC %.4f | acc %.4f' % (auc_lr, acc_lr))

    pd.DataFrame([
        dict(exp='exp1', modelo='GBT-%s' % kind, alvo=TARGET, split='aleatorio',
             auc=round(auc, 4), acc=round(acc, 4), n_test=len(te)),
        dict(exp='exp1', modelo='logistica(ref)', alvo=TARGET, split='aleatorio',
             auc=round(auc_lr, 4), acc=round(acc_lr, 4), n_test=len(te)),
    ]).to_csv(os.path.join(OUT, 'exp1.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'exp1.csv'))


if __name__ == '__main__':
    main()
