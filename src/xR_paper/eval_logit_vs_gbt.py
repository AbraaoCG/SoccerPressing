# -*- coding: utf-8 -*-
"""eval_logit_vs_gbt.py — logistica vs GBT no alvo y_5s_5m: a logistica e tao confiavel quanto o GBT
para uso como PROBABILIDADE / valor esperado (agregado)? Compara AUC, PR-AUC, Brier, ECE (calibracao)
e lift/precisao no top-10%. Saida: star_analysis/logit_vs_gbt.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'xR_experiments'))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
TARGET = 'y_5s_5m'


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def topk(y, p, frac=0.10):
    k = max(1, int(len(p) * frac)); idx = np.argsort(-p)[:k]
    return y[idx].mean(), y[idx].sum() / y.sum()


def metrics(name, yt, p, base):
    p10, r10 = topk(yt, p)
    return dict(modelo=name, AUC=round(roc_auc_score(yt, p), 3),
               PR_AUC=round(average_precision_score(yt, p), 3),
               Brier=round(brier_score_loss(yt, p), 4), ECE=round(ece(yt, p), 4),
               prec_top10=round(p10, 3), lift_top10=round(p10 / base, 2), rec_top10=round(r10, 3))


def main():
    meta = xd.load_meta()
    X = meta[xd.TAB_FEATURES].to_numpy(np.float32)
    y = meta[TARGET].to_numpy(np.int32)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    base = y[te].mean()

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
    p_lr = lr.predict_proba(X[te])[:, 1]
    gbt, kind = make_gbt(); fit_gbt(gbt, kind, X[tr2], y[tr2], X[va], y[va])
    p_gbt = proba(gbt, X[te])

    rows = [metrics('logistica', y[te], p_lr, base), metrics('GBT-%s' % kind, y[te], p_gbt, base)]
    df = pd.DataFrame(rows)
    df.insert(1, 'base_rate', round(float(base), 3))
    df.to_csv(os.path.join(OUT, 'logit_vs_gbt.csv'), index=False)
    print('alvo %s | base rate %.3f | teste n=%d\n' % (TARGET, base, len(te)))
    print(df.to_string(index=False))
    print('\nsalvo em', os.path.join(OUT, 'logit_vs_gbt.csv'))


if __name__ == '__main__':
    main()
