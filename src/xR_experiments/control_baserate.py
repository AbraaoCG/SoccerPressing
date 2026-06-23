# -*- coding: utf-8 -*-
"""control_baserate.py — a AUC maior do y_5s_5m vem da base menor, ou e sinal real?
(1) CONTROLE DE BASE: reduz os positivos do y_10s ate a base bater ~6,9% (= base do y_5s_5m),
    retreina a logistica e mede a AUC. Se continuar ~0,59, a base NAO e a causa.
(2) IC BOOTSTRAP (pareado) do gap AUC(y_5s_5m) - AUC(y_10s) no mesmo teste.
Logistica (metodologia original). Saida: star_analysis/control_baserate.csv.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import express_data as xd

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)


def fit_logit_auc(X, y, seed=42):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y[te], p), te, p, y[te]


def main():
    df = pd.read_pickle('cache/_star_all.pkl')
    X = np.nan_to_num(df[xd.TAB_FEATURES].to_numpy(np.float32))
    dist = np.hypot(df['loss_x'] - df['x'], df['loss_y'] - df['y'])
    y10 = (df['ev_loss'] == 1).astype(int).to_numpy()
    y55 = ((df['ev_loss'] == 1) & (df['t_loss'] <= 5.0) & (dist <= 5.0)).astype(int).to_numpy()
    b10, b55 = y10.mean(), y55.mean()
    print('bases: y_10s=%.4f | y_5s_5m=%.4f' % (b10, b55))

    # AUC plenas (referencia), no MESMO split p/ o bootstrap pareado
    auc10, te, p10, yte10 = fit_logit_auc(X, y10)
    auc55, te2, p55, yte55 = fit_logit_auc(X, y55)
    print('AUC plena: y_10s=%.4f | y_5s_5m=%.4f' % (auc10, auc55))

    # ---- (1) CONTROLE DE BASE: y_10s reduzido a base ~6,9% ----
    neg = np.where(y10 == 0)[0]; pos = np.where(y10 == 1)[0]
    n_pos_keep = int(round(b55 * len(neg) / (1 - b55)))   # base alvo = b55
    aucs_ctrl = []
    for s in range(25):
        r = np.random.default_rng(100 + s)
        keep = np.concatenate([neg, r.choice(pos, size=n_pos_keep, replace=False)])
        a, _, _, _ = fit_logit_auc(X[keep], y10[keep], seed=100 + s)
        aucs_ctrl.append(a)
    aucs_ctrl = np.array(aucs_ctrl)
    print('CONTROLE y_10s @ base %.3f: AUC %.4f +/- %.4f (25 reps) | base efetiva %.4f'
          % (b55, aucs_ctrl.mean(), aucs_ctrl.std(), n_pos_keep / (n_pos_keep + len(neg))))

    # ---- (2) IC BOOTSTRAP PAREADO do gap (mesmo conjunto de teste) ----
    # usa o teste do modelo y_10s (te) e reavalia y_5s_5m no MESMO te
    m55 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    idx = np.arange(len(y55)); tr_all = np.setdiff1d(idx, te)
    m55.fit(X[tr_all], y55[tr_all]); p55_te = m55.predict_proba(X[te])[:, 1]
    y55_te = y55[te]; y10_te = y10[te]
    gaps = []
    B = 1000; n = len(te)
    for _ in range(B):
        bi = rng.integers(0, n, n)
        if y10_te[bi].sum() == 0 or y55_te[bi].sum() == 0:
            continue
        g = roc_auc_score(y55_te[bi], p55_te[bi]) - roc_auc_score(y10_te[bi], p10[bi])
        gaps.append(g)
    gaps = np.array(gaps)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    gap_pt = roc_auc_score(y55_te, p55_te) - roc_auc_score(y10_te, p10)
    print('GAP AUC(y_5s_5m - y_10s) no mesmo teste: %.4f | IC95%% bootstrap [%.4f, %.4f]'
          % (gap_pt, lo, hi))

    pd.DataFrame([
        dict(item='AUC y_10s (base %.3f)' % b10, valor=round(auc10, 4)),
        dict(item='AUC y_5s_5m (base %.3f)' % b55, valor=round(auc55, 4)),
        dict(item='AUC y_10s CONTROLE @ base %.3f (media 25 reps)' % b55, valor=round(float(aucs_ctrl.mean()), 4)),
        dict(item='AUC y_10s CONTROLE desvio', valor=round(float(aucs_ctrl.std()), 4)),
        dict(item='GAP AUC (y_5s_5m - y_10s)', valor=round(float(gap_pt), 4)),
        dict(item='GAP IC95 inferior', valor=round(float(lo), 4)),
        dict(item='GAP IC95 superior', valor=round(float(hi), 4)),
    ]).to_csv(os.path.join(OUT, 'control_baserate.csv'), index=False)
    print('\nsalvo em', os.path.join(OUT, 'control_baserate.csv'))


if __name__ == '__main__':
    main()
