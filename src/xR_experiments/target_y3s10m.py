# -*- coding: utf-8 -*-
"""target_y3s10m.py — testa o alvo y_3s_10m (recuperar <=3 s E a <=10 m do ponto da pressao)
com a metodologia enxuta: REGRESSAO LOGISTICA (modelo original do projeto). GBT como referencia.
Usa o cache _star_all.pkl (tem tempo e local da recuperacao). Saida: star_analysis/target_y3s10m.csv.
Uso: ./venv1/Scripts/python.exe src/xR_experiments/target_y3s10m.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, brier_score_loss
import express_data as xd               # importa -> chdir raiz + TAB_FEATURES
from gbt_util import make_gbt, fit_gbt, proba

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def make_target(df, dt_max, d_max):
    dist = np.hypot(df['loss_x'] - df['x'], df['loss_y'] - df['y'])
    return ((df['ev_loss'] == 1) & (df['t_loss'] <= dt_max) & (dist <= d_max)).astype(int).to_numpy()


def eval_one(X, y, kind):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    if kind == 'logistica':
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
    else:
        tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
        m, k = make_gbt(); fit_gbt(m, k, X[tr2], y[tr2], X[va], y[va]); p = proba(m, X[te])
    yt = y[te]
    return dict(modelo=kind, base_rate=round(float(y.mean()), 4),
                AUC=round(float(roc_auc_score(yt, p)), 4),
                PR_AUC=round(float(average_precision_score(yt, p)), 4),
                Brier=round(float(brier_score_loss(yt, p)), 4), ECE=round(float(ece(yt, p)), 4),
                acc=round(float(accuracy_score(yt, (p >= 0.5).astype(int))), 4))


def main():
    df = pd.read_pickle('cache/_star_all.pkl')
    X = np.nan_to_num(df[xd.TAB_FEATURES].to_numpy(np.float32))
    rows = []
    for nome, (dt, dm) in {'y_3s_10m': (3.0, 10.0), 'y_5s_5m (ref)': (5.0, 5.0)}.items():
        y = make_target(df, dt, dm)
        for kind in ('logistica', 'GBT'):
            r = eval_one(X, y, kind); r['alvo'] = nome; rows.append(r)
            print('[%s | %s] base %.4f | AUC %.4f | PR-AUC %.4f | Brier %.4f | ECE %.4f | acc %.4f'
                  % (nome, r['modelo'], r['base_rate'], r['AUC'], r['PR_AUC'], r['Brier'], r['ECE'], r['acc']))
    res = pd.DataFrame(rows)[['alvo', 'modelo', 'base_rate', 'AUC', 'PR_AUC', 'Brier', 'ECE', 'acc']]
    res.to_csv(os.path.join(OUT, 'target_y3s10m.csv'), index=False)
    print('\n', res.to_string(index=False))
    print('\nsalvo em', os.path.join(OUT, 'target_y3s10m.csv'))


if __name__ == '__main__':
    main()
