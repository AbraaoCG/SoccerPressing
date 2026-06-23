# -*- coding: utf-8 -*-
"""Experimento 2 — Redefinir o alvo.
Modelo fixo (GBT) + split aleatorio; varre as definicoes de sucesso do pressing.
No artigo, "2 acoes" saltou para 0,712. Saida: express_analysis/exp2.csv (define best_target)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba, auc_acc

OUT = 'express_analysis'; os.makedirs(OUT, exist_ok=True)


def main():
    meta = xd.load_meta()
    X = meta[xd.TAB_FEATURES].to_numpy(np.float32)
    rows = []
    for tgt in xd.LABELS:
        y = meta[tgt].to_numpy(np.int32)
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
        tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
        model, kind = make_gbt()
        fit_gbt(model, kind, X[tr2], y[tr2], X[va], y[va])
        auc, acc = auc_acc(y[te], proba(model, X[te]))
        rows.append(dict(exp='exp2', modelo='GBT-%s' % kind, alvo=tgt, split='aleatorio',
                         taxa_positivos=round(float(y.mean()), 4), auc=round(auc, 4),
                         acc=round(acc, 4), n_test=len(te)))
        print('alvo %-8s | pos %.3f | AUC %.4f | acc %.4f' % (tgt, y.mean(), auc, acc))
    df = pd.DataFrame(rows).sort_values('auc', ascending=False)
    df.to_csv(os.path.join(OUT, 'exp2.csv'), index=False)
    best = df.iloc[0]['alvo']
    print('best_target =', best)
    print('salvo em', os.path.join(OUT, 'exp2.csv'))


if __name__ == '__main__':
    main()
