# -*- coding: utf-8 -*-
"""Experimento 3 — Split cross-competition (anti-leakage).
Modelo (GBT) e alvo (y_10s) fixos; muda SO o protocolo de avaliacao:
  (a) held-out de UMA competicao inteira (default comp 43 = FIFA WC 2022, igual ao artigo);
  (b) GroupKFold por match_id (5 folds).
Compara com o split aleatorio (que vaza eventos da mesma partida -> AUC inflada).
Saida: express_analysis/exp3.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba, auc_acc

OUT = 'express_analysis'; os.makedirs(OUT, exist_ok=True)
TARGET = 'y_10s'
HELDOUT_COMP = 43        # FIFA World Cup 2022 (mesmo conjunto de teste do artigo exPress)


def main():
    meta = xd.load_meta()
    X = meta[xd.TAB_FEATURES].to_numpy(np.float32)
    y = meta[TARGET].to_numpy(np.int32)
    comp = meta['competition_id'].to_numpy()
    grp = meta['match_id'].to_numpy()
    rows = []

    # (0) baseline split aleatorio (referencia do gap)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    model, kind = make_gbt(); fit_gbt(model, kind, X[tr2], y[tr2], X[va], y[va])
    auc_rand, acc_rand = auc_acc(y[te], proba(model, X[te]))
    print('split aleatorio    : AUC %.4f | acc %.4f' % (auc_rand, acc_rand))
    rows.append(dict(exp='exp3', protocolo='aleatorio', teste='random30%',
                     auc=round(auc_rand, 4), acc=round(acc_rand, 4), n_test=len(te)))

    # (a) held-out competicao inteira
    te_m = comp == HELDOUT_COMP
    tr_all = np.where(~te_m)[0]; te_idx = np.where(te_m)[0]
    tr2, va = train_test_split(tr_all, test_size=0.15, stratify=y[tr_all], random_state=42)
    model, kind = make_gbt(); fit_gbt(model, kind, X[tr2], y[tr2], X[va], y[va])
    auc_ho, acc_ho = auc_acc(y[te_idx], proba(model, X[te_idx]))
    nm = xd.COMP_NAMES.get(HELDOUT_COMP, str(HELDOUT_COMP))
    print('held-out %-14s: AUC %.4f | acc %.4f' % (nm, auc_ho, acc_ho))
    rows.append(dict(exp='exp3', protocolo='held-out-comp', teste=nm,
                     auc=round(auc_ho, 4), acc=round(acc_ho, 4), n_test=len(te_idx)))

    # (b) GroupKFold por match_id
    gkf = GroupKFold(n_splits=5); aucs, accs = [], []
    for tri, tei in gkf.split(X, y, groups=grp):
        tri2, vai = train_test_split(tri, test_size=0.15, stratify=y[tri], random_state=42)
        model, kind = make_gbt(); fit_gbt(model, kind, X[tri2], y[tri2], X[vai], y[vai])
        a, c = auc_acc(y[tei], proba(model, X[tei])); aucs.append(a); accs.append(c)
    print('GroupKFold(match)  : AUC %.4f +/- %.4f | acc %.4f' % (np.mean(aucs), np.std(aucs), np.mean(accs)))
    rows.append(dict(exp='exp3', protocolo='groupkfold-match', teste='5folds',
                     auc=round(float(np.mean(aucs)), 4), acc=round(float(np.mean(accs)), 4),
                     n_test=len(y)))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'exp3.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'exp3.csv'))


if __name__ == '__main__':
    main()
