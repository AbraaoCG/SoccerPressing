# -*- coding: utf-8 -*-
"""Experimento 6 — Ablacao: de onde vem o AUC 0,70? (geometria vs posicao vs contexto)
GBT (mesmo do exp1/2) em blocos de features, split aleatorio (comparavel ao exp2 = 0,702).
Blocos: GEOM (config. de jogadores) | POS (posicao da bola) | CTX (nao-espacial).
Conjuntos: GEOM, POS, CTX, POS+CTX (tudo MENOS geometria), ALL.
Saida: express_analysis/exp6_ablation.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba, auc_acc

OUT = 'express_analysis'; os.makedirs(OUT, exist_ok=True)
GEOM = xd.TAB_GEOM                                   # 12 features de configuracao de jogadores (360)
POS = ['x', 'y', 'dist_to_goal', 'zone_code']        # posicao da bola no campo
CTX = ['minute', 'duration']                         # contexto nao-espacial
BLOCKS = {'GEOM': GEOM, 'POS': POS, 'CTX': CTX, 'POS+CTX': POS + CTX, 'ALL': GEOM + POS + CTX}
TARGETS = ['y_5s_5m', 'y_10s']


def eval_block(meta, cols, tgt):
    X = meta[cols].to_numpy(np.float32)
    y = meta[tgt].to_numpy(np.int32)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    model, kind = make_gbt(); fit_gbt(model, kind, X[tr2], y[tr2], X[va], y[va])
    return auc_acc(y[te], proba(model, X[te])), float(y.mean())


def main():
    meta = xd.load_meta()
    rows = []
    for tgt in TARGETS:
        print('=== alvo %s ===' % tgt)
        aucs = {}
        for name, cols in BLOCKS.items():
            (auc, acc), pos = eval_block(meta, cols, tgt)
            aucs[name] = auc
            rows.append(dict(alvo=tgt, bloco=name, n_features=len(cols),
                             taxa_positivos=round(pos, 4), auc=round(auc, 4), acc=round(acc, 4)))
            print('  %-8s (%2d feats) | AUC %.4f | acc %.4f' % (name, len(cols), auc, acc))
        marg = aucs['ALL'] - aucs['POS+CTX']
        print('  -> geometria isolada (GEOM) = %.4f | marginal da geometria (ALL - POS+CTX) = %+.4f'
              % (aucs['GEOM'], marg))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'exp6_ablation.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'exp6_ablation.csv'))


if __name__ == '__main__':
    main()
