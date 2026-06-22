# -*- coding: utf-8 -*-
"""target_ladder_ablation.py — testa se a INFORMATIVIDADE DA GEOMETRIA depende do alvo, ao longo
de TODA a escada de alvos (não só 2 pontos). Para cada alvo, mede a AUC só-geometria e a
contribuição marginal da geometria = AUC(ALL) - AUC(POS+CTX). Se crescer conforme o alvo fica
mais local, 'depende do alvo' se sustenta. GBT (mesmo do exp6). Saída: star_analysis/target_ladder.csv.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
GEOM = xd.TAB_GEOM
POS = ['x', 'y', 'dist_to_goal', 'zone_code']
CTX = ['minute', 'duration']
# alvos ordenados do mais FROUXO ao mais LOCAL (escada espaco-temporal)
LADDER = ['y_10s', 'y_5s', 'y_2act', 'y_5s_9m', 'y_5s_5m']


def auc_of(meta, cols, y):
    X = np.nan_to_num(meta[cols].to_numpy(np.float32))
    idx = np.arange(len(y)); tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    m, k = make_gbt(); fit_gbt(m, k, X[tr2], y[tr2], X[va], y[va])
    return float(roc_auc_score(y[te], proba(m, X[te])))


def main():
    meta = xd.load_meta()
    rows = []
    for tgt in LADDER:
        y = meta[tgt].to_numpy(np.int32)
        a_geom = auc_of(meta, GEOM, y)
        a_posctx = auc_of(meta, POS + CTX, y)
        a_all = auc_of(meta, GEOM + POS + CTX, y)
        rows.append(dict(alvo=tgt, base=round(float(y.mean()), 4),
                         AUC_geom=round(a_geom, 4), AUC_pos_ctx=round(a_posctx, 4),
                         AUC_all=round(a_all, 4), marginal_geom=round(a_all - a_posctx, 4)))
        print('[%-8s] base %.3f | geom %.4f | pos+ctx %.4f | ALL %.4f | marginal geom %+.4f'
              % (tgt, y.mean(), a_geom, a_posctx, a_all, a_all - a_posctx))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'target_ladder.csv'), index=False)
    print('\n', res.to_string(index=False))
    # tendencia na escada espacial a 5 s: y_5s (sem raio) -> y_5s_9m -> y_5s_5m
    sp = res.set_index('alvo')
    print('\nescada espacial @5s (geom-only): y_5s %.3f -> y_5s_9m %.3f -> y_5s_5m %.3f'
          % (sp.loc['y_5s', 'AUC_geom'], sp.loc['y_5s_9m', 'AUC_geom'], sp.loc['y_5s_5m', 'AUC_geom']))
    print('salvo em', os.path.join(OUT, 'target_ladder.csv'))


if __name__ == '__main__':
    main()
