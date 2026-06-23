# -*- coding: utf-8 -*-
"""Abordagem A — Insight do alvo (alvo x geometria), SO FIFA WC 2022.
Prediz DIRETAMENTE P(portador perde a bola) a partir do contexto. Modelo: GBT (XGBoost).
Varre alvos + ablacao de blocos de features. Saida: star_analysis/A_target.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'express'))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import star_data as sd
from gbt_util import make_gbt, fit_gbt, proba, auc_acc

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
GEOM = sd.TAB_GEOM
POS = ['x', 'y', 'dist_to_goal', 'zone_code']
CTX = ['minute', 'duration']
BLOCKS = {'GEOM': GEOM, 'POS': POS, 'CTX': CTX, 'POS+CTX': POS + CTX, 'ALL': GEOM + POS + CTX}


def derive_targets(df):
    within = np.hypot(df['loss_x'] - df['x'], df['loss_y'] - df['y'])
    df['y_5s_5m'] = ((df['y_5s'] == 1) & (within <= 5.0)).astype(int)
    df['y_5s_9m'] = ((df['y_5s'] == 1) & (within <= 9.0)).astype(int)
    return df


def run_gbt(df, cols, tgt):
    X = df[cols].to_numpy(np.float32); y = df[tgt].to_numpy(np.int32)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    m, k = make_gbt(); fit_gbt(m, k, X[tr2], y[tr2], X[va], y[va])
    return auc_acc(y[te], proba(m, X[te]))


def main():
    df = derive_targets(sd.build())
    rows = []
    print('=== varredura de alvo (GBT, ALL features) ===')
    for tgt in ['y_5s', 'y_10s', 'y_5s_5m', 'y_5s_9m']:
        auc, acc = run_gbt(df, BLOCKS['ALL'], tgt)
        rows.append(dict(parte='alvo', alvo=tgt, bloco='ALL', taxa_pos=round(df[tgt].mean(), 4),
                         auc=round(auc, 4), acc=round(acc, 4)))
        print('  %-8s | pos %.3f | AUC %.4f' % (tgt, df[tgt].mean(), auc))
    best = max([r for r in rows if r['parte'] == 'alvo'], key=lambda r: r['auc'])['alvo']
    print('best_target =', best, '| ablacao:')
    aucs = {}
    for name, cols in BLOCKS.items():
        auc, acc = run_gbt(df, cols, best)
        aucs[name] = auc
        rows.append(dict(parte='ablacao', alvo=best, bloco=name, taxa_pos=round(df[best].mean(), 4),
                         auc=round(auc, 4), acc=round(acc, 4)))
        print('  %-8s | AUC %.4f' % (name, auc))
    print('  geometria isolada %.4f | marginal (ALL-POS+CTX) %+.4f' % (aucs['GEOM'], aucs['ALL'] - aucs['POS+CTX']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'A_target.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'A_target.csv'))


if __name__ == '__main__':
    main()
