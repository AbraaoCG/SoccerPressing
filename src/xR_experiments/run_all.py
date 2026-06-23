# -*- coding: utf-8 -*-
"""run_all.py — consolida os 5 experimentos + referencias de hoje em express_analysis/summary.csv.
Uso: ./venv1/Scripts/python.exe scripts/express/run_all.py
(roda os 5 na ordem se os CSVs nao existirem; senao so consolida)."""
import os, sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import express_data as xd  # garante chdir p/ raiz + cache

OUT = 'express_analysis'; os.makedirs(OUT, exist_ok=True)
HERE = Path(__file__).resolve().parent
PY = sys.executable
SCRIPTS = ['exp1_xgboost.py', 'exp2_target.py', 'exp3_crosscomp.py',
           'exp4_soccermap_cnn.py', 'exp5_combined.py']

# referencias de hoje (resultados ja medidos no projeto, alvo recovered y_10s)
HOJE = [
    dict(exp='hoje', modelo='CNN raster (1ch)', alvo='y_10s', split='aleatorio', auc=0.589, acc=None),
    dict(exp='hoje', modelo='stacking (ctx+CNN)', alvo='y_10s', split='aleatorio', auc=0.592, acc=None),
    dict(exp='hoje', modelo='GNN within30 (C++)', alvo='y_10s', split='aleatorio', auc=0.5747, acc=0.720),
    dict(exp='hoje', modelo='logistica contexto', alvo='y_10s', split='aleatorio', auc=0.541, acc=None),
]


def main():
    run = '--run' in sys.argv
    for s in SCRIPTS:
        csv = os.path.join(OUT, s.split('_')[0].replace('exp', 'exp') + '.csv')
        # nome do csv = exp1.csv etc
        key = s.split('_')[0]
        csv = os.path.join(OUT, key + '.csv')
        if run or not os.path.exists(csv):
            print('=== rodando', s, '===')
            subprocess.run([PY, str(HERE / s)], check=True)

    frames = []
    for key in ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']:
        p = os.path.join(OUT, key + '.csv')
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    frames.append(pd.DataFrame(HOJE))
    summ = pd.concat(frames, ignore_index=True)
    cols = [c for c in ['exp', 'modelo', 'alvo', 'split', 'protocolo', 'teste',
                        'taxa_positivos', 'auc', 'acc', 'n_test'] if c in summ.columns]
    summ = summ[cols]
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    print('\n===== SUMMARY (express) =====')
    print(summ.to_string(index=False))
    print('\nsalvo em', os.path.join(OUT, 'summary.csv'))


if __name__ == '__main__':
    main()
