# -*- coding: utf-8 -*-
"""label_noise.py — quantifica o "ruído de rótulo" do alvo frouxo (y_10s) que o alvo estrito
(y_5s_5m) remove: entre as recuperações em <=10 s, que fração ocorre LONGE (espaço) e/ou TARDE
(tempo) do ponto da pressão --- isto é, dificilmente causada por aquela pressão.
Sustenta o enquadramento de "mecanismo". Saída: star_analysis/label_noise.csv.
Uso: ./venv1/Scripts/python.exe src/xR_experiments/label_noise.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import express_data as xd  # chdir raiz

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)


def main():
    df = pd.read_pickle('cache/_star_all.pkl')
    pos = df[df['ev_loss'] == 1].copy()                       # recuperou em <=10 s (positivos de y_10s)
    pos['dist'] = np.hypot(pos['loss_x'] - pos['x'], pos['loss_y'] - pos['y'])
    n = len(pos)
    far5 = (pos['dist'] > 5).mean()
    far10 = (pos['dist'] > 10).mean()
    late5 = (pos['t_loss'] > 5).mean()
    excl_55 = ((pos['t_loss'] > 5) | (pos['dist'] > 5) | pos['dist'].isna()).mean()   # fora do y_5s_5m
    rows = [
        dict(metric='positivos y_10s (recup. <=10 s)', valor=n),
        dict(metric='mediana tempo ate recuperar (s)', valor=round(float(pos['t_loss'].median()), 2)),
        dict(metric='mediana distancia da recuperacao ao ponto de pressao (m)', valor=round(float(pos['dist'].median()), 2)),
        dict(metric='% recuperacoes a >5 m do ponto', valor=round(100 * float(far5), 1)),
        dict(metric='% recuperacoes a >10 m do ponto', valor=round(100 * float(far10), 1)),
        dict(metric='% recuperacoes apos >5 s', valor=round(100 * float(late5), 1)),
        dict(metric='% positivos de y_10s EXCLUIDOS por y_5s_5m (ruido removido)', valor=round(100 * float(excl_55), 1)),
    ]
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, 'label_noise.csv'), index=False)
    print(res.to_string(index=False))
    print('\nsalvo em', os.path.join(OUT, 'label_noise.csv'))


if __name__ == '__main__':
    main()
