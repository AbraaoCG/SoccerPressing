# -*- coding: utf-8 -*-
"""xt_lite.py — grade de Expected Threat (xT, estilo Karun Singh) estimada do evento da FIFA WC 2022.
xT(celula) = P(chutar)*P(gol|chute) + P(mover)*sum_c' T(c->c') xT(c').  Cache cache/_xt43.npy."""
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
from express_data import load_json, ts  # reuso (mesmo sys.path do chamador)

DATA_DIR = Path('StatsBomb_2/data')
COMP_ID = 43
GX, GY = 16, 12
XT_NPY = 'cache/_xt43.npy'


def cell(x, y):
    cx = min(max(int(x / 120.0 * GX), 0), GX - 1)
    cy = min(max(int(y / 80.0 * GY), 0), GY - 1)
    return cy * GX + cx


def build_xt(force=False):
    if not force and os.path.exists(XT_NPY):
        return np.load(XT_NPY)
    n = GX * GY
    shots = np.zeros(n); goals = np.zeros(n); moves = np.zeros(n)
    trans = np.zeros((n, n))
    mfiles = glob.glob(str(DATA_DIR / 'matches' / str(COMP_ID) / '*.json'))
    mids = []
    for mf in mfiles:
        data = load_json(mf)
        if data:
            for mm in data:
                if (DATA_DIR / 'three-sixty' / (str(mm['match_id']) + '.json')).exists():
                    mids.append(mm['match_id'])
    for mid in sorted(set(mids)):
        ev = load_json(DATA_DIR / 'events' / (str(mid) + '.json'))
        if not ev:
            continue
        ev = pd.json_normalize(ev, sep='_')
        for _, r in ev.iterrows():
            loc = r.get('location')
            if not isinstance(loc, list):
                continue
            c = cell(loc[0], loc[1]); tp = r['type_name']
            if tp == 'Shot':
                shots[c] += 1
                if r.get('shot_outcome_name') == 'Goal':
                    goals[c] += 1
            elif tp in ('Pass', 'Carry'):
                end = r.get('pass_end_location') if tp == 'Pass' else r.get('carry_end_location')
                if isinstance(end, list):
                    moves[c] += 1; trans[c, cell(end[0], end[1])] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        tot = shots + moves
        s = np.where(tot > 0, shots / tot, 0.0)          # P(chute|acao na celula)
        g = np.where(shots > 0, goals / shots, 0.0)      # P(gol|chute)
        T = np.where(moves[:, None] > 0, trans / np.maximum(moves[:, None], 1), 0.0)
    xt = np.zeros(n)
    for _ in range(30):
        xt = s * g + (1 - s) * (T @ xt)
    np.save(XT_NPY, xt)
    print('xT-lite: %d celulas | media %.4f | max %.4f' % (n, xt.mean(), xt.max()))
    return xt


def xt_at(xt, x, y):
    return float(xt[cell(x, y)])
