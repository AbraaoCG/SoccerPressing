# -*- coding: utf-8 -*-
"""express_data.py — backbone de dados compartilhado dos 5 experimentos (ideias do exPress).

Constroi UMA vez, para TODAS as competicoes com 360 (mesma base do _run2b, ~91k pressoes):
  - identificadores: competition_id, match_id  (necessarios p/ split cross-competition)
  - features TABULARES (geometria de adversarios + contexto), SEM as endogenas (had_tackle/foul/block)
  - raster MULTICANAL (4 canais, estilo SoccerMap) centrado no portador, frente=+X
  - MULTIPLOS rotulos por pressao: y_5s, y_10s, y_2act, y_5s_5m, y_5s_9m

Nao altera nenhum arquivo existente. Caches proprios em cache/_express_*.
Uso:   ./venv1/Scripts/python.exe scripts/express/express_data.py   (forca rebuild do cache)
       (os experimentos chamam load() para reusar o cache)
"""
import os as _os
from pathlib import Path as _Path
_os.chdir(str(_Path(__file__).resolve().parent.parent.parent))  # raiz do projeto

import json, os, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = Path('StatsBomb_2/data')
LOCAL_R = 30.0
G, HALF, SIG = 32, 25.0, 2.0                  # raster: grade 32x32, extensao +-25 m, sigma do bump
WIN_5S, WIN_10S = 5.0, 10.0
os.makedirs('cache', exist_ok=True)
META_PKL = 'cache/_express_meta.pkl'
ARR_NPZ = 'cache/_express_arr.npz'

# colunas tabulares expostas aos modelos (SEM endogenas)
TAB_GEOM = ['n_adv_5m', 'n_adv_10m', 'n_adv_15m', 'nearest_adv_dist', 'adv_arc_coverage',
            'largest_free_angle', 'free_lane_forward', 'adv_centroid_dist', 'adv_centroid_angle',
            'adv_spread', 'n_sup_10m', 'nearest_sup_dist']
TAB_CTX = ['dist_to_goal', 'zone_code', 'minute', 'duration', 'x', 'y']
TAB_FEATURES = TAB_CTX + TAB_GEOM
LABELS = ['y_5s', 'y_10s', 'y_2act', 'y_5s_5m', 'y_5s_9m']

# eventos "on-ball" (acoes) para o criterio de "2 acoes" (exclui Pressure e marcacoes passivas)
ACTION_TYPES = {'Pass', 'Carry', 'Ball Receipt*', 'Shot', 'Dribble', 'Clearance', 'Ball Recovery',
                'Interception', 'Duel', 'Dispossessed', 'Miscontrol', 'Foul Committed', 'Block',
                'Goal Keeper', 'Shield', 'Dribbled Past', '50/50'}


def load_json(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def ts(t):
    hh, mm, ss = t.split(':'); return int(hh) * 3600 + int(mm) * 60 + float(ss)


def geom_of(ball, carrier, advs, sups):
    """Primitivas geometricas dos adversarios (e algumas de apoio) em torno do portador."""
    res = {}
    d = advs - carrier
    adv_r = np.column_stack([-d[:, 0], d[:, 1]])               # frente = +X
    dist = np.linalg.norm(adv_r, axis=1)
    for r in (5, 10, 15):
        res['n_adv_%dm' % r] = int((dist <= r).sum())
    res['nearest_adv_dist'] = float(dist.min())
    near = adv_r[dist <= 10]
    angs = sorted(((np.degrees(np.arctan2(near[:, 1], near[:, 0])) + 360) % 360).tolist()) if len(near) else []
    res['adv_arc_coverage'] = len(set(int(a // 45) for a in angs)) / 8.0
    if len(angs) == 0:
        res['largest_free_angle'] = 360.0; fm = 0.0
    elif len(angs) == 1:
        res['largest_free_angle'] = 360.0; fm = (angs[0] + 180) % 360
    else:
        gaps = [((angs[(i + 1) % len(angs)] - angs[i]) % 360, i) for i in range(len(angs))]
        gap, i = max(gaps); fm = (angs[i] + gap / 2) % 360; res['largest_free_angle'] = gap
    res['free_lane_forward'] = int(math.cos(math.radians(fm)) > 0)
    loc = adv_r[dist <= LOCAL_R]
    cen = loc.mean(0) if len(loc) else np.zeros(2)
    res['adv_centroid_dist'] = float(np.linalg.norm(cen))
    res['adv_centroid_angle'] = float((np.degrees(np.arctan2(cen[1], cen[0])) + 360) % 360)
    res['adv_spread'] = float(np.linalg.norm(loc, axis=1).std()) if len(loc) > 1 else 0.0
    # apoio (companheiros do portador) reorientado
    if len(sups):
        ds = sups - carrier
        sup_r = np.column_stack([-ds[:, 0], ds[:, 1]])
        sdist = np.linalg.norm(sup_r, axis=1)
        res['n_sup_10m'] = int((sdist <= 10).sum())
        res['nearest_sup_dist'] = float(sdist.min())
        sup_local = sup_r[sdist <= LOCAL_R]
    else:
        res['n_sup_10m'] = 0; res['nearest_sup_dist'] = float(LOCAL_R)
        sup_local = np.zeros((0, 2))
    adv_local = adv_r[dist <= LOCAL_R]
    return res, adv_local, sup_local


def process_match(mid, comp_id):
    ev_data = load_json(DATA_DIR / 'events' / (str(mid) + '.json'))
    fr_data = load_json(DATA_DIR / 'three-sixty' / (str(mid) + '.json'))
    if ev_data is None or fr_data is None:
        return []
    ev = pd.json_normalize(ev_data, sep='_')
    frames = {fr['event_uuid']: fr for fr in fr_data}
    ev['t'] = ev['timestamp'].apply(ts)
    pr = ev[ev['type_name'] == 'Pressure'].copy()
    if len(pr) == 0:
        return []

    rec_fail = ev['ball_recovery_recovery_failure'].fillna(False) if 'ball_recovery_recovery_failure' in ev.columns else pd.Series(False, index=ev.index)
    duel_out = ev.get('duel_outcome_name', pd.Series('', index=ev.index)).fillna('')
    rm = ((ev['type_name'].eq('Ball Recovery') & ~rec_fail) | ev['type_name'].eq('Interception')
          | (ev['type_name'].eq('Duel') & duel_out.str.contains('Won|Success')))

    # localizacao da recuperacao (para os rotulos com restricao espacial)
    recev = ev.loc[rm, ['period', 'team_id', 't', 'location']].copy()
    recev['rx'] = recev['location'].apply(lambda L: float(L[0]) if isinstance(L, list) else np.nan)
    recev['ry'] = recev['location'].apply(lambda L: float(L[1]) if isinstance(L, list) else np.nan)
    R = recev[['period', 'team_id', 't', 'rx', 'ry']].rename(columns={'t': 'tr'}).sort_values('tr')
    L = pr[['id', 'period', 'team_id', 't']].rename(columns={'t': 'tp'}).sort_values('tp')

    def merge_win(win):
        m = pd.merge_asof(L, R, left_on='tp', right_on='tr', by=['period', 'team_id'],
                          direction='forward', tolerance=win)
        return m
    m5 = merge_win(WIN_5S); m10 = merge_win(WIN_10S)
    rec5 = dict(zip(m5['id'], m5['tr'].notna().astype(int)))
    rec10 = dict(zip(m10['id'], m10['tr'].notna().astype(int)))
    rxy5 = dict(zip(m5['id'], zip(m5['rx'], m5['ry'])))

    # criterio "2 acoes": fluxo de acoes on-ball por periodo
    act = ev[ev['type_name'].isin(ACTION_TYPES)][['period', 't', 'team_id']].copy()
    act['is_rec'] = rm.reindex(act.index).fillna(False).to_numpy()
    per_act = {}
    for p, g in act.sort_values('t').groupby('period'):
        per_act[p] = (g['t'].to_numpy(), g['team_id'].to_numpy(), g['is_rec'].to_numpy())

    def y_two_actions(period, tp, pteam):
        if period not in per_act:
            return 0
        tt, team, isr = per_act[period]
        j = int(np.searchsorted(tt, tp, side='right'))
        for k in range(j, min(j + 2, len(tt))):       # proximas 2 acoes
            if isr[k] and team[k] == pteam:
                return 1
        return 0

    out = []
    for _, row in pr.iterrows():
        fr = frames.get(row['id']); loc = row['location']
        if fr is None or not isinstance(loc, list):
            continue
        ball = np.array(loc[:2], float)
        advs, sups = [], []
        for p in fr['freeze_frame']:
            (advs if p.get('teammate') else sups).append(p['location'][:2])
        if len(sups) == 0 or len(advs) == 0:
            continue
        sups = np.array(sups, float); advs = np.array(advs, float)
        ci = int(np.argmin(np.linalg.norm(sups - ball, axis=1)))
        carrier = sups[ci]
        sup_others = np.delete(sups, ci, axis=0)
        if np.min(np.linalg.norm(advs - carrier, axis=1)) > LOCAL_R:
            continue
        prim, adv_local, sup_local = geom_of(ball, carrier, advs, sup_others)

        x, y = float(ball[0]), float(ball[1])
        pid = row['id']; pteam = row['team_id']; period = row['period']; tp = row['t']
        y5 = int(rec5.get(pid, 0)); y10 = int(rec10.get(pid, 0))
        rx, ry = rxy5.get(pid, (np.nan, np.nan))
        within = np.nan
        if y5 and isinstance(rx, float) and not math.isnan(rx):
            within = math.hypot(rx - x, ry - y)
        y5_5m = int(y5 and (within == within) and within <= 5.0)
        y5_9m = int(y5 and (within == within) and within <= 9.0)
        y2 = y_two_actions(period, tp, pteam)

        rec_d = {'competition_id': int(comp_id), 'match_id': int(mid),
                 'x': x, 'y': y, 'dist_to_goal': math.sqrt((120 - x) ** 2 + (40 - y) ** 2),
                 'zone_code': (0 if x <= 40 else (1 if x <= 80 else 2)),
                 'minute': float(row.get('minute', 0)), 'duration': float(row.get('duration', 0) or 0),
                 'y_5s': y5, 'y_10s': y10, 'y_2act': y2, 'y_5s_5m': y5_5m, 'y_5s_9m': y5_9m}
        rec_d.update(prim)
        out.append((rec_d, adv_local.astype(np.float32), sup_local.astype(np.float32)))
    return out


def _build_raster(meta, adv_l, sup_l):
    """4 canais centrados no portador (origem, frente=+X): pressers, apoio, dist-ao-gol, ang-ao-gol."""
    N = len(meta)
    xs = np.linspace(-HALF, HALF, G); gx, gy = np.meshgrid(xs, xs)
    IMG = np.zeros((N, 4, G, G), np.float16)
    cx = meta['x'].to_numpy(); cy = meta['y'].to_numpy()
    for i in range(N):
        a = adv_l[i]; s = sup_l[i]
        for (px, py) in a:
            IMG[i, 0] += np.exp(-(((gx - px) ** 2 + (gy - py) ** 2) / (2 * SIG ** 2))).astype(np.float16)
        for (px, py) in s:
            IMG[i, 1] += np.exp(-(((gx - px) ** 2 + (gy - py) ** 2) / (2 * SIG ** 2))).astype(np.float16)
        # gol (120,40) reorientado p/ o referencial local do portador
        glx = cx[i] - 120.0; gly = 40.0 - cy[i]
        dgoal = np.sqrt((gx - glx) ** 2 + (gy - gly) ** 2)
        IMG[i, 2] = (dgoal / 100.0).astype(np.float16)                         # distancia ao gol (norm.)
        IMG[i, 3] = (np.arctan2(gly - gy, glx - gx) / np.pi).astype(np.float16)  # angulo ao gol (norm.)
    return IMG


def build(force=False):
    if not force and os.path.exists(META_PKL) and os.path.exists(ARR_NPZ):
        return load()
    mfiles = glob.glob(str(DATA_DIR / 'matches' / '*' / '*.json'))
    mid2comp = {}
    for mf in mfiles:
        comp_id = Path(mf).parent.name
        data = load_json(mf)
        if not data:
            continue
        for mm in data:
            mid = mm['match_id']
            if (DATA_DIR / 'three-sixty' / (str(mid) + '.json')).exists():
                mid2comp[mid] = comp_id
    mids = sorted(mid2comp)
    if os.environ.get('SMOKE') == '1':
        mids = mids[:8]
    print('jogos com 360:', len(mids))
    recs, adv_l, sup_l = [], [], []
    for j, mid in enumerate(mids):
        for rec_d, a, s in process_match(mid, mid2comp[mid]):
            recs.append(rec_d); adv_l.append(a); sup_l.append(s)
        if (j + 1) % 40 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(recs)))
    meta = pd.DataFrame(recs)
    IMG = _build_raster(meta, adv_l, sup_l)
    meta.to_pickle(META_PKL)
    np.savez_compressed(ARR_NPZ, IMG=IMG)
    print('cache salvo:', META_PKL, ARR_NPZ, '| IMG', IMG.shape)
    return meta, IMG


COMP_NAMES = {9: 'Bundesliga 23/24', 43: 'FIFA WC 2022', 53: "Women's Euro",
              55: 'UEFA Euro 20+24', 72: "Women's WC 2023"}


def load_meta():
    """So a tabela (rapido) — para os experimentos tabulares (GBT)."""
    return pd.read_pickle(META_PKL)


def load():
    meta = pd.read_pickle(META_PKL)
    IMG = np.load(ARR_NPZ)['IMG']
    return meta, IMG


if __name__ == '__main__':
    meta, IMG = build(force=True)
    print('amostras:', len(meta), '| IMG', IMG.shape)
    print('competicoes:', meta['competition_id'].nunique(),
          '| top:', meta['competition_id'].value_counts().head(6).to_dict())
    for lb in LABELS:
        print('  taxa %-8s = %.3f' % (lb, meta[lb].mean()))
