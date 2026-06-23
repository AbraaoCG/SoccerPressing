# -*- coding: utf-8 -*-
"""star_data.py — builder dos dados das 3 abordagens candidatas a estrela (SO FIFA WC 2022, comp 43).

Por pressao registra: features tabulares (reuso geom_of do express_data), rotulos de perda,
TEMPO ate a perda + local da perda (p/ survival), e a acao seguinte do portador + posicoes dos
companheiros/adversarios (p/ contrafactual de decisao) + id do portador (p/ metrica de jogador).
Cache: cache/_star43.pkl.  Nao altera nada existente.
Uso: ./venv1/Scripts/python.exe scripts/star/star_data.py
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_os.chdir(str(_Path(__file__).resolve().parent.parent.parent))         # raiz
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'xR_experiments'))

import os, json, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from express_data import geom_of, TAB_GEOM, TAB_CTX, load_json, ts, LOCAL_R  # reuso

warnings.filterwarnings('ignore')
DATA_DIR = Path('StatsBomb_2/data')
COMP_ID = 43                       # FIFA World Cup 2022
HORIZON = 10.0                     # janela maxima p/ a perda (survival censura em 10 s)
PKL = 'cache/_star43.pkl'


def process_match(mid):
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

    # eventos de recuperacao do time que pressiona (= "portador perde a bola")
    rec_fail = ev['ball_recovery_recovery_failure'].fillna(False) if 'ball_recovery_recovery_failure' in ev.columns else pd.Series(False, index=ev.index)
    duel_out = ev.get('duel_outcome_name', pd.Series('', index=ev.index)).fillna('')
    rm = ((ev['type_name'].eq('Ball Recovery') & ~rec_fail) | ev['type_name'].eq('Interception')
          | (ev['type_name'].eq('Duel') & duel_out.str.contains('Won|Success')))
    recev = ev.loc[rm, ['period', 'team_id', 't', 'location']].copy()
    recev['rx'] = recev['location'].apply(lambda L: float(L[0]) if isinstance(L, list) else np.nan)
    recev['ry'] = recev['location'].apply(lambda L: float(L[1]) if isinstance(L, list) else np.nan)

    # acoes on-ball do portador (time pressionado) sob pressao — p/ id do portador + decisao
    up = ev.get('under_pressure', pd.Series(False, index=ev.index)).fillna(False)
    act_types = ['Pass', 'Carry', 'Dribble', 'Ball Receipt*', 'Shot', 'Clearance']
    cact = ev[up & ev['type_name'].isin(act_types)][
        ['period', 'team_id', 't', 'type_name', 'player_id', 'player_name', 'location',
         'pass_end_location']].copy()

    out = []
    for _, row in pr.iterrows():
        fr = frames.get(row['id']); loc = row['location']
        if fr is None or not isinstance(loc, list):
            continue
        ball = np.array(loc[:2], float)
        advs, nonadv = [], []
        for p in fr['freeze_frame']:
            (advs if p.get('teammate') else nonadv).append(p['location'][:2])
        if len(nonadv) == 0 or len(advs) == 0:
            continue
        nonadv = np.array(nonadv, float); advs = np.array(advs, float)
        ci = int(np.argmin(np.linalg.norm(nonadv - ball, axis=1)))
        carrier = nonadv[ci]
        mates = np.delete(nonadv, ci, axis=0)                # companheiros do portador (opcoes de passe)
        if np.min(np.linalg.norm(advs - carrier, axis=1)) > LOCAL_R:
            continue
        prim, adv_local, mate_local = geom_of(ball, carrier, advs, mates)

        x, y = float(ball[0]), float(ball[1])
        period = row['period']; tp = row['t']; pteam = row['team_id']; possteam = row.get('possession_team_id')

        # tempo ate a perda (recuperacao do time que pressiona em (tp, tp+HORIZON])
        rsel = recev[(recev['period'] == period) & (recev['team_id'] == pteam) &
                     (recev['t'] > tp) & (recev['t'] <= tp + HORIZON)]
        if len(rsel):
            j = rsel['t'].values.argmin()
            t_loss = float(rsel['t'].values[j] - tp); ev_loss = 1
            lx, ly = float(rsel['rx'].values[j]), float(rsel['ry'].values[j])
        else:
            t_loss = HORIZON; ev_loss = 0; lx, ly = np.nan, np.nan
        y_5s = int(ev_loss and t_loss <= 5.0)
        y_10s = int(ev_loss)

        # portador (id) + acao escolhida (decisao) — acao do time pressionado mais proxima no tempo
        ca = cact[(cact['period'] == period) & (cact['team_id'] == possteam) &
                  (cact['t'] >= tp - 1.0) & (cact['t'] <= tp + 3.0)]
        if len(ca):
            k = (ca['t'] - tp).abs().values.argmin()
            carrier_id = ca['player_id'].values[k]; act_type = ca['type_name'].values[k]
            pend = ca['pass_end_location'].values[k]
            act_end = (np.array(pend[:2], float) if isinstance(pend, list) else np.array([np.nan, np.nan]))
        else:
            carrier_id = np.nan; act_type = None; act_end = np.array([np.nan, np.nan])

        rec_d = {'match_id': int(mid), 'x': x, 'y': y,
                 'dist_to_goal': math.sqrt((120 - x) ** 2 + (40 - y) ** 2),
                 'zone_code': (0 if x <= 40 else (1 if x <= 80 else 2)),
                 'minute': float(row.get('minute', 0)), 'duration': float(row.get('duration', 0) or 0),
                 'y_5s': y_5s, 'y_10s': y_10s, 't_loss': t_loss, 'ev_loss': ev_loss,
                 'loss_x': lx, 'loss_y': ly, 'carrier_id': carrier_id, 'act_type': act_type,
                 'act_end_x': float(act_end[0]), 'act_end_y': float(act_end[1]),
                 'carrier_x': float(carrier[0]), 'carrier_y': float(carrier[1]),
                 'mates': mates.astype(np.float32), 'advs': advs.astype(np.float32)}
        rec_d.update(prim)
        out.append(rec_d)
    return out


def build(force=False):
    if not force and os.path.exists(PKL):
        return pd.read_pickle(PKL)
    mfiles = glob.glob(str(DATA_DIR / 'matches' / str(COMP_ID) / '*.json'))
    mids = []
    for mf in mfiles:
        data = load_json(mf)
        if not data:
            continue
        for mm in data:
            mid = mm['match_id']
            if (DATA_DIR / 'three-sixty' / (str(mid) + '.json')).exists():
                mids.append(mid)
    mids = sorted(set(mids))
    print('jogos comp %d com 360: %d' % (COMP_ID, len(mids)))
    recs = []
    for j, mid in enumerate(mids):
        recs.extend(process_match(mid))
        if (j + 1) % 16 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(recs)))
    df = pd.DataFrame(recs)
    df.to_pickle(PKL)
    print('cache salvo:', PKL, '| pressoes', len(df))
    return df


# colunas tabulares (mesma divisao do express): geometria de jogadores / posicao / contexto
TAB_FEATURES = TAB_CTX + TAB_GEOM

if __name__ == '__main__':
    df = build(force=True)
    print('pressoes:', len(df), '| taxa perda y_5s %.3f | y_10s %.3f' % (df['y_5s'].mean(), df['y_10s'].mean()))
    print('com portador id:', df['carrier_id'].notna().mean().round(3),
          '| jogadores distintos:', df['carrier_id'].nunique())
    print('ev_loss=1: %.3f | t_loss medio (perdas): %.2f s' %
          (df['ev_loss'].mean(), df.loc[df['ev_loss'] == 1, 't_loss'].mean()))
