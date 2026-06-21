# -*- coding: utf-8 -*-
"""triggers_xR.py — conecta a análise de GATILHOS (triggers_v0.ipynb, colega) ao nosso modelo xR.
Para cada pressão da Euro 2020 (comp 55/43): computa nossos atributos (geom_of) + alvo y_5s_5m
(recuperar <=5 s E a <=5 m) e os GATILHOS de passe do colega (recuo, lateral, perto da lateral,
cobrança de lateral, jogador isolado). Avalia se o tipo de gatilho ADICIONA SINAL ao modelo
logístico, com as métricas padronizadas (AUC, PR-AUC, Brier, ECE, lift). Saída: star_analysis/.
Uso: ./venv1/Scripts/python.exe src/xR_experiments/triggers_xR.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import express_data as xd                       # importa -> chdir raiz; geom_of/load_json/ts/DATA_DIR

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
COMP_ID, SEASON_ID = 55, 43                     # UEFA Euro 2020 (mesma base do colega)
WIN10, PASSWIN = 10.0, 6.0
ISO_R, BACK_M, LAT_M, SIDE_M = 15.0, 10.0, 10.0, 5.0
FLAGS = ['recuo', 'lateral', 'perto_lateral', 'cobranca_lateral', 'isolado', 'gatilho_passe']


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def topk(y, p, frac=0.10):
    k = max(1, int(len(p) * frac)); idx = np.argsort(-p)[:k]
    return y[idx].mean()


def xy(v):
    return (float(v[0]), float(v[1])) if isinstance(v, list) and len(v) >= 2 else (np.nan, np.nan)


def process_match(mid):
    ev_data = xd.load_json(xd.DATA_DIR / 'events' / (str(mid) + '.json'))
    fr_data = xd.load_json(xd.DATA_DIR / 'three-sixty' / (str(mid) + '.json'))
    if ev_data is None or fr_data is None:
        return []
    ev = pd.json_normalize(ev_data, sep='_')
    frames = {fr['event_uuid']: fr for fr in fr_data}
    ev['t'] = ev['timestamp'].apply(xd.ts)
    pr = ev[ev['type_name'] == 'Pressure'].copy()
    if len(pr) == 0:
        return []

    # recuperação (perda do portador) com tempo e local
    rec_fail = ev['ball_recovery_recovery_failure'].fillna(False) if 'ball_recovery_recovery_failure' in ev.columns else pd.Series(False, index=ev.index)
    duel_out = ev.get('duel_outcome_name', pd.Series('', index=ev.index)).fillna('')
    rm = ((ev['type_name'].eq('Ball Recovery') & ~rec_fail) | ev['type_name'].eq('Interception')
          | (ev['type_name'].eq('Duel') & duel_out.str.contains('Won|Success', case=False)))
    rec = ev.loc[rm, ['period', 'team_id', 't', 'location']].copy()
    rec['rx'] = rec['location'].apply(lambda v: xy(v)[0]); rec['ry'] = rec['location'].apply(lambda v: xy(v)[1])
    R = rec[['period', 'team_id', 't', 'rx', 'ry']].rename(columns={'t': 'tr'}).sort_values('tr')
    L = pr[['id', 'period', 'team_id', 't']].rename(columns={'t': 'tp'}).sort_values('tp')
    m = pd.merge_asof(L, R, left_on='tp', right_on='tr', by=['period', 'team_id'], direction='forward', tolerance=WIN10)
    trec = dict(zip(m['id'], m['tr'])); rxy = dict(zip(m['id'], zip(m['rx'], m['ry'])))

    # passes completos (para os gatilhos de passe)
    po = ev.get('pass_outcome_name', pd.Series(np.nan, index=ev.index))
    pt = ev.get('pass_type_name', pd.Series(np.nan, index=ev.index))
    passes = ev[ev['type_name'].eq('Pass') & po.isna()].copy()
    passes['ptype'] = pt.reindex(passes.index)

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
        carrier = nonadv[ci]; mates = np.delete(nonadv, ci, axis=0)
        prim, _, _ = xd.geom_of(ball, carrier, advs, mates)

        x, y = float(ball[0]), float(ball[1]); tp = row['t']; pteam = row['team_id']; period = row['period']
        tr = trec.get(row['id'], np.nan); rx, ry = rxy.get(row['id'], (np.nan, np.nan))
        has = pd.notna(tr); dt = (tr - tp) if has else np.nan
        dist = (np.hypot(rx - x, ry - y) if (has and pd.notna(rx)) else np.nan)
        y5 = int(has and dt <= 5.0)
        y5_5 = int(y5 and pd.notna(dist) and dist <= 5.0)

        # gatilho = passe completo do adversário (time pressionado) mais recente em <=6 s antes
        cand = passes[(passes['period'] == period) & (passes['team_id'] != pteam) &
                      (passes['t'] >= tp - PASSWIN) & (passes['t'] <= tp)]
        fl = {k: 0 for k in FLAGS}
        if len(cand):
            c = cand.sort_values('t').iloc[-1]
            sx, sy = xy(c['location']); ex, ey = xy(c.get('pass_end_location'))
            fl['gatilho_passe'] = 1
            if pd.notna(sx) and pd.notna(ex):
                dx, dy = ex - sx, ey - sy
                if dx <= -BACK_M:
                    fl['recuo'] = 1
                if abs(dy) >= LAT_M and abs(dx) < LAT_M:
                    fl['lateral'] = 1
                if (ey <= SIDE_M or ey >= 80 - SIDE_M) and c['ptype'] != 'Throw-in':
                    fl['perto_lateral'] = 1
            if c['ptype'] == 'Throw-in':
                fl['cobranca_lateral'] = 1
        fl['isolado'] = int(prim.get('nearest_sup_dist', 99) > ISO_R)   # portador sem companheiro <=15 m

        rec_d = {'x': x, 'y': y, 'dist_to_goal': float(np.hypot(120 - x, 40 - y)),
                 'zone_code': (0 if x <= 40 else (1 if x <= 80 else 2)),
                 'minute': float(row.get('minute', 0)), 'duration': float(row.get('duration', 0) or 0),
                 'y_5s': y5, 'y_5s_5m': y5_5}
        rec_d.update(prim); rec_d.update(fl)
        out.append(rec_d)
    return out


def rates_table(df, tgt):
    base = df[tgt].mean(); rows = [dict(grupo='BASE (todas as pressões)', n=len(df),
                                        taxa=round(float(base), 4), razao=1.0, p=np.nan)]
    for f in [x for x in FLAGS if x != 'gatilho_passe']:
        sub = df[df[f] == 1]; oth = df[df[f] == 0]
        if len(sub) < 10:
            continue
        chi2, p, _, _ = stats.chi2_contingency(
            [[int(sub[tgt].sum()), len(sub) - int(sub[tgt].sum())],
             [int(oth[tgt].sum()), len(oth) - int(oth[tgt].sum())]])
        rows.append(dict(grupo=f, n=len(sub), taxa=round(float(sub[tgt].mean()), 4),
                         razao=round(float(sub[tgt].mean() / base), 3), p=float(p)))
    return pd.DataFrame(rows)


def eval_logit(X, y, tag):
    idx = np.arange(len(y)); tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]; yt = y[te]; base = yt.mean()
    p10 = topk(yt, p)
    return dict(conjunto=tag, AUC=round(float(roc_auc_score(yt, p)), 4),
               PR_AUC=round(float(average_precision_score(yt, p)), 4),
               Brier=round(float(brier_score_loss(yt, p)), 4), ECE=round(float(ece(yt, p)), 4),
               lift_top10=round(float(p10 / base), 2))


def main():
    matches = xd.load_json(xd.DATA_DIR / 'matches' / str(COMP_ID) / (str(SEASON_ID) + '.json'))
    mids = sorted(m['match_id'] for m in matches)
    recs = []
    for j, mid in enumerate(mids):
        recs.extend(process_match(mid))
        if (j + 1) % 17 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(recs)))
    df = pd.DataFrame(recs)
    print('Euro 2020 | pressoes %d | base y_5s_5m %.4f | base y_5s %.4f' %
          (len(df), df['y_5s_5m'].mean(), df['y_5s'].mean()))
    print('cobertura de gatilhos:', {f: int(df[f].sum()) for f in FLAGS})

    # (1) taxa de recuperacao por gatilho (baseline honesto), y_5s_5m e y_5s
    for tgt in ['y_5s_5m', 'y_5s']:
        rt = rates_table(df, tgt); rt.to_csv(os.path.join(OUT, 'triggers_rates_%s.csv' % tgt), index=False)
        print('\n=== taxa de recuperacao por gatilho (%s) ===' % tgt); print(rt.to_string(index=False))

    # (2) sinal incremental do tipo de gatilho no modelo logistico (metricas padronizadas)
    base_cols = xd.TAB_FEATURES
    rows = []
    for tgt in ['y_5s_5m', 'y_5s']:
        y = df[tgt].to_numpy(np.int32)
        Xb = np.nan_to_num(df[base_cols].to_numpy(np.float32))
        Xt = np.nan_to_num(df[base_cols + FLAGS].to_numpy(np.float32))
        rb = eval_logit(Xb, y, 'base (%s)' % tgt); rt = eval_logit(Xt, y, 'base+gatilhos (%s)' % tgt)
        rb['delta_AUC'] = 0.0; rt['delta_AUC'] = round(rt['AUC'] - rb['AUC'], 4)
        rows += [rb, rt]
        print('\n[%s] AUC base %.4f -> base+gatilhos %.4f (delta %+.4f)' %
              (tgt, rb['AUC'], rt['AUC'], rt['AUC'] - rb['AUC']))
    inc = pd.DataFrame(rows)
    inc.to_csv(os.path.join(OUT, 'triggers_increment.csv'), index=False)
    print('\n=== sinal incremental (logistica, metricas padronizadas) ===')
    print(inc.to_string(index=False))
    print('\nsalvo em', OUT)


if __name__ == '__main__':
    main()
