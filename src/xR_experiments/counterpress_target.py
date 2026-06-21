# -*- coding: utf-8 -*-
"""counterpress_target.py — testa a abordagem de CONTRAPRESSÃO (master: counterpress_v1.ipynb)
com o NOSSO alvo y_5s_5m (recuperar a posse em <=5 s E a <=5 m do ponto da pressão).

Replica o is_counterpress do colega (pressão <=5 s após o próprio time perder a bola, ou flag
nativa) na Euro 2020, e compara contrapressão vs pressão regular sob vários alvos
(y_10s, y_5s, y_5s_5m, y_5s_9m), global e no terço ofensivo (x>80) — qui-quadrado + razão de taxas
+ odds da contrapressão numa logística para y_5s_5m. Não altera artigo nem notebook do colega.
Uso: ./venv1/Scripts/python.exe src/xR_experiments/counterpress_target.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import express_data as xd   # importa -> chdir raiz; reusa load_json/ts/DATA_DIR/COMP_ID/SEASON_ID

OUT = 'counterpress_analysis'; os.makedirs(OUT, exist_ok=True)
COMP_ID, SEASON_ID = 55, 43       # UEFA Euro 2020 (mesma base do counterpress_v1 do colega)
WIN10, WIN5 = 10.0, 5.0
TARGETS = ['y_10s', 'y_5s', 'y_5s_5m', 'y_5s_9m']


def process_match(mid):
    ev_data = xd.load_json(xd.DATA_DIR / 'events' / (str(mid) + '.json'))
    if ev_data is None:
        return []
    ev = pd.json_normalize(ev_data, sep='_')
    ev['t'] = ev['timestamp'].apply(xd.ts)
    pr = ev[ev['type_name'] == 'Pressure'].copy()
    if len(pr) == 0:
        return []
    xy = pr['location'].apply(lambda v: v if isinstance(v, list) and len(v) == 2 else [np.nan, np.nan])
    pr['px'] = xy.apply(lambda v: v[0]); pr['py'] = xy.apply(lambda v: v[1])
    pr = pr.dropna(subset=['px', 'py'])

    # recuperacao do time que pressiona (perda da bola pelo portador), com tempo e local
    rec_fail = ev['ball_recovery_recovery_failure'].fillna(False) if 'ball_recovery_recovery_failure' in ev.columns else pd.Series(False, index=ev.index)
    duel_out = ev.get('duel_outcome_name', pd.Series('', index=ev.index)).fillna('')
    rm = ((ev['type_name'].eq('Ball Recovery') & ~rec_fail) | ev['type_name'].eq('Interception')
          | (ev['type_name'].eq('Duel') & duel_out.str.contains('Won|Success', case=False)))
    rec = ev.loc[rm, ['period', 'team_id', 't', 'location']].copy()
    rec['rx'] = rec['location'].apply(lambda v: v[0] if isinstance(v, list) else np.nan)
    rec['ry'] = rec['location'].apply(lambda v: v[1] if isinstance(v, list) else np.nan)
    R = rec[['period', 'team_id', 't', 'rx', 'ry']].rename(columns={'t': 'tr'}).sort_values('tr')
    L = pr[['id', 'period', 'team_id', 't']].rename(columns={'t': 'tp'}).sort_values('tp')
    m = pd.merge_asof(L, R, left_on='tp', right_on='tr', by=['period', 'team_id'],
                      direction='forward', tolerance=WIN10)
    trec = dict(zip(m['id'], m['tr'])); rxy = dict(zip(m['id'], zip(m['rx'], m['ry'])))

    # is_counterpress: perda do PROPRIO time <=5 s antes (backward) ou flag nativa
    pass_out = ev.get('pass_outcome_name', pd.Series(np.nan, index=ev.index))
    loss_mask = (ev['type_name'].isin(['Dispossessed', 'Miscontrol']) |
                 (ev['type_name'].eq('Pass') & pass_out.isin(['Incomplete', 'Out', 'Injury Clearance', 'Unknown'])))
    loss = ev.loc[loss_mask, ['period', 'team_id', 't']].rename(columns={'t': 'tl'}).sort_values('tl')
    ml = pd.merge_asof(L, loss, left_on='tp', right_on='tl', by=['period', 'team_id'],
                       direction='backward', tolerance=WIN5)
    loss_map = dict(zip(ml['id'], ml['tl'].notna()))
    native_cp = ev.get('counterpress', pd.Series(False, index=ev.index)).fillna(False)
    pr['native_cp'] = native_cp.reindex(pr.index).fillna(False)

    out = []
    for _, row in pr.iterrows():
        pid = row['id']; px, py = float(row['px']), float(row['py']); tp = row['t']
        tr = trec.get(pid, np.nan); rx, ry = rxy.get(pid, (np.nan, np.nan))
        has = pd.notna(tr)
        dt = (tr - tp) if has else np.nan
        dist = (np.hypot(rx - px, ry - py) if (has and pd.notna(rx)) else np.nan)
        y10 = int(has)
        y5 = int(has and dt <= 5.0)
        y5_5 = int(y5 and pd.notna(dist) and dist <= 5.0)
        y5_9 = int(y5 and pd.notna(dist) and dist <= 9.0)
        is_cp = int(bool(loss_map.get(pid, False)) or bool(row['native_cp']))
        out.append(dict(x=px, y=py, is_counterpress=is_cp, att_third=int(px > 80.0),
                        y_10s=y10, y_5s=y5, y_5s_5m=y5_5, y_5s_9m=y5_9))
    return out


def compare(df, tgt, scope):
    d = df if scope == 'global' else df[df['att_third'] == 1]
    cp = d[d['is_counterpress'] == 1][tgt]; rg = d[d['is_counterpress'] == 0][tgt]
    a, b = int(cp.sum()), int(len(cp) - cp.sum())
    c, e = int(rg.sum()), int(len(rg) - rg.sum())
    chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, e]])
    rcp = cp.mean() if len(cp) else np.nan; rrg = rg.mean() if len(rg) else np.nan
    return dict(alvo=tgt, escopo=scope, n_CounterPress=len(cp), taxa_CounterPress=round(float(rcp), 4),
                n_reg=len(rg), taxa_reg=round(float(rrg), 4),
                razao=round(float(rcp / rrg), 3) if rrg else np.nan,
                chi2=round(float(chi2), 3), p=float(p), signif=bool(p < 0.05))


def main():
    matches = xd.load_json(xd.DATA_DIR / 'matches' / str(COMP_ID) / (str(SEASON_ID) + '.json'))
    mids = sorted(m['match_id'] for m in matches)
    recs = []
    for j, mid in enumerate(mids):
        recs.extend(process_match(mid))
        if (j + 1) % 17 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(recs)))
    df = pd.DataFrame(recs)
    print('Euro 2020 | pressoes: %d | contrapressoes: %d (%.1f%%)' %
          (len(df), df['is_counterpress'].sum(), 100 * df['is_counterpress'].mean()))
    print('taxas globais:', {t: round(float(df[t].mean()), 4) for t in TARGETS})

    rows = [compare(df, t, s) for t in TARGETS for s in ('global', 'terco_ofensivo')]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'counterpress_vs_regular.csv'), index=False)
    print('\n=== CONTRAPRESSAO vs PRESSAO REGULAR (taxa de recuperacao por alvo) ===')
    print(res.to_string(index=False))

    # odds da contrapressao numa logistica para y_5s_5m (global), controlando por terco ofensivo
    X = sm.add_constant(df[['is_counterpress', 'att_third']].astype(float))
    logit = sm.Logit(df['y_5s_5m'].astype(int), X).fit(disp=0)
    orr = np.exp(logit.params['is_counterpress'])
    ci = np.exp(logit.conf_int().loc['is_counterpress'])
    print('\nLogit y_5s_5m ~ is_counterpress + att_third: OR(contrapressao)=%.3f [%.3f, %.3f] p=%.4g'
          % (orr, ci[0], ci[1], logit.pvalues['is_counterpress']))
    pd.DataFrame([dict(alvo='y_5s_5m', OR_counterpress=round(float(orr), 3),
                       ci_inf=round(float(ci[0]), 3), ci_sup=round(float(ci[1]), 3),
                       p=float(logit.pvalues['is_counterpress']))]
                 ).to_csv(os.path.join(OUT, 'counterpress_logit_y5s5m.csv'), index=False)
    print('\nsalvo em', OUT)


if __name__ == '__main__':
    main()
