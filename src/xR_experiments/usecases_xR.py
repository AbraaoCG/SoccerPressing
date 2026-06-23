# -*- coding: utf-8 -*-
"""usecases_xR.py — exemplos de uso do xR como "historias" (pergunta da comissao -> passo a passo
-> resposta do modelo), para a Secao "Aplicacao pratica" e para slides.

PROTOCOLO HONESTO: a logistica e TREINADA em um conjunto de JOGOS de treino e os exemplos sao
extraidos de JOGOS de TESTE held-out (o modelo nunca viu esses dados). xR = prob. calibrada de
recuperar a bola em <=5 s e a <=5 m (alvo y_5s_5m). Euro 2020.
Saidas: star_analysis/usecases.md (passo a passo) e paper/figs/fig_usecase.png (ilustracao real).
Uso: ./venv1/Scripts/python.exe src/xR_experiments/usecases_xR.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import express_data as xd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
FIGDIR = Path('paper/figs'); FIGDIR.mkdir(parents=True, exist_ok=True)
COMP_ID, SEASON_ID = 55, 43
WIN10 = 10.0
ZONE = {0: 'Defensivo', 1: 'Meio', 2: 'Ataque'}
NAVY, PURPLE, TEAL, TERRA, INK = '#26215C', '#534AB7', '#1D9E75', '#D85A30', '#2C2C2A'


def xy(v):
    return (float(v[0]), float(v[1])) if isinstance(v, list) and len(v) >= 2 else (np.nan, np.nan)


def process_match(mid):
    ev_data = xd.load_json(xd.DATA_DIR / 'events' / (str(mid) + '.json'))
    fr_data = xd.load_json(xd.DATA_DIR / 'three-sixty' / (str(mid) + '.json'))
    if ev_data is None or fr_data is None:
        return []
    ev = pd.json_normalize(ev_data, sep='_'); frames = {fr['event_uuid']: fr for fr in fr_data}
    ev['t'] = ev['timestamp'].apply(xd.ts)
    pr = ev[ev['type_name'] == 'Pressure'].copy()
    if len(pr) == 0:
        return []
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
        x, y = float(ball[0]), float(ball[1]); tp = row['t']
        tr = trec.get(row['id'], np.nan); rx, ry = rxy.get(row['id'], (np.nan, np.nan))
        has = pd.notna(tr); dt = (tr - tp) if has else np.nan
        dist = (np.hypot(rx - x, ry - y) if (has and pd.notna(rx)) else np.nan)
        y5_5 = int(has and dt <= 5.0 and pd.notna(dist) and dist <= 5.0)
        rec_d = {'match_id': int(mid), 'x': x, 'y': y, 'dist_to_goal': float(np.hypot(120 - x, 40 - y)),
                 'zone_code': (0 if x <= 40 else (1 if x <= 80 else 2)),
                 'minute': float(row.get('minute', 0)), 'duration': float(row.get('duration', 0) or 0),
                 'y_5s_5m': y5_5, 'pressing_team': row.get('team_name'),
                 'pressed_team': row.get('possession_team_name')}
        rec_d.update(prim)
        out.append(rec_d)
    return out


def build():
    raw = xd.load_json(xd.DATA_DIR / 'matches' / str(COMP_ID) / (str(SEASON_ID) + '.json'))
    mids = sorted(m['match_id'] for m in raw)
    mname = {m['match_id']: '%s x %s' % (m['home_team']['home_team_name'], m['away_team']['away_team_name']) for m in raw}
    recs = []
    for mid in mids:
        recs.extend(process_match(mid))
    df = pd.DataFrame(recs); df['match'] = df['match_id'].map(mname)
    return df, mids


def step_figure(question, n, team, exp, act, base, clips):
    """Ilustra o passo a passo (dados reais) do exemplo de revisao pos-jogo."""
    fig, ax = plt.subplots(figsize=(8.6, 3.2)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    ax.text(50, 37.5, '“%s”' % question, ha='center', va='center', fontsize=11.5,
            style='italic', color=NAVY, weight='bold')
    boxes = [
        (PURPLE, '1. Predição (xR)', 'Para as %d pressões\nda %s no jogo,\no modelo prevê o xR =\nP(recuperar ≤5 s e ≤5 m)' % (n, team)),
        (TEAL, '2. Agregação', 'Σ xR = %.1f\n(recuperações esperadas)\nvs %d recuperações reais' % (exp, act)),
        (TERRA, '3. Resposta', '%s%.1f acima do esperado.\n3 clipes de maior xR\nque falharam (≈%d–%d%%)\npara revisão' % (
            '+' if act - exp >= 0 else '', act - exp, int(100 * clips[-1]), int(100 * clips[0]))),
    ]
    xs = [4, 37, 70]; w = 26
    for (col, title, body), x0 in zip(boxes, xs):
        ax.add_patch(FancyBboxPatch((x0, 6), w, 22, boxstyle='round,pad=0.6,rounding_size=2',
                                    linewidth=1.6, edgecolor=col, facecolor='white'))
        ax.text(x0 + w / 2, 25, title, ha='center', va='center', fontsize=10.5, weight='bold', color=col)
        ax.text(x0 + w / 2, 14.5, body, ha='center', va='center', fontsize=8.6, color=INK)
    for x0 in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x0 + w + 0.5, 17), (x0 + w + 4.5, 17),
                                     arrowstyle='-|>', mutation_scale=14, color=NAVY, linewidth=1.6))
    fig.tight_layout(); fig.savefig(FIGDIR / 'fig_usecase2.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def main():
    df, mids = build()
    # split por JOGO: 70% treino / 30% teste (held-out). O modelo NAO ve os jogos de teste.
    rng = np.random.default_rng(7)
    test_m = set(rng.choice(mids, size=4, replace=False).tolist())   # 4 jogos held-out (resto p/ treino)
    tr = df[~df['match_id'].isin(test_m)].copy(); te = df[df['match_id'].isin(test_m)].copy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(np.nan_to_num(tr[xd.TAB_FEATURES].to_numpy(np.float32)), tr['y_5s_5m'].to_numpy(np.int32))
    te['xR'] = pipe.predict_proba(np.nan_to_num(te[xd.TAB_FEATURES].to_numpy(np.float32)))[:, 1]
    base = float(te['y_5s_5m'].mean())

    md = ["# Exemplos de uso do xR (Euro 2020) — pergunta da comissão → modelo → resposta\n",
          "**Protocolo:** logística treinada em %d jogos de treino; exemplos extraídos de %d jogos de TESTE "
          "held-out (o modelo nunca viu esses dados). xR = prob. calibrada de recuperar a bola em ≤5 s e a "
          "≤5 m. Taxa-base no teste = %.1f%%.\n" % (len(mids) - len(test_m), len(test_m), 100 * base)]

    # ---- Caso 1: analise de adversario (3 times mais pressionados no teste) ----
    md += ["\n## Caso 1 — Análise de adversário (pré-jogo)\n"]
    for alvo in te['pressed_team'].value_counts().head(3).index:
        sub = te[te['pressed_team'] == alvo]
        zr = sub.groupby('zone_code')['xR'].mean().rename(index=ZONE).sort_values(ascending=False)
        md += ["**Pergunta:** \"Vamos enfrentar a **%s**. Em que zona a nossa pressão tem maior chance de "
               "recuperar a bola?\"  \n  *Predição:* xR por pressão sofrida pela %s (%d pressões); agregado por zona.  \n"
               "  *Resposta:* pressionar no terço **%s** (xR médio **%.1f%%**) — vs %.1f%% no terço de menor risco.\n"
               % (alvo, alvo, len(sub), zr.index[0], 100 * zr.iloc[0], 100 * zr.iloc[-1])]

    # ---- Caso 2: revisao pos-jogo (2 jogos de teste com mais pressoes) ----
    md += ["\n## Caso 2 — Revisão pós-jogo (xR esperado vs realizado)\n"]
    demo = None
    for mid in te['match_id'].value_counts().head(2).index:
        g = te[te['match_id'] == mid]; team = g['pressing_team'].value_counts().index[0]
        gt = g[g['pressing_team'] == team]
        exp, act = float(gt['xR'].sum()), int(gt['y_5s_5m'].sum())
        fails = gt[gt['y_5s_5m'] == 0].nlargest(3, 'xR')
        clips = fails['xR'].tolist()
        md += ["**Pergunta:** \"No jogo **%s**, a pressão do **%s** rendeu o que se esperava?\"  \n"
               "  *Predição:* xR de cada uma das %d pressões do %s.  \n"
               "  *Resposta:* **%.1f** recuperações esperadas (Σ xR) vs **%d** reais (%s%.1f). Maiores xR que "
               "falharam (clipes): %s.\n"
               % (g['match'].iloc[0], team, len(gt), team, exp, act, '+' if act - exp >= 0 else '', act - exp,
                  "; ".join("min %d (%s, xR %.0f%%)" % (r.minute, ZONE[int(r.zone_code)], 100 * r.xR)
                            for r in fails.itertuples()))]
        if demo is None and len(clips) == 3:
            demo = dict(question="No jogo %s, a pressão do %s rendeu o esperado?" % (g['match'].iloc[0], team),
                        n=len(gt), team=team, exp=exp, act=act, clips=clips)

    # ---- Caso 3: triagem (top-k% no teste) ----
    md += ["\n## Caso 3 — Triagem de vídeo / desenho de treino\n",
           "**Pergunta:** \"Temos pouco tempo de análise. Quais situações de pressão revisar primeiro?\"  \n"
           "  *Predição:* xR de todas as %d pressões de teste; ordena por xR.  \n"
           "  *Resposta:* densidade de recuperações e \\emph{lift} (vs acaso %.1f%%) por fração revisada:\n" % (len(te), 100 * base)]
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(frac * len(te))); top = te.nlargest(k, 'xR')
        prec = float(top['y_5s_5m'].mean())
        md += ["  - top %.0f%%: precisão %.1f%% (lift %.1f×), captura %.0f%% das recuperações\n"
               % (100 * frac, 100 * prec, prec / base, 100 * top['y_5s_5m'].sum() / te['y_5s_5m'].sum())]

    open(os.path.join(OUT, 'usecases.md'), 'w', encoding='utf-8').write("\n".join(md))
    if demo:
        step_figure(demo['question'], demo['n'], demo['team'], demo['exp'], demo['act'], base, demo['clips'])
    print('treino %d jogos / teste %d jogos | base teste %.3f' % (len(mids) - len(test_m), len(test_m), base))
    print('salvo: %s e %s' % (os.path.join(OUT, 'usecases.md'), FIGDIR / 'fig_usecase2.png'))


if __name__ == '__main__':
    main()
