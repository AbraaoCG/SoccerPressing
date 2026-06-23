# -*- coding: utf-8 -*-
"""Abordagem C — Contrafactual de decisao [basquete EPV-added/decision eval], SO FIFA WC 2022.
Modela P(reter posse | opcao) e compara a opcao escolhida com a melhor disponivel (regret).
Snapshot-only -> versao MVP (so passes observados), limitacoes documentadas.
Saidas: star_analysis/C_decision.csv, C_player_rating.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'express'))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr
import star_data as sd
from gbt_util import make_gbt, fit_gbt

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)


def seg_dist(p, a, b):
    """distancia ponto p ao segmento a-b (a=carrier, b=destino), p = adversarios (Nx2)."""
    ab = b - a; L2 = float(ab @ ab)
    if L2 < 1e-6:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip((p - a) @ ab / L2, 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def opt_feats(c, d, advs):
    """features de uma opcao (destino d) a partir do portador c, dado adversarios advs (pitch)."""
    prog = d[0] - c[0]                                   # avanco em x (frente = gol em 120)
    dist_cd = float(np.hypot(*(d - c)))
    if len(advs):
        dadv = np.linalg.norm(advs - d, axis=1)
        recv_press = float(dadv.min()); n_adv5 = int((dadv <= 5).sum())
        lane_block = float(seg_dist(advs, c, d).min())
    else:
        recv_press = 30.0; n_adv5 = 0; lane_block = 30.0
    return [prog, dist_cd, recv_press, lane_block, n_adv5]


COLS = ['prog', 'dist_cd', 'recv_press', 'lane_block', 'n_adv5']


def main():
    df = sd.build().reset_index(drop=True)
    # ---- treina P(reter | opcao) nos passes observados ----
    mask = (df['act_type'] == 'Pass') & df['act_end_x'].notna()
    obs = df[mask]
    X, y, who, mid = [], [], [], []
    for _, r in obs.iterrows():
        c = np.array([r['carrier_x'], r['carrier_y']]); d = np.array([r['act_end_x'], r['act_end_y']])
        X.append(opt_feats(c, d, r['advs'])); y.append(int(1 - r['y_5s']))      # reter = nao perdeu em 5s
        who.append(r['carrier_id']); mid.append(r['match_id'])
    X = np.array(X, np.float32); y = np.array(y, np.int32)
    print('passes observados:', len(y), '| taxa retencao %.3f' % y.mean())
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    tr2, va = train_test_split(tr, test_size=0.20, random_state=42, stratify=y[tr])
    model, kind = make_gbt(); fit_gbt(model, kind, X[tr2], y[tr2], X[va], y[va])
    p_te = model.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p_te); acc = accuracy_score(y[te], (p_te >= 0.5).astype(int))
    print('modelo de opcao (%s): AUC(reter)=%.4f | acc=%.4f' % (kind, auc, acc))

    # ---- regret = melhor opcao - opcao escolhida (em todas as pressoes com passe + companheiros) ----
    regrets, who2, mid2 = [], [], []
    for _, r in obs.iterrows():
        c = np.array([r['carrier_x'], r['carrier_y']]); advs = r['advs']; mates = r['mates']
        cands = []
        for m in mates:
            cands.append(opt_feats(c, np.asarray(m, float), advs))
        cands.append(opt_feats(c, c + np.array([5.0, 0.0]), advs))            # opcao "conduzir p/ frente"
        if not cands:
            continue
        preds = model.predict_proba(np.array(cands, np.float32))[:, 1]
        best = float(preds.max())
        chosen = float(model.predict_proba(np.array([opt_feats(c, np.array([r['act_end_x'], r['act_end_y']]), advs)], np.float32))[:, 1][0])
        regrets.append(max(best - chosen, 0.0)); who2.append(r['carrier_id']); mid2.append(r['match_id'])
    reg = pd.DataFrame({'carrier_id': who2, 'match_id': mid2, 'regret': regrets}).dropna(subset=['carrier_id'])
    print('regret medio: %.4f (0 = escolheu sempre a melhor opcao do modelo)' % reg['regret'].mean())

    # ---- confiabilidade split-half do regret por jogador ----
    mids = np.sort(reg['match_id'].unique()); h1 = set(mids[::2])
    reg['half'] = np.where(reg['match_id'].isin(h1), 0, 1)
    rate = reg.groupby(['carrier_id', 'half'])['regret'].mean().unstack()
    cnt = reg.groupby(['carrier_id', 'half']).size().unstack()
    keep = (cnt[0] >= 10) & (cnt[1] >= 10)
    rr = rate[keep].dropna()
    rel, _ = spearmanr(rr[0], rr[1]) if len(rr) > 5 else (np.nan, np.nan)
    print('confiabilidade split-half (Spearman) do regret: %.3f | jogadores: %d' % (rel, len(rr)))

    pd.DataFrame([dict(abordagem='C', modelo='retencao por opcao (GBT)', alvo='reter|opcao',
                       auc_opcao=round(auc, 4), acc=round(acc, 4), regret_medio=round(float(reg['regret'].mean()), 4),
                       reliab_splithalf=round(float(rel), 4), n_jogadores_rel=len(rr))]
                 ).to_csv(os.path.join(OUT, 'C_decision.csv'), index=False)
    rating = reg.groupby('carrier_id').agg(n=('regret', 'size'), regret=('regret', 'mean')).reset_index()
    rating[rating['n'] >= 20].sort_values('regret').to_csv(os.path.join(OUT, 'C_player_rating.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'C_decision.csv'), 'e C_player_rating.csv')


if __name__ == '__main__':
    main()
