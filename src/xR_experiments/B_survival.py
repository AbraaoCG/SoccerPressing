# -*- coding: utf-8 -*-
"""Abordagem B — Survival ponderado por valor (xBLV) [NFL ball-security x VAEP/xT], SO FIFA WC 2022.
(1) hazard de tempo discreto da perda de posse;  (2) xT-lite pesa a perda -> xBLV;
(3) metrica de portador = valor perdido - esperado, com confiabilidade split-half.
Saidas: star_analysis/B_survival.csv, B_player_rating.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'xR_experiments'))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr
import star_data as sd
from gbt_util import make_gbt, fit_gbt
from xt_lite import build_xt, xt_at

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
NBIN = 10                       # bins de 1 s, horizonte 10 s
FEATS = sd.TAB_FEATURES


def expand_person_period(df, idx):
    """Cada pressao -> linhas (bin) ate o evento/censura. event=perdeu naquele bin."""
    rows_X, rows_b, rows_y = [], [], []
    Xall = df[FEATS].to_numpy(np.float32)
    tloss = df['t_loss'].to_numpy(); evl = df['ev_loss'].to_numpy()
    for i in idx:
        be = int(np.ceil(tloss[i])) if evl[i] else NBIN     # bin do evento (ou censura em 10)
        be = max(1, min(be, NBIN))
        for b in range(1, be + 1):
            rows_X.append(Xall[i]); rows_b.append(b)
            rows_y.append(1 if (evl[i] and b == be) else 0)
    Xp = np.column_stack([np.array(rows_X, np.float32), np.array(rows_b, np.float32)])
    return Xp, np.array(rows_y, np.int32)


def p_lose_5s(model, kind, df, idx):
    """P(perder <=5s) = 1 - prod_{b=1..5}(1 - hazard_b)."""
    Xall = df[FEATS].to_numpy(np.float32)[idx]
    surv = np.ones(len(idx))
    for b in range(1, 6):
        Xb = np.column_stack([Xall, np.full(len(idx), b, np.float32)])
        h = model.predict_proba(Xb)[:, 1]
        surv *= (1 - h)
    return 1 - surv


def main():
    if os.environ.get('STAR_DATA') == 'all':
        df = pd.read_pickle('cache/_star_all.pkl').reset_index(drop=True)
        print('FONTE: todas as competicoes (%d pressoes)' % len(df))
    else:
        df = sd.build().reset_index(drop=True)
        print('FONTE: FIFA WC 2022 (%d pressoes)' % len(df))
    xt = build_xt()

    # ---- (1)(2) hazard + AUC head-to-head (compara com A no mesmo y_5s) ----
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, random_state=42,
                              stratify=df['y_5s'].to_numpy())
    Xtr, ytr = expand_person_period(df, tr)
    # split de validacao dentro do treino (por linha) p/ early stopping
    rtr, rva = train_test_split(np.arange(len(ytr)), test_size=0.15, random_state=42, stratify=ytr)
    model, kind = make_gbt(); fit_gbt(model, kind, Xtr[rtr], ytr[rtr], Xtr[rva], ytr[rva])
    p5_te = p_lose_5s(model, kind, df, te)
    auc = roc_auc_score(df['y_5s'].to_numpy()[te], p5_te)
    acc = accuracy_score(df['y_5s'].to_numpy()[te], (p5_te >= 0.5).astype(int))
    print('hazard discreto (%s): AUC(y_5s)=%.4f | acc=%.4f' % (kind, auc, acc))

    # ---- (3) xBLV + metrica de jogador (modelo reajustado em todos p/ rating descritivo) ----
    Xall_pp, yall_pp = expand_person_period(df, idx)
    rtr, rva = train_test_split(np.arange(len(yall_pp)), test_size=0.15, random_state=1, stratify=yall_pp)
    mfull, kfull = make_gbt(); fit_gbt(mfull, kfull, Xall_pp[rtr], yall_pp[rtr], Xall_pp[rva], yall_pp[rva])
    p5 = p_lose_5s(mfull, kfull, df, idx)

    x = df['x'].to_numpy(); y = df['y'].to_numpy()
    lx = df['loss_x'].to_numpy(); ly = df['loss_y'].to_numpy(); y5 = df['y_5s'].to_numpy()
    # ameaca entregue ao adversario: esperada (bola refletida p/ frame do oponente) e observada (local da perda)
    xt_exp = np.array([xt_at(xt, 120 - x[i], 80 - y[i]) for i in idx])
    xt_obs = np.array([xt_at(xt, lx[i], ly[i]) if (y5[i] and lx[i] == lx[i]) else xt_exp[i] for i in idx])
    df['xBLV'] = p5 * xt_exp                       # valor esperado da perda
    df['val_perdido'] = y5 * xt_obs                # valor realmente perdido
    df['vplus'] = df['val_perdido'] - df['xBLV']   # acima/abaixo do esperado (neg = bom retentor)

    # ---- confiabilidade split-half (por jogador, metades de jogos) ----
    d = df[df['carrier_id'].notna()].copy()
    mids = np.sort(d['match_id'].unique())
    h1 = set(mids[::2]);
    d['half'] = np.where(d['match_id'].isin(h1), 0, 1)
    rate = d.groupby(['carrier_id', 'half'])['vplus'].mean().unstack()
    cnt = d.groupby(['carrier_id', 'half']).size().unstack()
    keep = (cnt[0] >= 10) & (cnt[1] >= 10)
    r = rate[keep].dropna()
    rel, _ = spearmanr(r[0], r[1]) if len(r) > 5 else (np.nan, np.nan)
    print('confiabilidade split-half (Spearman) do rating xBLV: %.3f | jogadores: %d' % (rel, len(r)))

    pd.DataFrame([dict(abordagem='B', modelo='hazard discreto + xT', alvo='y_5s',
                       auc_contexto=round(auc, 4), acc=round(acc, 4),
                       reliab_splithalf=round(float(rel), 4), n_jogadores_rel=len(r))]
                 ).to_csv(os.path.join(OUT, 'B_survival.csv'), index=False)
    rating = (d.groupby('carrier_id')
                .agg(n=('vplus', 'size'), vplus=('vplus', 'mean'),
                     xBLV=('xBLV', 'mean'), val_perdido=('val_perdido', 'mean')).reset_index())
    rating[rating['n'] >= 20].sort_values('vplus').to_csv(os.path.join(OUT, 'B_player_rating.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'B_survival.csv'), 'e B_player_rating.csv')
    print('melhores retentores (vplus mais negativo, >=20 pressoes):')
    print(rating[rating['n'] >= 20].sort_values('vplus').head(5)[['carrier_id', 'n', 'vplus']].to_string(index=False))


if __name__ == '__main__':
    main()
