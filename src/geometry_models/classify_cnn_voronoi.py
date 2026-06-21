# -*- coding: utf-8 -*-
"""classify_cnn_voronoi.py
CNN para prever recuperação da bola (recovered, <=10s) a partir da geometria dos adversários.
Seleção dos adversários = VIZINHOS DE VORONOI do portador (cells adjacentes ao cell do portador),
em vez de um raio fixo em metros. Dataset: UEFA Euro 2020 (comp 55, season 43). Só CNN.

Execução:  ./venv1/Scripts/python.exe classify_cnn_voronoi.py
Env vars:  EPOCHS (default 30); SMOKE=1 (poucos jogos + 1 época, teste rápido).
"""
import os as _os; from pathlib import Path as _Path
_os.chdir(str(_Path(__file__).resolve().parent.parent.parent))  # roda a partir da raiz do projeto
import json, os, math, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
np.random.seed(42)
from scipy.spatial import Voronoi
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
torch.manual_seed(42)

DATA_DIR = Path('StatsBomb_2/data')
COMP_ID, SEASON_ID = 55, 43          # UEFA Euro 2020
RECOVERY_WINDOW = 10.0
G, HALF, SIG = 48, 30.0, 3.0         # raster: grade 48x48, extensao +-30 m, sigma do bump
OUT = 'cnn_voronoi_analysis'; os.makedirs(OUT, exist_ok=True)
os.makedirs('cache', exist_ok=True)
META_PKL, IMG_NPZ = 'cache/_vor_meta.pkl', 'cache/_vor_img.npz'
EPOCHS = int(os.environ.get('EPOCHS', '30'))
SMOKE = os.environ.get('SMOKE') == '1'


def load_json(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        print('[WARN] falha ao ler %s: %s' % (p, e)); return None


def ts(t):
    hh, mm, ss = t.split(':'); return int(hh) * 3600 + int(mm) * 60 + float(ss)


def voronoi_adv_neighbors(players, is_adv, carrier_idx):
    """Indices dos adversarios cujo cell de Voronoi e adjacente ao cell do portador."""
    if len(players) < 4:
        return None
    try:
        vor = Voronoi(players)
    except Exception:
        return None
    nb = set()
    for a, b in vor.ridge_points:
        if a == carrier_idx:
            nb.add(int(b))
        elif b == carrier_idx:
            nb.add(int(a))
    return [j for j in nb if is_adv[j]]


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
    rec_fail = ev['ball_recovery_recovery_failure'].fillna(False) if 'ball_recovery_recovery_failure' in ev.columns else pd.Series(False, index=ev.index)
    duel_out = ev.get('duel_outcome_name', pd.Series('', index=ev.index)).fillna('')
    rm = ((ev['type_name'].eq('Ball Recovery') & ~rec_fail) | ev['type_name'].eq('Interception')
          | (ev['type_name'].eq('Duel') & duel_out.str.contains('Won|Success')))
    rec = ev.loc[rm, ['period', 'team_id', 't']]
    L = pr[['id', 'period', 'team_id', 't']].rename(columns={'t': 'tp'}).sort_values('tp')
    R = rec.rename(columns={'t': 'tr'}).sort_values('tr')
    m = pd.merge_asof(L, R, left_on='tp', right_on='tr', by=['period', 'team_id'],
                      direction='forward', tolerance=RECOVERY_WINDOW)
    recov = dict(zip(m['id'], m['tr'].notna().astype(int)))
    out = []
    for _, row in pr.iterrows():
        fr = frames.get(row['id']); loc = row['location']
        if fr is None or not isinstance(loc, list):
            continue
        ball = np.array(loc[:2], float)
        locs, isadv = [], []
        for p in fr['freeze_frame']:
            locs.append(p['location'][:2]); isadv.append(bool(p.get('teammate')))
        players = np.array(locs, float); isadv = np.array(isadv)
        nonadv = np.where(~isadv)[0]
        if len(nonadv) == 0 or isadv.sum() == 0:
            continue
        carrier_idx = int(nonadv[np.argmin(np.linalg.norm(players[nonadv] - ball, axis=1))])
        carrier = players[carrier_idx]
        nb = voronoi_adv_neighbors(players, isadv, carrier_idx)
        vor_ok = nb is not None
        # adversarios vizinhos, reorientados (portador na origem, frente = +X)
        if vor_ok and len(nb):
            d = players[nb] - carrier
            pts = np.column_stack([-d[:, 0], d[:, 1]]).astype(np.float32)
        else:
            pts = np.zeros((0, 2), np.float32)
        x, y = float(ball[0]), float(ball[1])
        rec_d = {'recovered': int(recov.get(row['id'], 0)), 'x': x, 'y': y,
                 'dist_to_goal': math.sqrt((120 - x) ** 2 + (40 - y) ** 2),
                 'n_adv_nb': int(len(pts)), 'vor_ok': int(vor_ok)}
        out.append((rec_d, pts))
    return out


def build():
    if os.path.exists(META_PKL) and os.path.exists(IMG_NPZ):
        meta = pd.read_pickle(META_PKL)
        z = np.load(IMG_NPZ)
        return meta, z['IMG'], z['y']
    matches = load_json(DATA_DIR / 'matches' / str(COMP_ID) / (str(SEASON_ID) + '.json'))
    mids = sorted(m['match_id'] for m in matches)
    if SMOKE:
        mids = mids[:6]
    recs, pts_l = [], []
    for j, mid in enumerate(mids):
        for rec_d, pts in process_match(mid):
            recs.append(rec_d); pts_l.append(pts)
        if (j + 1) % 10 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(recs)))
    meta = pd.DataFrame(recs)
    N = len(meta)
    xs = np.linspace(-HALF, HALF, G); gx, gy = np.meshgrid(xs, xs)
    IMG = np.zeros((N, 1, G, G), np.float32)
    for i in range(N):
        for (px, py) in pts_l[i]:
            IMG[i, 0] += np.exp(-(((gx - px) ** 2 + (gy - py) ** 2) / (2 * SIG ** 2)))
    y = meta['recovered'].to_numpy().astype(np.float32)
    if not SMOKE:
        meta.to_pickle(META_PKL); np.savez_compressed(IMG_NPZ, IMG=IMG, y=y)
    return meta, IMG, y


class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(2))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(s, x):
        return s.head(s.c(x)).squeeze(-1)


def train_eval(IMG, y, epochs, verbose=True):
    pos = np.arange(len(y))
    tr, te = train_test_split(pos, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    IMGt = torch.tensor(IMG); yt = torch.tensor(y)
    model = CNN(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    if verbose:
        print(f"Splits -> train={len(tr2)} | val={len(va)} | test={len(te)}")
        print(f"Batches por época: {int(np.ceil(len(tr2) / 256))}")
        print("-" * 72)

    def fwd(idx, aug):
        ib = IMGt[torch.as_tensor(idx, dtype=torch.long)]
        if aug and np.random.rand() < 0.5:
            ib = torch.flip(ib, dims=[2])
        return model(ib)

    best, best_state, best_ep = -1, None, -1
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr2)
        running_loss, n_batches = 0.0, 0
        for sx in range(0, len(perm), 256):
            b = perm[sx:sx + 256]; opt.zero_grad()
            loss = lossf(fwd(b, True), yt[torch.as_tensor(b, dtype=torch.long)])
            loss.backward(); opt.step()
            running_loss += loss.item(); n_batches += 1

        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y[va], torch.sigmoid(fwd(va, False)).numpy())

        improved = auc > best
        if improved:
            best = auc; best_ep = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            tag = "  <-- best" if improved else ""
            print(f"Epoch {ep+1:03d}/{epochs} | train_loss={running_loss/n_batches:.4f} "
                  f"| val_auc={auc:.4f}{tag}")

    if verbose:
        print("-" * 72)
        print(f"Best val_auc={best:.4f} na época {best_ep+1}. Avaliando no teste...")

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(fwd(te, False)).numpy()
    auc_te = roc_auc_score(y[te], p_te)
    acc_te = accuracy_score(y[te], (p_te >= 0.5).astype(int))

    if verbose:
        print(f"Teste -> AUC={auc_te:.4f} | ACC={acc_te:.4f}")

    return auc_te, acc_te, len(tr2), len(va), len(te)


def main():
    meta, IMG, y = build()
    N = len(meta)
    print('pressoes (Euro 2020):', N, '| taxa recuperacao %.3f' % y.mean())
    print('Voronoi valido: %.1f%% | com >=1 vizinho adversario: %.1f%% | media vizinhos: %.2f'
          % (100 * meta['vor_ok'].mean(), 100 * (meta['n_adv_nb'] >= 1).mean(), meta['n_adv_nb'].mean()))
    ep = 1 if SMOKE else EPOCHS
    auc, acc, ntr, nva, nte = train_eval(IMG, y, ep)
    print('AUC teste = %.4f | acuracia = %.4f' % (auc, acc))
    pd.DataFrame([dict(dataset='Euro2020', selecao='vizinhos_voronoi', alvo='recovered',
                       n_amostras=N, taxa_recuperacao=round(float(y.mean()), 4),
                       auc_teste=round(auc, 4), acc_teste=round(acc, 4),
                       media_vizinhos_adv=round(float(meta['n_adv_nb'].mean()), 3),
                       frac_voronoi_ok=round(float(meta['vor_ok'].mean()), 4),
                       frac_ge1_vizinho=round(float((meta['n_adv_nb'] >= 1).mean()), 4),
                       epochs=ep, n_train=ntr, n_val=nva, n_test=nte)]).to_csv(os.path.join(OUT, 'result.csv'), index=False)
    meta['n_adv_nb'].value_counts().sort_index().to_csv(os.path.join(OUT, 'neighbor_distribution.csv'))
    print('salvo em', os.path.join(OUT, 'result.csv'))
    print('OK')


if __name__ == '__main__':
    main()
