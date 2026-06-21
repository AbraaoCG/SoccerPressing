# -*- coding: utf-8 -*-
"""Ponte Python <-> C++ para os modelos GNN/CNN (Euro 2020, alvo = recovered).

Uso:
  python algorithms/bridge.py prepare [--smoke]   # carrega dados, monta arrays, exporta io/*.bin
  python algorithms/bridge.py build               # compila train.cpp -> train.exe (g++/Eigen)
  python algorithms/bridge.py run --model gnn|cnn # roda o exe e reporta AUC/acuracia

Rode com o Python do venv1 (tem pandas/scipy/sklearn). O C++ usa g++ (MinGW64) + Eigen.
"""
import json, os, sys, math, argparse, subprocess
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
ALGO = Path(__file__).resolve().parent
DATA_DIR = PROJ / 'StatsBomb_2' / 'data'
IO = ALGO / 'io'; IO.mkdir(exist_ok=True)
CACHE = PROJ / 'cache' / '_cpp_data.npz'
CACHE_VOR = PROJ / 'cache' / '_cpp_data_voronoi.npz'
COMP_ID, SEASON_ID = 55, 43
RECOVERY_WINDOW = 10.0
LOCAL_R, NEIGH_R, KADV = 30.0, 10.0, 14
G, HALF, SIG = 48, 30.0, 3.0


# ---------- formato binario (compartilhado com nn.hpp) ----------
def write_tensor(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, 'wb') as f:
        np.array([arr.ndim], dtype=np.int32).tofile(f)
        np.array(arr.shape, dtype=np.int32).tofile(f)
        arr.tofile(f)


def read_tensor(path):
    with open(path, 'rb') as f:
        ndim = int(np.fromfile(f, np.int32, 1)[0])
        dims = np.fromfile(f, np.int32, ndim)
        data = np.fromfile(f, np.float32)
    return data.reshape(dims)


# ---------- preparacao dos dados (Euro 2020) ----------
def load_json(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def ts(t):
    hh, mm, ss = t.split(':'); return int(hh) * 3600 + int(mm) * 60 + float(ss)


def voronoi_neighbors_and_ridges(players, is_adv, carrier_idx):
    """Vizinhos de Voronoi do portador (cells adjacentes) + arestas adv-adv do grafo de Voronoi.

    Retorna (nb, ridge_set) onde:
      - nb        = indices globais dos ADVERSARIOS com cell adjacente a do portador;
      - ridge_set = conjunto de pares (frozenset {a,b}) de TODAS as arestas de Voronoi (vor.ridge_points),
                    usado para montar o subgrafo entre os nos selecionados.
    Retorna (None, None) para Voronoi degenerado (poucos pontos / colinear).
    """
    from scipy.spatial import Voronoi
    if len(players) < 4:
        return None, None
    try:
        vor = Voronoi(players)
    except Exception:
        return None, None
    nb = set()
    ridge_set = set()
    for a, b in vor.ridge_points:
        a, b = int(a), int(b)
        ridge_set.add(frozenset((a, b)))
        if a == carrier_idx and is_adv[b]:
            nb.add(b)
        elif b == carrier_idx and is_adv[a]:
            nb.add(a)
    return sorted(nb), ridge_set


def process_match(mid, select='within30'):
    import pandas as pd
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
        rec = int(recov.get(row['id'], 0))
        if select == 'voronoi':
            res = _press_voronoi(fr, ball)
        else:
            res = _press_within30(fr, ball)
        if res is None:
            continue
        pts, adj = res                                          # pts:[k,2] reorientado; adj:[k,k] ou None
        out.append((rec, pts.astype(np.float32), adj))
    return out


def _press_within30(fr, ball):
    """Adversarios a <=30 m do portador, reorientados (frente=+X). Adjacencia por distancia (adj=None)."""
    advs, sups = [], []
    for p in fr['freeze_frame']:
        (advs if p.get('teammate') else sups).append(p['location'][:2])
    if len(sups) == 0 or len(advs) == 0:
        return None
    sups = np.array(sups, float); advs = np.array(advs, float)
    carrier = sups[int(np.argmin(np.linalg.norm(sups - ball, axis=1)))]
    d = advs - carrier
    adv_r = np.column_stack([-d[:, 0], d[:, 1]])               # frente = +X
    adv_r = adv_r[np.linalg.norm(adv_r, axis=1) <= LOCAL_R]
    if len(adv_r) < 1:
        return None
    return adv_r, None


def _press_voronoi(fr, ball):
    """Vizinhos de Voronoi (adversarios) do portador, reorientados; adjacencia = subgrafo de Voronoi.

    Portador degenerado / Voronoi invalido -> None (amostra pulada).
    0 vizinhos -> amostra valida, grafo vazio (pts vazio, adj 0x0).
    """
    locs, isadv = [], []
    for p in fr['freeze_frame']:
        locs.append(p['location'][:2]); isadv.append(bool(p.get('teammate')))
    players = np.array(locs, float); isadv = np.array(isadv)
    nonadv = np.where(~isadv)[0]
    if len(nonadv) == 0 or isadv.sum() == 0:
        return None
    carrier_idx = int(nonadv[int(np.argmin(np.linalg.norm(players[nonadv] - ball, axis=1)))])
    carrier = players[carrier_idx]
    nb, ridge_set = voronoi_neighbors_and_ridges(players, isadv, carrier_idx)
    if nb is None:                                             # Voronoi degenerado
        return None
    if len(nb) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 0), np.float32)
    d = players[nb] - carrier
    pts = np.column_stack([-d[:, 0], d[:, 1]])                 # frente = +X
    k = len(nb)
    adj = np.zeros((k, k), np.float32)                         # arestas adv-adv via subgrafo de Voronoi
    for i in range(k):
        for j in range(i + 1, k):
            if frozenset((nb[i], nb[j])) in ridge_set:
                adj[i, j] = adj[j, i] = 1.0
    return pts, adj


def prepare(smoke=False, select='within30'):
    matches = load_json(DATA_DIR / 'matches' / str(COMP_ID) / (str(SEASON_ID) + '.json'))
    mids = sorted(m['match_id'] for m in matches)
    if smoke:
        mids = mids[:6]
    ys, pts_l, adj_l = [], [], []
    for j, mid in enumerate(mids):
        for rec, pts, adj in process_match(mid, select):
            ys.append(rec); pts_l.append(pts); adj_l.append(adj)
        if (j + 1) % 10 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(ys)))
    N = len(ys)
    Xp = np.zeros((N, KADV, 3), np.float32); A = np.zeros((N, KADV, KADV), np.float32)
    M = np.zeros((N, KADV), np.float32); IMG = np.zeros((N, 1, G, G), np.float32)
    xs = np.linspace(-HALF, HALF, G); gx, gy = np.meshgrid(xs, xs)
    for i, (pts, adj) in enumerate(zip(pts_l, adj_l)):
        order = np.argsort(np.linalg.norm(pts, axis=1))[:KADV]
        p = pts[order]; k = len(p)
        if k == 0:                                              # voronoi sem vizinhos: grafo/imagem vazios
            continue
        dist = np.linalg.norm(p, axis=1)
        Xp[i, :k] = np.column_stack([p[:, 0] / 30.0, p[:, 1] / 30.0, dist / 30.0]); M[i, :k] = 1.0
        if adj is None:                                         # within30: arestas adv-adv por distancia
            Dm = np.linalg.norm(p[:, None] - p[None], axis=2)
            a = (Dm <= NEIGH_R).astype(float)
        else:                                                   # voronoi: subgrafo de Voronoi (alinhado por order)
            a = adj[np.ix_(order, order)].astype(float)
        np.fill_diagonal(a, 1.0)
        dinv = 1.0 / np.sqrt(a.sum(1, keepdims=True)); A[i, :k, :k] = a * dinv * dinv.T
        for (px, py) in p:
            IMG[i, 0] += np.exp(-(((gx - px) ** 2 + (gy - py) ** 2) / (2 * SIG ** 2)))
    y = np.array(ys, np.float32)
    nz = M.sum(1) > 0
    if not smoke:
        np.savez_compressed(CACHE_VOR if select == 'voronoi' else CACHE,
                            Xp=Xp, A=A, M=M, IMG=IMG, y=y)
    print('selecao: %s | pressoes: %d | taxa recuperacao %.3f | media vizinhos %.2f | com >=1 vizinho %.1f%%'
          % (select, N, y.mean(), M.sum(1).mean(), 100.0 * nz.mean()))
    export(Xp, A, M, IMG, y)


def export(Xp, A, M, IMG, y):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    write_tensor(IO / 'cnn_Xtr.bin', IMG[tr]); write_tensor(IO / 'cnn_Xte.bin', IMG[te])
    write_tensor(IO / 'gnn_Xtr.bin', Xp[tr]); write_tensor(IO / 'gnn_Xte.bin', Xp[te])
    write_tensor(IO / 'gnn_Atr.bin', A[tr]); write_tensor(IO / 'gnn_Ate.bin', A[te])
    write_tensor(IO / 'gnn_Mtr.bin', M[tr]); write_tensor(IO / 'gnn_Mte.bin', M[te])
    write_tensor(IO / 'ytr.bin', y[tr]); write_tensor(IO / 'yte.bin', y[te])
    np.save(IO / 'yte.npy', y[te])
    print('exportado para', IO, '| treino', len(tr), '| teste', len(te))


def build():
    eigen = os.environ.get('EIGEN_INC', '/mingw64/include/eigen3')
    cmd = ['g++', '-O3', '-march=native', '-std=c++17', '-I', eigen,
           str(ALGO / 'train.cpp'), '-o', str(ALGO / 'train.exe')]
    print('compilando:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print('OK -> train.exe')


def run(model):
    from sklearn.metrics import roc_auc_score, accuracy_score
    exe = ALGO / ('train.exe' if os.name == 'nt' else 'train')
    subprocess.run([str(exe), '--model', model, '--io', str(IO)], check=True)
    pred = read_tensor(IO / ('pred_%s.bin' % model))
    yte = np.load(IO / 'yte.npy')
    print('%s -> AUC %.4f | acuracia %.4f' % (model, roc_auc_score(yte, pred),
                                              accuracy_score(yte, (pred >= 0.5).astype(int))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'build', 'run'])
    ap.add_argument('--model', choices=['gnn', 'cnn'], default='cnn')
    ap.add_argument('--select', choices=['within30', 'voronoi'], default='within30')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare(a.smoke, a.select)
    elif a.cmd == 'build':
        build()
    else:
        run(a.model)
