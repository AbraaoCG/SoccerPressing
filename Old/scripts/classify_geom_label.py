# -*- coding: utf-8 -*-
"""classify_geom_label.py
Alvo = ROTULO da geometria (tipologia por regras, recalculada por raio).
Compara GNN x 3 CNNs maiores, em 3 raios (10/20/30 m). Euro 2020.
Grade 3 raios x 4 modelos = 12 execucoes. Saidas em geom_label_analysis/.

Env vars: EPOCHS (default 25); SMOKE=1 (subamostra + 1 epoca, para teste rapido).
"""
import os as _os; from pathlib import Path as _Path
_os.chdir(str(_Path(__file__).resolve().parent.parent))  # roda a partir da raiz do projeto
import json, os, math, glob, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
torch.manual_seed(42)

DATA_DIR = Path('StatsBomb_2/data')
COMP_ID, SEASON_ID = 55, 43
LOCAL_R = 30.0
RADII = [10.0, 20.0, 30.0]
KMAX = 20
G = 48
CLASSES = ['Livre', 'Cercado', 'Frente_livre', 'Frente_bloqueada']
CIDX = {c: i for i, c in enumerate(CLASSES)}
OUT = 'geom_label_analysis'; os.makedirs(OUT, exist_ok=True)
ADV_PKL = 'cache/_advlist_euro.pkl'
os.makedirs('cache', exist_ok=True)
EPOCHS = int(os.environ.get('EPOCHS', '25'))
SMOKE = os.environ.get('SMOKE') == '1'


def load_json(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def ts(t):
    hh, mm, ss = t.split(':'); return int(hh) * 3600 + int(mm) * 60 + float(ss)


def build_adv_list():
    """Para cada pressao (Euro 2020) com >=1 adversario em 30m: array de adversarios reorientados
    (frente=+X) em torno do portador na origem."""
    if os.path.exists(ADV_PKL):
        with open(ADV_PKL, 'rb') as fh:
            return pickle.load(fh)
    matches = load_json(DATA_DIR / 'matches' / str(COMP_ID) / (str(SEASON_ID) + '.json'))
    mids = sorted(m['match_id'] for m in matches)
    adv_list = []
    for mid in mids:
        ev = pd.json_normalize(load_json(DATA_DIR / 'events' / (str(mid) + '.json')), sep='_')
        frames = {fr['event_uuid']: fr for fr in load_json(DATA_DIR / 'three-sixty' / (str(mid) + '.json'))}
        pr = ev[ev['type_name'] == 'Pressure']
        for _, row in pr.iterrows():
            fr = frames.get(row['id']); loc = row['location']
            if fr is None or not isinstance(loc, list):
                continue
            ball = np.array(loc[:2], float)
            advs, sups = [], []
            for p in fr['freeze_frame']:
                xyp = np.array(p['location'], float)
                (advs if p.get('teammate') else sups).append(xyp)
            if len(sups) == 0 or len(advs) == 0:
                continue
            sups = np.array(sups); advs = np.array(advs)
            carrier = sups[int(np.argmin(np.linalg.norm(sups - ball, axis=1)))]
            d = advs - carrier
            adv_r = np.column_stack([-d[:, 0], d[:, 1]])
            dist = np.linalg.norm(adv_r, axis=1)
            adv_r = adv_r[dist <= LOCAL_R]
            if len(adv_r) < 1:
                continue
            adv_list.append(adv_r.astype(np.float32))
    with open(ADV_PKL, 'wb') as fh:
        pickle.dump(adv_list, fh)
    return adv_list


def free_stats(pts):
    """Maior vao angular livre e se o vao aponta para a frente (+X)."""
    if len(pts) == 0:
        return 360.0, 1
    ang = sorted(((np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) + 360) % 360).tolist())
    if len(ang) == 1:
        gap = 360.0; mid = (ang[0] + 180) % 360
    else:
        gaps = [((ang[(i + 1) % len(ang)] - ang[i]) % 360, i) for i in range(len(ang))]
        gap, i = max(gaps); mid = (ang[i] + gap / 2) % 360
    return gap, int(math.cos(math.radians(mid)) > 0)


def build_radius(adv_list, R):
    """Retorna ylab(int), Xp, A, Mk, IMG para o raio R; rotulo por regras recalculado em R."""
    N = len(adv_list)
    sub = [a[np.linalg.norm(a, axis=1) <= R] for a in adv_list]
    # densidade local continua (adversarios mais proximos pesam mais) -> calibra por quantis no raio R
    dens = np.array([float(np.sum(1.0 - np.linalg.norm(p, axis=1) / R)) if len(p) else 0.0 for p in sub])
    gaps = np.zeros(N); fwd = np.zeros(N, int)
    for i, p in enumerate(sub):
        gaps[i], fwd[i] = free_stats(p)
    FREE_MED = float(np.median(gaps))
    q1, q2 = np.quantile(dens, [1.0 / 3, 2.0 / 3])
    if q1 == q2:               # evita classe vazia se a densidade for muito concentrada
        q2 = q1 + 1e-6
    ylab = np.empty(N, int)
    for i in range(N):
        if dens[i] <= q1:                                   # terco menos cercado
            c = 'Livre'
        elif dens[i] >= q2:                                 # terco mais cercado
            c = 'Cercado'
        elif fwd[i] == 1 and gaps[i] >= FREE_MED:           # densidade media, saida a frente
            c = 'Frente_livre'
        else:                                               # densidade media, frente fechada
            c = 'Frente_bloqueada'
        ylab[i] = CIDX[c]
    # arrays
    Xp = np.zeros((N, KMAX, 3), np.float32); Mk = np.zeros((N, KMAX), np.float32)
    A = np.zeros((N, KMAX, KMAX), np.float32); IMG = np.zeros((N, 1, G, G), np.float32)
    xs = np.linspace(-R, R, G); gx, gy = np.meshgrid(xs, xs); sig = R / 12.0
    for i, p in enumerate(sub):
        if len(p):
            p = p[np.argsort(np.linalg.norm(p, axis=1))[:KMAX]]
            k = len(p); dist = np.linalg.norm(p, axis=1)
            Xp[i, :k] = np.column_stack([p[:, 0] / R, p[:, 1] / R, dist / R]); Mk[i, :k] = 1.0
            Dm = np.linalg.norm(p[:, None] - p[None], axis=2)
            a = (Dm <= R / 3).astype(float); np.fill_diagonal(a, 1.0)
            dinv = 1.0 / np.sqrt(a.sum(1, keepdims=True)); A[i, :k, :k] = a * dinv * dinv.T
            for (px, py) in p:
                IMG[i, 0] += np.exp(-(((gx - px) ** 2 + (gy - py) ** 2) / (2 * sig ** 2)))
    return ylab, Xp, A, Mk, IMG, FREE_MED


# ---------------- modelos ----------------
class DenseGCN(nn.Module):
    def __init__(s, f=3, h=48, L=2, ncls=4):
        super().__init__(); s.lins = nn.ModuleList([nn.Linear(f if i == 0 else h, h) for i in range(L)])
        s.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, ncls))
    def forward(s, X, Adj, M):
        H = X
        for lin in s.lins:
            H = torch.relu(torch.bmm(Adj, lin(H))) * M.unsqueeze(-1)
        m = M.unsqueeze(-1); mean = (H * m).sum(1) / m.sum(1).clamp(min=1)
        mx = torch.nan_to_num(H.masked_fill(m == 0, -1e9).max(1).values, neginf=0.0)
        return s.head(torch.cat([mean, mx], 1))


class CNNDeep(nn.Module):
    """CNN1 - convolucional profunda simples 32-64-128."""
    def __init__(s, ncls=4):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        s.c = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), nn.AdaptiveAvgPool2d(2))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, ncls))
    def forward(s, x):
        return s.head(s.c(x))


class CNNVGG(nn.Module):
    """CNN2 - estilo VGG (2 convs por bloco)."""
    def __init__(s, ncls=4):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        s.c = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), nn.AdaptiveAvgPool2d(2))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, ncls))
    def forward(s, x):
        return s.head(s.c(x))


class ResBlk(nn.Module):
    def __init__(s, ci, co, stride=1):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1); s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1); s.b2 = nn.BatchNorm2d(co)
        s.sc = nn.Sequential() if (stride == 1 and ci == co) else nn.Sequential(nn.Conv2d(ci, co, 1, stride), nn.BatchNorm2d(co))
    def forward(s, x):
        h = torch.relu(s.b1(s.c1(x))); h = s.b2(s.c2(h))
        return torch.relu(h + s.sc(x))


class CNNRes(nn.Module):
    """CNN3 - estilo ResNet (blocos residuais)."""
    def __init__(s, ncls=4):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        s.b = nn.Sequential(ResBlk(32, 32), ResBlk(32, 64, 2), ResBlk(64, 128, 2), nn.AdaptiveAvgPool2d(1))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(128, ncls))
    def forward(s, x):
        return s.head(s.b(s.stem(x)))


def build_model(kind):
    return {'gnn': DenseGCN(), 'cnn1_deep': CNNDeep(), 'cnn2_vgg': CNNVGG(), 'cnn3_resnet': CNNRes()}[kind]


def run_model(kind, T, ylab_t, tr, va, te, epochs):
    Xpt, At, Mkt, IMGt = T
    model = build_model(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def fwd(idx, aug):
        ix = torch.as_tensor(idx, dtype=torch.long)
        if kind.startswith('cnn'):
            ib = IMGt[ix]
            if aug and np.random.rand() < 0.5:
                ib = torch.flip(ib, dims=[2])
            return model(ib)
        xb = Xpt[ix]
        if aug and np.random.rand() < 0.5:
            xb = xb.clone(); xb[..., 1] = -xb[..., 1]
        return model(xb, At[ix], Mkt[ix])

    best, best_state, best_ep = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for sx in range(0, len(perm), 256):
            b = perm[sx:sx + 256]; opt.zero_grad()
            loss = lossf(fwd(b, True), ylab_t[torch.as_tensor(b, dtype=torch.long)]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = fwd(va, False).argmax(1).numpy()
        acc = accuracy_score(ylab_t.numpy()[va], pv)
        if acc > best:
            best, best_ep = acc, ep; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = fwd(te, False).argmax(1).numpy()
    yt = ylab_t.numpy()[te]
    return accuracy_score(yt, pt), f1_score(yt, pt, average='macro'), best, best_ep + 1


def main():
    adv_list = build_adv_list()
    if SMOKE:
        adv_list = adv_list[:1500]
    N = len(adv_list)
    print('amostras (Euro 2020, >=1 adversario em 30m):', N)
    ep = 1 if SMOKE else EPOCHS
    pos = np.arange(N)
    grid_acc = pd.DataFrame(index=['gnn', 'cnn1_deep', 'cnn2_vgg', 'cnn3_resnet'], columns=[int(r) for r in RADII], dtype=float)
    grid_f1 = grid_acc.copy()
    per_run = []; label_dist = []
    for R in RADII:
        ylab, Xp, A, Mk, IMG, FREE_MED = build_radius(adv_list, R)
        dist = {CLASSES[i]: int((ylab == i).sum()) for i in range(4)}
        label_dist.append(dict(raio=int(R), free_med=round(FREE_MED, 1), **dist))
        print('R=%dm | dist: %s | free_med=%.1f' % (R, dist, FREE_MED))
        tr, te = train_test_split(pos, test_size=0.30, stratify=ylab, random_state=42)
        tr2, va = train_test_split(tr, test_size=0.20, stratify=ylab[tr], random_state=42)
        T = (torch.tensor(Xp), torch.tensor(A), torch.tensor(Mk), torch.tensor(IMG))
        ylab_t = torch.tensor(ylab, dtype=torch.long)
        for kind in ['gnn', 'cnn1_deep', 'cnn2_vgg', 'cnn3_resnet']:
            acc, f1, vacc, bep = run_model(kind, T, ylab_t, tr2, va, te, ep)
            grid_acc.loc[kind, int(R)] = round(acc, 4); grid_f1.loc[kind, int(R)] = round(f1, 4)
            per_run.append(dict(raio=int(R), modelo=kind, acuracia=round(acc, 4), f1_macro=round(f1, 4),
                                val_acc=round(vacc, 4), melhor_epoca=bep))
            print('  R=%dm %-12s acc=%.4f f1=%.4f' % (R, kind, acc, f1))
        del T, Xp, A, Mk, IMG
    grid_acc.to_csv(os.path.join(OUT, 'grid_accuracy.csv'))
    grid_f1.to_csv(os.path.join(OUT, 'grid_f1.csv'))
    pd.DataFrame(per_run).to_csv(os.path.join(OUT, 'per_run.csv'), index=False)
    pd.DataFrame(label_dist).to_csv(os.path.join(OUT, 'label_dist.csv'), index=False)
    pd.DataFrame([dict(n_amostras=N, raios=str(RADII), epochs=ep, has_torch=True, kmax=KMAX, grid=G, smoke=SMOKE)]).to_csv(os.path.join(OUT, 'run_info.csv'), index=False)
    print('=== grid acuracia ===')
    print(grid_acc.to_string())
    print('=== grid f1-macro ===')
    print(grid_f1.to_string())
    print('OK')


if __name__ == '__main__':
    main()
