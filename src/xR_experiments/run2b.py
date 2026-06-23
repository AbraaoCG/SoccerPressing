# -*- coding: utf-8 -*-
"""Runner v2b: TODAS as competicoes com 360 + CNN multi-canal (adversarios + apoio).
Processa jogo a jogo (memoria limitada). Escreve classify_analysis/."""
import os as _os; from pathlib import Path as _Path
_os.chdir(str(_Path(__file__).resolve().parent.parent.parent))  # roda a partir da raiz do projeto
import json, os, math, glob, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
np.random.seed(42)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
torch.manual_seed(42)

DATA_DIR = Path('StatsBomb_2/data')
RECOVERY_WINDOW = 10.0
LOCAL_R, NEIGH_R, KADV, KSUP = 30.0, 10.0, 14, 12
G, HALF, SIG = 32, 25.0, 2.0
OUT = 'classify_analysis'; os.makedirs(OUT, exist_ok=True)
os.makedirs('cache', exist_ok=True)
META_PKL, ARR_NPZ = 'cache/_meta2b.pkl', 'cache/_arr2b.npz'
PRIM2 = ['n_adv_5m', 'n_adv_10m', 'n_adv_15m', 'nearest_adv_dist', 'adv_arc_coverage',
         'largest_free_angle', 'free_lane_forward', 'adv_centroid_dist', 'adv_centroid_angle', 'adv_spread']


def load_json(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        print(f'[WARN] JSON inválido em {p}: {e}')
        return None  # sinaliza falha

def ts(t):
    hh, mm, ss = t.split(':'); return int(hh) * 3600 + int(mm) * 60 + float(ss)


def geom_of(ball, carrier, advs, sups):
    res = {k: 0 for k in PRIM2}
    d = advs - carrier
    adv_r = np.column_stack([-d[:, 0], d[:, 1]])
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
    cen = loc.mean(0)
    res['adv_centroid_dist'] = float(np.linalg.norm(cen))
    res['adv_centroid_angle'] = float((np.degrees(np.arctan2(cen[1], cen[0])) + 360) % 360)
    res['adv_spread'] = float(np.linalg.norm(loc, axis=1).std()) if len(loc) > 1 else 0.0
    adv_local = adv_r[dist <= LOCAL_R]
    if len(sups):
        ds = sups - carrier
        sup_r = np.column_stack([-ds[:, 0], ds[:, 1]])
        sup_local = sup_r[np.linalg.norm(sup_r, axis=1) <= LOCAL_R]
    else:
        sup_local = np.zeros((0, 2))
    return res, adv_local, sup_local


def process_match(mid):
    ev_path = DATA_DIR / 'events' / (str(mid) + '.json')
    fr_path = DATA_DIR / 'three-sixty' / (str(mid) + '.json')

    ev_data = load_json(ev_path)
    fr_data = load_json(fr_path)

    if ev_data is None or fr_data is None:
        return []  # pula o jogo com arquivo corrompido

    ev = pd.json_normalize(ev_data, sep='_')
    frames = {fr['event_uuid']: fr for fr in fr_data}
    ev['t'] = ev['timestamp'].apply(ts)
    pr = ev[ev['type_name'] == 'Pressure'].copy()
    if len(pr) == 0:
        return []
    rec_fail = ev['ball_recovery_recovery_failure'].fillna(False) if 'ball_recovery_recovery_failure' in ev.columns else pd.Series(False, index=ev.index)
    duel_out = ev.get('duel_outcome_name', pd.Series('', index=ev.index)).fillna('')
    rm = ((ev['type_name'].eq('Ball Recovery') & ~rec_fail) | ev['type_name'].eq('Interception') | (ev['type_name'].eq('Duel') & duel_out.str.contains('Won|Success')))
    rec = ev.loc[rm, ['period', 'team_id', 't']]
    L = pr[['id', 'period', 'team_id', 't']].rename(columns={'t': 'tp'}).sort_values('tp')
    R = rec.rename(columns={'t': 'tr'}).sort_values('tr')
    m = pd.merge_asof(L, R, left_on='tp', right_on='tr', by=['period', 'team_id'], direction='forward', tolerance=RECOVERY_WINDOW)
    recov = dict(zip(m['id'], m['tr'].notna().astype(int)))
    out = []
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
        sup_others = np.delete(sups, int(np.argmin(np.linalg.norm(sups - ball, axis=1))), axis=0)
        if np.min(np.linalg.norm(advs - carrier, axis=1)) > LOCAL_R:
            continue
        prim, adv_local, sup_local = geom_of(ball, carrier, advs, sup_others)
        x, y = float(ball[0]), float(ball[1])
        rec_d = {'recovered': int(recov.get(row['id'], 0)), 'x': x, 'y': y,
                 'dist_to_goal': math.sqrt((120 - x) ** 2 + (40 - y) ** 2),
                 'zone': 'Def' if x <= 40 else ('Mid' if x <= 80 else 'Att'),
                 'minute': float(row.get('minute', 0)), 'duration': float(row.get('duration', 0) or 0)}
        rec_d.update(prim)
        out.append((rec_d, adv_local.astype(np.float32), sup_local.astype(np.float32)))
    return out


def build():
    if os.path.exists(META_PKL) and os.path.exists(ARR_NPZ):
        meta = pd.read_pickle(META_PKL)
        z = np.load(ARR_NPZ)
        return meta, z['ADV'], z['SUP'], z['AM'], z['SM'], z['IMG'], z['y']
    mfiles = glob.glob(str(DATA_DIR / 'matches' / '*' / '*.json'))
    mids = []
    for mf in mfiles:
        for mm in load_json(mf):
            mid = mm['match_id']
            if (DATA_DIR / 'three-sixty' / (str(mid) + '.json')).exists():
                mids.append(mid)
    mids = sorted(set(mids))
    print('jogos com 360:', len(mids))
    recs, adv_l, sup_l = [], [], []
    for j, mid in enumerate(mids):
        for rec_d, a, s in process_match(mid):
            recs.append(rec_d); adv_l.append(a); sup_l.append(s)
        if (j + 1) % 40 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(recs)))
    meta = pd.DataFrame(recs)
    N = len(meta)
    xs = np.linspace(-HALF, HALF, G); gx, gy = np.meshgrid(xs, xs)
    ADV = np.zeros((N, KADV, 2), np.float32); AM = np.zeros((N, KADV), np.float32)
    SUP = np.zeros((N, KSUP, 2), np.float32); SM = np.zeros((N, KSUP), np.float32)
    IMG = np.zeros((N, 2, G, G), np.float32)
    for i in range(N):
        a = adv_l[i]; s = sup_l[i]
        a = a[np.argsort(np.linalg.norm(a, axis=1))[:KADV]] if len(a) else a
        s = s[np.argsort(np.linalg.norm(s, axis=1))[:KSUP]] if len(s) else s
        ka, ks = len(a), len(s)
        ADV[i, :ka] = a; AM[i, :ka] = 1.0
        SUP[i, :ks] = s; SM[i, :ks] = 1.0
        for (px, py) in a:
            IMG[i, 0] += np.exp(-(((gx - px) ** 2 + (gy - py) ** 2) / (2 * SIG ** 2)))
        for (px, py) in s:
            IMG[i, 1] += np.exp(-(((gx - px) ** 2 + (gy - py) ** 2) / (2 * SIG ** 2)))
    y = meta['recovered'].to_numpy().astype(np.float32)
    meta.to_pickle(META_PKL)
    np.savez_compressed(ARR_NPZ, ADV=ADV, SUP=SUP, AM=AM, SM=SM, IMG=IMG, y=y)
    return meta, ADV, SUP, AM, SM, IMG, y


meta, ADV, SUP, AM, SM, IMG, y = build()
N = len(meta)
print('amostras:', N, '| taxa recuperacao %.3f' % y.mean(), '| IMG', IMG.shape)

# point features (adversarios) e adjacencia
Xp = np.concatenate([ADV / 30.0, (np.linalg.norm(ADV, axis=2, keepdims=True) / 30.0)], axis=2).astype(np.float32)
Xp = Xp * AM[:, :, None]
A = np.zeros((N, KADV, KADV), np.float32)
for i in range(N):
    k = int(AM[i].sum())
    if k >= 1:
        p = ADV[i, :k]
        Dm = np.linalg.norm(p[:, None] - p[None], axis=2)
        a = (Dm <= NEIGH_R).astype(float); np.fill_diagonal(a, 1.0)
        dinv = 1.0 / np.sqrt(a.sum(1, keepdims=True)); A[i, :k, :k] = (a * dinv * dinv.T)

# ---- Baseline regras ----
FREE_MED = float(meta['largest_free_angle'].median())

def rule_row(r):
    if r['n_adv_10m'] <= 1: return 'Livre'
    if r['n_adv_5m'] >= 2: return 'Cercado'
    if r['free_lane_forward'] == 1 and r['largest_free_angle'] >= FREE_MED: return 'Frente_livre'
    return 'Frente_bloqueada'
meta['ctx_rule'] = meta.apply(rule_row, axis=1)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))
rows = []
for cl, g in meta.groupby('ctx_rule'):
    n = len(g); k = int(g['recovered'].sum()); lo, hi = wilson(k, n)
    rows.append(dict(classe=cl, n=n, taxa_recuperacao=k / n, ic_inf=lo, ic_sup=hi))
pd.DataFrame(rows).sort_values('taxa_recuperacao').to_csv(os.path.join(OUT, 'rules_summary.csv'), index=False)
ct = pd.crosstab(meta['ctx_rule'], meta['recovered']); chi2, pchi, dof, _ = stats.chi2_contingency(ct)
cramer = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
pd.DataFrame([dict(chi2=chi2, p_valor=pchi, cramers_v=cramer, n_classes=ct.shape[0])]).to_csv(os.path.join(OUT, 'rules_test.csv'), index=False)
print('REGRAS classes:', dict(meta['ctx_rule'].value_counts()), '| CramersV %.3f' % cramer)

# ---- k-means ----
PRIMV = meta[PRIM2].astype(float).fillna(meta[PRIM2].median()).to_numpy()
Z = StandardScaler().fit_transform(PRIMV)
ks = range(2, 9); ksel = []
for k in ks:
    kmf = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    ksel.append(dict(k=k, inertia=kmf.inertia_, silhouette=silhouette_score(Z, kmf.labels_, sample_size=4000, random_state=42)))
ksel = pd.DataFrame(ksel); ksel.to_csv(os.path.join(OUT, 'kmeans_k_selection.csv'), index=False)
K_BEST = int(ksel.loc[ksel['silhouette'].idxmax(), 'k'])
KMLAB = KMeans(n_clusters=K_BEST, n_init=10, random_state=42).fit_predict(Z)
crows = []
for c in range(K_BEST):
    msk = KMLAB == c; n = int(msk.sum()); kk = int(y[msk].sum()); lo, hi = wilson(kk, n)
    crows.append(dict(cluster=c, n=n, taxa_recuperacao=kk / n, ic_inf=lo, ic_sup=hi))
pd.DataFrame(crows).to_csv(os.path.join(OUT, 'kmeans_clusters.csv'), index=False)
pd.DataFrame(Z, columns=PRIM2).assign(cluster=KMLAB).groupby('cluster').mean().round(3).to_csv(os.path.join(OUT, 'kmeans_profile.csv'))
print('KMEANS k=%d' % K_BEST)

# ---- modelos torch ----
pos = np.arange(N)
tr, tmp = train_test_split(pos, test_size=0.40, stratify=y, random_state=42)
stack_i, test_i = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
tr2, val_i = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
print('train %d val %d stack %d test %d' % (len(tr2), len(val_i), len(stack_i), len(test_i)))
yt = torch.tensor(y)


class DeepSets(nn.Module):
    def __init__(s, f=3, h=32):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        s.rho = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))
    def forward(s, X, M):
        h = s.phi(X) * M.unsqueeze(-1); m = M.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1); mx = h.masked_fill(m == 0, -1e9).max(1).values
        return s.rho(torch.cat([mean, mx], 1)).squeeze(-1)


class DenseGCN(nn.Module):
    def __init__(s, f=3, h=32, L=2):
        super().__init__(); s.lins = nn.ModuleList([nn.Linear(f if i == 0 else h, h) for i in range(L)])
        s.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))
    def forward(s, X, Adj, M):
        H = X
        for lin in s.lins:
            H = torch.relu(torch.bmm(Adj, lin(H))) * M.unsqueeze(-1)
        m = M.unsqueeze(-1); mean = (H * m).sum(1) / m.sum(1).clamp(min=1); mx = H.masked_fill(m == 0, -1e9).max(1).values
        return s.head(torch.cat([mean, mx], 1)).squeeze(-1)


class SetTransformer(nn.Module):
    def __init__(s, f=3, h=32, heads=4, L=2):
        super().__init__(); s.inp = nn.Linear(f, h)
        s.att = nn.ModuleList([nn.MultiheadAttention(h, heads, batch_first=True) for _ in range(L)])
        s.ff = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)) for _ in range(L)])
        s.n1 = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)]); s.n2 = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        s.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))
    def forward(s, X, M):
        h = torch.relu(s.inp(X)); kpm = (M == 0)
        for att, ff, n1, n2 in zip(s.att, s.ff, s.n1, s.n2):
            a, _ = att(h, h, h, key_padding_mask=kpm); h = n1(h + a); h = n2(h + ff(h))
        m = M.unsqueeze(-1); pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return s.head(pooled).squeeze(-1)


class MultiChannelCNN(nn.Module):
    def __init__(s, ch=2):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(s, img):
        return s.head(s.c(img)).squeeze(-1)


def make_fwd(kind, model):
    def fwd(idx, aug):
        if kind == 'cnn':
            ib = torch.as_tensor(IMG[idx])
            if aug and np.random.rand() < 0.5:
                ib = torch.flip(ib, dims=[2])
            return model(ib)
        xb = torch.as_tensor(Xp[idx])
        if aug and np.random.rand() < 0.5:
            xb = xb.clone(); xb[..., 1] = -xb[..., 1]
        mb = torch.as_tensor(AM[idx])
        if kind == 'gcn':
            return model(xb, torch.as_tensor(A[idx]), mb)
        return model(xb, mb)
    return fwd


def train_model(kind, model, epochs=35, bs=256, lr=1e-3):
    fwd = make_fwd(kind, model); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); best, best_state = -1, None
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr2)
        for sx in range(0, len(perm), bs):
            b = perm[sx:sx + bs]; opt.zero_grad()
            loss = lossf(fwd(b, True), yt[torch.as_tensor(b)]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y[val_i], torch.sigmoid(fwd(val_i, False)).numpy())
        if auc > best:
            best = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return best


def predict(kind, model, idx):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(make_fwd(kind, model)(idx, False)).numpy()


models = {'deepsets': DeepSets(), 'gcn': DenseGCN(), 'settransf': SetTransformer(), 'cnn': MultiChannelCNN()}
geom_auc, gst, gte = {}, {}, {}
for kind, mdl in models.items():
    vb = train_model(kind, mdl)
    gst[kind] = predict(kind, mdl, stack_i); gte[kind] = predict(kind, mdl, test_i)
    geom_auc[kind] = roc_auc_score(y[test_i], gte[kind])
    print('  %-10s val %.3f test %.3f' % (kind, vb, geom_auc[kind]))

def lrp(labels, a, b):
    s = pd.Series(y[a]).groupby(labels[a]).mean(); g = y[a].mean()
    return np.array([s.get(l, g) for l in labels[b]])
RULEV = meta['ctx_rule'].to_numpy()
auc_rule = roc_auc_score(y[test_i], lrp(RULEV, tr, test_i))
auc_km = roc_auc_score(y[test_i], lrp(KMLAB, tr, test_i))

def ctxmat(idx):
    s = meta.iloc[idx]; z = pd.get_dummies(s['zone'])
    for c in ['Def', 'Mid', 'Att']:
        if c not in z: z[c] = 0
    return np.nan_to_num(np.column_stack([s['x'], s['y'], s['dist_to_goal'], s['minute'], s['duration'], z['Def'], z['Mid'], z['Att']]).astype(float))
CTX = ctxmat(np.arange(N))
scaler = StandardScaler().fit(CTX[tr])
cm = LogisticRegression(max_iter=1000).fit(scaler.transform(CTX[tr]), y[tr])
p_ctx_stack = cm.predict_proba(scaler.transform(CTX[stack_i]))[:, 1]; p_ctx_test = cm.predict_proba(scaler.transform(CTX[test_i]))[:, 1]
auc_ctx = roc_auc_score(y[test_i], p_ctx_test)
best_geom = max(geom_auc, key=lambda k: roc_auc_score(y[stack_i], gst[k]))
meta_m = LogisticRegression(max_iter=1000).fit(np.column_stack([p_ctx_stack, gst[best_geom]]), y[stack_i])
p_meta = meta_m.predict_proba(np.column_stack([p_ctx_test, gte[best_geom]]))[:, 1]
auc_meta = roc_auc_score(y[test_i], p_meta)

comp = [dict(modelo='regras_1d', auc_teste=auc_rule), dict(modelo='kmeans_1d', auc_teste=auc_km)]
for k in ['deepsets', 'gcn', 'settransf', 'cnn']:
    comp.append(dict(modelo=k, auc_teste=geom_auc[k]))
comp += [dict(modelo='contexto', auc_teste=auc_ctx), dict(modelo='stacking(contexto+%s)' % best_geom, auc_teste=auc_meta)]
pd.DataFrame(comp).to_csv(os.path.join(OUT, 'model_comparison.csv'), index=False)
pd.DataFrame([dict(coef_contexto=meta_m.coef_[0][0], coef_geometria=meta_m.coef_[0][1], intercepto=meta_m.intercept_[0], melhor_geometria=best_geom)]).to_csv(os.path.join(OUT, 'stacking_meta.csv'), index=False)
pd.DataFrame([dict(n_amostras=N, taxa_recuperacao=float(y.mean()), n_canais=2, has_torch=True, n_train=len(tr2), n_val=len(val_i), n_stack=len(stack_i), n_test=len(test_i), kadv=KADV, ksup=KSUP)]).to_csv(os.path.join(OUT, 'run_info.csv'), index=False)
print('=== model_comparison ===')
print(pd.DataFrame(comp).round(4).to_string(index=False))
print('melhor geom:', best_geom, '| meta coef [ctx,geom]:', np.round(meta_m.coef_[0], 3))
print('OK')
