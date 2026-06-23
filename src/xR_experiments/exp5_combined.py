# -*- coding: utf-8 -*-
"""Experimento 5 — Combinado (junta as 4 ideias).
best_target (do exp2) + split cross-competition (held-out comp 43) + stacking de
GBT-tabular (ideia 1) e CNN-multicanal (ideia 4). Saida: express_analysis/exp5.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import torch, torch.nn as nn
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba, auc_acc
from exp4_soccermap_cnn import CNN

torch.manual_seed(42); np.random.seed(42)
OUT = 'express_analysis'; os.makedirs(OUT, exist_ok=True)
HELDOUT_COMP = 43
EPOCHS = 1 if os.environ.get('SMOKE') == '1' else int(os.environ.get('EPOCHS', '15'))


def best_target():
    p = os.path.join(OUT, 'exp2.csv')
    if os.path.exists(p):
        return pd.read_csv(p).sort_values('auc', ascending=False).iloc[0]['alvo']
    return 'y_5s_5m'


def train_cnn(IMG, y, tr, va, epochs):
    yt = torch.tensor(y, dtype=torch.float32)
    model = CNN(IMG.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    bat = lambda idx: torch.tensor(np.asarray(IMG[idx], np.float32))
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for sx in range(0, len(perm), 256):
            b = perm[sx:sx + 256]; opt.zero_grad()
            loss = lossf(model(bat(b)), yt[torch.as_tensor(b)]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y[va], torch.sigmoid(model(bat(va))).numpy())
        if auc > best:
            best = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model


def cnn_proba(model, IMG, idx):
    with torch.no_grad():
        out = []
        for sx in range(0, len(idx), 1024):
            b = idx[sx:sx + 1024]
            out.append(torch.sigmoid(model(torch.tensor(np.asarray(IMG[b], np.float32)))).numpy())
    return np.concatenate(out)


def main():
    tgt = best_target()
    meta, IMG = xd.load()
    y = meta[tgt].to_numpy(np.int32)
    X = meta[xd.TAB_FEATURES].to_numpy(np.float32)
    comp = meta['competition_id'].to_numpy()
    nm = xd.COMP_NAMES.get(HELDOUT_COMP, str(HELDOUT_COMP))
    print('exp5 | alvo %s | held-out %s | epocas %d' % (tgt, nm, EPOCHS))

    te = np.where(comp == HELDOUT_COMP)[0]
    pool = np.where(comp != HELDOUT_COMP)[0]
    base, calib = train_test_split(pool, test_size=0.25, stratify=y[pool], random_state=42)
    base2, val = train_test_split(base, test_size=0.15, stratify=y[base], random_state=42)

    # modelos base
    gbt, kind = make_gbt(); fit_gbt(gbt, kind, X[base2], y[base2], X[val], y[val])
    cnn = train_cnn(IMG, y, base2, val, EPOCHS)

    # probabilidades
    pg_te, pc_te = proba(gbt, X[te]), cnn_proba(cnn, IMG, te)
    pg_ca, pc_ca = proba(gbt, X[calib]), cnn_proba(cnn, IMG, calib)

    auc_g, acc_g = auc_acc(y[te], pg_te)
    auc_c, acc_c = auc_acc(y[te], pc_te)

    # meta-modelo (stacking) ajustado na calibracao
    meta_lr = LogisticRegression(max_iter=1000)
    meta_lr.fit(np.column_stack([pg_ca, pc_ca]), y[calib])
    p_stack = meta_lr.predict_proba(np.column_stack([pg_te, pc_te]))[:, 1]
    auc_s, acc_s = auc_acc(y[te], p_stack)

    print('GBT       : AUC %.4f | acc %.4f' % (auc_g, acc_g))
    print('CNN(4ch)  : AUC %.4f | acc %.4f' % (auc_c, acc_c))
    print('Stacking  : AUC %.4f | acc %.4f' % (auc_s, acc_s))

    pd.DataFrame([
        dict(exp='exp5', modelo='GBT-tabular', alvo=tgt, split='held-out %s' % nm, auc=round(auc_g, 4), acc=round(acc_g, 4), n_test=len(te)),
        dict(exp='exp5', modelo='CNN-4canais', alvo=tgt, split='held-out %s' % nm, auc=round(auc_c, 4), acc=round(acc_c, 4), n_test=len(te)),
        dict(exp='exp5', modelo='stacking(GBT+CNN)', alvo=tgt, split='held-out %s' % nm, auc=round(auc_s, 4), acc=round(acc_s, 4), n_test=len(te)),
    ]).to_csv(os.path.join(OUT, 'exp5.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'exp5.csv'))


if __name__ == '__main__':
    main()
