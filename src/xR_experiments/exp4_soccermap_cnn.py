# -*- coding: utf-8 -*-
"""Experimento 4 — CNN multicanal estilo SoccerMap.
Raster de 4 canais (pressers, apoio, dist-ao-gol, ang-ao-gol) vs. o raster 1-canal de hoje.
Alvo y_10s, split aleatorio. Saida: express_analysis/exp4.csv.
Compara com CNN 1-canal de hoje (0,589) e SoccerMap do artigo (0,596)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch, torch.nn as nn
import express_data as xd

torch.manual_seed(42); np.random.seed(42)
OUT = 'express_analysis'; os.makedirs(OUT, exist_ok=True)
TARGET = 'y_10s'
EPOCHS = 1 if os.environ.get('SMOKE') == '1' else int(os.environ.get('EPOCHS', '20'))


class CNN(nn.Module):
    def __init__(s, cin=4):
        super().__init__()
        s.c = nn.Sequential(
            nn.Conv2d(cin, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(2))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4, 128), nn.ReLU(),
                               nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(s, x):
        return s.head(s.c(x)).squeeze(-1)


def train_eval(IMG, y, tr, va, te, epochs):
    yt = torch.tensor(y, dtype=torch.float32)
    model = CNN(IMG.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def batch(idx):
        return torch.tensor(np.asarray(IMG[idx], np.float32))

    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        run, nb = 0.0, 0
        for sx in range(0, len(perm), 256):
            b = perm[sx:sx + 256]; opt.zero_grad()
            loss = lossf(model(batch(b)), yt[torch.as_tensor(b)]); loss.backward(); opt.step()
            run += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(batch(va))).numpy()
        auc = roc_auc_score(y[va], pv)
        print('  epoca %02d/%d | loss %.4f | val_auc %.4f%s'
              % (ep + 1, epochs, run / nb, auc, '  <-- best' if auc > best else ''))
        if auc > best:
            best = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pte = torch.sigmoid(model(batch(te))).numpy()
    return float(roc_auc_score(y[te], pte)), float(accuracy_score(y[te], (pte >= 0.5).astype(int)))


def main():
    meta, IMG = xd.load()
    y = meta[TARGET].to_numpy(np.int32)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    print('CNN multicanal | canais %d | epocas %d | train %d val %d test %d'
          % (IMG.shape[1], EPOCHS, len(tr2), len(va), len(te)))
    auc, acc = train_eval(IMG, y, tr2, va, te, EPOCHS)
    print('CNN(4ch): AUC %.4f | acc %.4f' % (auc, acc))
    pd.DataFrame([dict(exp='exp4', modelo='CNN-4canais', alvo=TARGET, split='aleatorio',
                       auc=round(auc, 4), acc=round(acc, 4), n_test=len(te))]
                 ).to_csv(os.path.join(OUT, 'exp4.csv'), index=False)
    print('salvo em', os.path.join(OUT, 'exp4.csv'))


if __name__ == '__main__':
    main()
