# -*- coding: utf-8 -*-
"""eval_usability.py — a AUC ~0,70 e utilizavel por uma comissao tecnica?
Calibracao (reliability + ECE + Brier) e precisao/recall (limiar + lift top-k%) do GBT no melhor
alvo. Mostra que discriminacao modesta + base rate baixo => alertas individuais pouco precisos,
mas uso agregado/calibrado e defensavel. Saida: star_analysis/usability.csv + calibration_*.png."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'xR_experiments'))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba

OUT = 'star_analysis'; os.makedirs(OUT, exist_ok=True)
TARGETS = ['y_5s_5m', 'y_2act', 'y_10s']


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return e


def topk_precision(y, p, frac):
    k = max(1, int(len(p) * frac)); idx = np.argsort(-p)[:k]
    return y[idx].mean(), y[idx].sum() / y.sum()      # precisao no top-k%, recall capturado


def main():
    meta = xd.load_meta()
    X = meta[xd.TAB_FEATURES].to_numpy(np.float32)
    rows = []
    for tgt in TARGETS:
        y = meta[tgt].to_numpy(np.int32)
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
        tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
        model, kind = make_gbt(); fit_gbt(model, kind, X[tr2], y[tr2], X[va], y[va])
        p = proba(model, X[te]); yt = y[te]
        base = yt.mean()

        auc = roc_auc_score(yt, p); ap = average_precision_score(yt, p)
        brier = brier_score_loss(yt, p); e = ece(yt, p)
        # calibracao isotonica (ajustada na validacao) -> melhora possivel
        iso = IsotonicRegression(out_of_bounds='clip').fit(proba(model, X[va]), y[va])
        p_cal = iso.predict(p); brier_cal = brier_score_loss(yt, p_cal); e_cal = ece(yt, p_cal)
        # limiar 0.5
        yhat = (p >= 0.5).astype(int)
        prec = precision_score(yt, yhat, zero_division=0); rec = recall_score(yt, yhat, zero_division=0)
        f1 = f1_score(yt, yhat, zero_division=0)
        # lift por triagem top-k%
        p5, r5 = topk_precision(yt, p, 0.05); p10, r10 = topk_precision(yt, p, 0.10)
        p20, r20 = topk_precision(yt, p, 0.20)

        rows.append(dict(alvo=tgt, base_rate=round(base, 3), AUC=round(auc, 3), PR_AUC=round(ap, 3),
                         Brier=round(brier, 4), ECE=round(e, 4), Brier_cal=round(brier_cal, 4),
                         ECE_cal=round(e_cal, 4), prec_0p5=round(prec, 3), rec_0p5=round(rec, 3),
                         f1_0p5=round(f1, 3), prec_top5=round(p5, 3), rec_top5=round(r5, 3),
                         prec_top10=round(p10, 3), rec_top10=round(r10, 3),
                         prec_top20=round(p20, 3), lift_top10=round(p10 / base, 2)))
        print('[%s] base=%.3f AUC=%.3f PR-AUC=%.3f Brier=%.4f ECE=%.3f | '
              'top10%%: prec=%.3f (lift %.2fx) rec=%.3f | thr0.5: P=%.3f R=%.3f'
              % (tgt, base, auc, ap, brier, e, p10, p10 / base, r10, prec, rec))

        # figura de calibracao (alvo principal)
        if tgt == 'y_5s_5m':
            fx, fy = calibration_curve(yt, p, n_bins=10, strategy='quantile')
            fxc, fyc = calibration_curve(yt, p_cal, n_bins=10, strategy='quantile')
            plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], 'k--', lw=1, label='perfeita')
            plt.plot(fx, fy, 'o-', label='GBT bruto (ECE %.3f)' % e)
            plt.plot(fxc, fyc, 's-', label='GBT isotonico (ECE %.3f)' % e_cal)
            plt.xlabel('risco previsto'); plt.ylabel('frequencia observada')
            plt.title('Calibracao — %s' % tgt); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(OUT, 'calibration_y5s5m.png'), dpi=110); plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'usability.csv'), index=False)
    print('\n', df.to_string(index=False))
    print('salvo em', os.path.join(OUT, 'usability.csv'), '+ calibration_y5s5m.png')


if __name__ == '__main__':
    main()
