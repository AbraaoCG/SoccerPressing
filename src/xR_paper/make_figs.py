# -*- coding: utf-8 -*-
"""make_figs.py — gera as figuras do artigo (PNG) na pasta do template SBPO.
Figuras: efeito do alvo, ablacao da geometria, calibracao (logit+GBT), curva de ganho/lift,
e o diagrama de fluxo de aplicacao do xR. Usa o cache express (91k)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'xR_experiments'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve
import express_data as xd
from gbt_util import make_gbt, fit_gbt, proba

FIG = Path(r"C:\Users\Abraao Ideapad3\Documents\Projetos\COE609\Template_SBPO2026_LaTeX_ptbr\Template_SBPO2026_LaTeX_ptbr\figs")
FIG.mkdir(parents=True, exist_ok=True)
NAVY = '#26215C'; PURPLE = '#534AB7'; TEAL = '#1D9E75'; TERRA = '#D85A30'; SAND = '#B4B2A9'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#888780', 'axes.linewidth': 0.8})


def fig_target():
    labels = ['y_10s', 'y_5s', 'y_2act', 'y_5s_9m', 'y_5s_5m']
    auc = [0.591, 0.633, 0.643, 0.689, 0.702]
    cols = [SAND, SAND, SAND, PURPLE, NAVY]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b = ax.bar(labels, auc, color=cols)
    ax.axhline(0.591, ls='--', lw=1, color='#888780')
    ax.set_ylim(0.5, 0.75); ax.set_ylabel('AUC (teste)')
    ax.set_title(u'Efeito da definição do alvo (GBT, mesmas features)')
    for r, v in zip(b, auc):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.004, '%.3f' % v, ha='center', fontsize=9.5)
    ax.text(4, 0.60, u'alvo restrito\ngeograficamente', ha='center', color=NAVY, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / 'fig_target.png', dpi=150); plt.close(fig)


def fig_ablation():
    blocks = ['GEOM', 'POS', 'CTX', 'POS+CTX', 'ALL']
    y55 = [0.679, 0.609, 0.558, 0.632, 0.702]
    y10 = [0.585, 0.527, 0.524, 0.538, 0.591]
    x = np.arange(len(blocks)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w / 2, y55, w, label='alvo y_5s_5m', color=NAVY)
    ax.bar(x + w / 2, y10, w, label='alvo y_10s', color=SAND)
    ax.set_xticks(x); ax.set_xticklabels(blocks); ax.set_ylim(0.5, 0.75)
    ax.set_ylabel('AUC (teste)'); ax.legend(frameon=False, fontsize=9)
    ax.set_title(u'Ablação por bloco de features')
    fig.tight_layout(); fig.savefig(FIG / 'fig_ablation.png', dpi=150); plt.close(fig)


def fit_models():
    meta = xd.load_meta()
    X = meta[xd.TAB_FEATURES].to_numpy(np.float32); y = meta['y_5s_5m'].to_numpy(np.int32)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=42)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
    p_lr = lr.predict_proba(X[te])[:, 1]
    gbt, kind = make_gbt(); fit_gbt(gbt, kind, X[tr2], y[tr2], X[va], y[va]); p_gbt = proba(gbt, X[te])
    return y[te], p_lr, p_gbt


def fig_calibration(yt, p_lr, p_gbt):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 0.5], [0, 0.5], 'k--', lw=1, label='perfeita')
    fx, fy = calibration_curve(yt, p_lr, n_bins=10, strategy='quantile')
    gx, gy = calibration_curve(yt, p_gbt, n_bins=10, strategy='quantile')
    ax.plot(fx, fy, 'o-', color=NAVY, label=u'logística (ECE 0,002)')
    ax.plot(gx, gy, 's-', color=TEAL, label='GBT (ECE 0,003)')
    ax.set_xlabel('risco previsto (xR)'); ax.set_ylabel('frequência observada')
    ax.set_title(u'Calibração — alvo y_5s_5m'); ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5)
    fig.tight_layout(); fig.savefig(FIG / 'fig_calibration.png', dpi=150); plt.close(fig)


def fig_lift(yt, p_lr):
    order = np.argsort(-p_lr); ys = yt[order]; base = yt.mean()
    fracs = np.linspace(0.02, 1.0, 50)
    prec = [ys[:max(1, int(f * len(ys)))].mean() for f in fracs]
    rec = [ys[:max(1, int(f * len(ys)))].sum() / ys.sum() for f in fracs]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(fracs * 100, np.array(prec) / base, color=NAVY, lw=2, label=u'lift (precisão / base)')
    ax.axhline(1.0, ls='--', lw=1, color='#888780')
    ax.set_xlabel(u'% de jogadas revisadas (maior risco primeiro)'); ax.set_ylabel('lift', color=NAVY)
    ax.tick_params(axis='y', colors=NAVY)
    ax2 = ax.twinx()
    ax2.plot(fracs * 100, np.array(rec) * 100, color=TERRA, lw=2, label='recall acumulado')
    ax2.set_ylabel('recall acumulado (%)', color=TERRA); ax2.tick_params(axis='y', colors=TERRA)
    ax.axvline(10, ls=':', lw=1, color='#888780')
    ax.set_title(u'Curva de ganho — triagem por risco (logística, y_5s_5m)')
    fig.tight_layout(); fig.savefig(FIG / 'fig_lift.png', dpi=150); plt.close(fig)


def fig_flow():
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    stages = ['1\nDados\n(eventos+360)', '2\nModelo xR\n(prob. calibrada)', '3\nAgregação\n(zona/gatilho)',
              u'4\nPré-jogo\n(adversário)', u'5\nPós-jogo\n(xR vs real)', '6\nSazonal\n(scouting)']
    xs = np.linspace(1.0, 9.0, 6)
    ax.plot([1.0, 9.0], [5.0, 5.0], color=SAND, lw=2, zorder=1)
    for i, (xc, s) in enumerate(zip(xs, stages)):
        col = PURPLE if i == 1 else '#F1EFE8'
        tc = 'white' if i == 1 else NAVY
        ax.add_patch(Circle((xc, 5.0), 0.30, color=col, ec=NAVY, lw=1.4, zorder=2))
        ax.text(xc, 5.0, str(i + 1), ha='center', va='center', color=tc, fontweight='bold', zorder=3, fontsize=11)
        ax.text(xc, 4.35, s.split('\n', 1)[1], ha='center', va='top', fontsize=8.2, color=NAVY)
    ax.add_patch(FancyArrowPatch((xs[4], 5.35), (xs[1], 5.35), connectionstyle='arc3,rad=-0.35',
                                 arrowstyle='-|>', mutation_scale=12, color=PURPLE, lw=1.3, ls='--'))
    ax.text(5.0, 5.95, u'monitora · recalibra', ha='center', color=PURPLE, fontsize=8.5)
    # valid band
    ax.add_patch(FancyBboxPatch((0.5, 1.95), 9.0, 1.45, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc='#E1F5EE', ec=TEAL, lw=1.3))
    ax.text(0.75, 3.15, u'✓ USO VÁLIDO — agregado e calibrado (ECE 0,003)', color='#0F6E56', fontweight='bold', fontsize=9.5)
    ax.text(0.75, 2.70, u'• Análise de adversário (zonas/gatilhos)    • Triagem de vídeo — lift 2,3×', color='#0F6E56', fontsize=8.6)
    ax.text(0.75, 2.35, u'• Scouting com amostra grande    • Revisão pós-jogo: xR esperado vs. realizado', color='#0F6E56', fontsize=8.6)
    # invalid band
    ax.add_patch(FancyBboxPatch((0.5, 0.35), 9.0, 1.30, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc='#FAECE7', ec=TERRA, lw=1.3))
    ax.text(0.75, 1.38, u'✗ USO INDEVIDO — veredito individual', color='#993C1D', fontweight='bold', fontsize=9.5)
    ax.text(0.75, 0.95, u'• Julgar uma jogada isolada (~84% falso alarme)', color='#993C1D', fontsize=8.6)
    ax.text(0.75, 0.60, u'• Rating de jogador individual em dados de torneio (confiabilidade 0,09)', color='#993C1D', fontsize=8.6)
    fig.tight_layout(); fig.savefig(FIG / 'fig_flow.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    fig_target(); print('fig_target ok')
    fig_ablation(); print('fig_ablation ok')
    yt, p_lr, p_gbt = fit_models()
    fig_calibration(yt, p_lr, p_gbt); print('fig_calibration ok')
    fig_lift(yt, p_lr); print('fig_lift ok')
    fig_flow(); print('fig_flow ok')
    print('figuras em', FIG)
