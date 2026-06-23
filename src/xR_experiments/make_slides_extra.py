# -*- coding: utf-8 -*-
"""make_slides_extra.py — figuras extras dos slides: (1) controle de base rate; (2) confiabilidade
split-half (esquema + histograma de pressões por jogador). Saída: slides/figs/.
Uso: ./venv1/Scripts/python.exe src/xR_experiments/make_slides_extra.py
"""
import os
from pathlib import Path
os.chdir(str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path('slides/figs'); FIG.mkdir(parents=True, exist_ok=True)
NAVY, PURPLE, TEAL, TERRA, SAND, GRAY = '#26215C', '#534AB7', '#1D9E75', '#D85A30', '#B4B2A9', '#888780'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})


# 1) controle de base rate
def baserate():
    labels = ['y_10s\n(base 29%)', 'y_10s ↓\n(base 6,9%)', 'y_5s_5m\n(base 6,9%)']
    vals = [0.571, 0.573, 0.684]; cols = [NAVY, '#9a95d8', TEAL]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.bar(labels, vals, color=cols)
    ax.bar_label(bars, labels=['%.3f' % v for v in vals], padding=3, fontsize=12, weight='bold')
    ax.set_ylim(0.5, 0.72); ax.set_ylabel('AUC (logística)')
    ax.set_title('Igualar o base rate NÃO muda a AUC', color=NAVY, weight='bold')
    ax.annotate('mesmo base rate', xy=(1, 0.60), xytext=(1, 0.535), ha='center', fontsize=9.5, color=GRAY)
    fig.tight_layout(); fig.savefig(FIG / 'slide_baserate.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# 2) confiabilidade split-half: esquema + histograma de pressoes/jogador
def reliability():
    df = pd.read_pickle('cache/_star_all.pkl')
    vc = df['carrier_id'].dropna().value_counts()
    med = float(vc.median())
    fig, (axd, axh) = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={'width_ratios': [1.05, 1]})

    # esquema (esquerda)
    axd.axis('off'); axd.set_xlim(0, 10); axd.set_ylim(0, 10)
    axd.text(5, 9.3, 'Jogos de um jogador', ha='center', fontsize=11, weight='bold', color=NAVY)
    for (x0, lab, col) in [(0.4, 'Metade A\n→ rating A', PURPLE), (5.4, 'Metade B\n→ rating B', TEAL)]:
        axd.add_patch(FancyBboxPatch((x0, 5.0), 4.2, 2.8, boxstyle='round,pad=0.1,rounding_size=0.3',
                                     linewidth=1.6, edgecolor=col, facecolor='white'))
        axd.text(x0 + 2.1, 6.4, lab, ha='center', va='center', fontsize=10.5, color=col)
    axd.add_patch(FancyArrowPatch((4.7, 3.2), (4.7, 4.8), arrowstyle='-', color=GRAY, lw=0))
    axd.annotate('', xy=(6.0, 4.9), xytext=(4.0, 4.9), arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.8))
    axd.text(5, 3.6, 'correlação entre jogadores\n= 0,09  (≈ 0 → ruído)', ha='center', va='center',
             fontsize=11, color=TERRA, weight='bold')

    # histograma (direita)
    axh.hist(np.clip(vc.values, 0, 60), bins=24, color=NAVY, alpha=0.85)
    axh.axvline(med, color=TERRA, ls='--', lw=1.8)
    axh.text(med + 1.5, axh.get_ylim()[1] * 0.85, 'mediana %.0f' % med, color=TERRA, fontsize=10)
    axh.set_xlabel('pressões sofridas por jogador'); axh.set_ylabel('nº de jogadores')
    axh.set_title('Poucos eventos por jogador', color=NAVY, weight='bold')
    fig.tight_layout(); fig.savefig(FIG / 'slide_reliability.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('mediana pressoes/jogador:', med, '| jogadores:', len(vc))


if __name__ == '__main__':
    baserate(); reliability()
    print('figuras extras em', FIG)
