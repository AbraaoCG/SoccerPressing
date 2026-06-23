# -*- coding: utf-8 -*-
"""make_slides_figs.py — gera os graficos de resultados da APRESENTACAO (slides/figs/).
Numeros fixos (verificados nas analises): competicoes, plato de modelos, contrapressao, gatilhos.
Uso: ./venv1/Scripts/python.exe src/xR_experiments/make_slides_figs.py
"""
import os
from pathlib import Path
os.chdir(str(Path(__file__).resolve().parent.parent.parent))   # raiz
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = Path('slides/figs'); FIG.mkdir(parents=True, exist_ok=True)
NAVY, PURPLE, TEAL, TERRA, SAND, GRAY = '#26215C', '#534AB7', '#1D9E75', '#D85A30', '#B4B2A9', '#888780'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150, bbox_inches='tight'); plt.close(fig)


# 1) competicoes (pressoes por competicao)
def competitions():
    data = [('Euro 2020/24', 26868), ("Copa Fem. 2023", 22059), ('Euro Fem.', 18181),
            ('Copa 2022', 14504), ('Bundesliga 23/24', 10080)]
    labels = [d[0] for d in data]; vals = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.barh(labels[::-1], vals[::-1], color=NAVY)
    ax.bar_label(bars, labels=['%d' % v for v in vals[::-1]], padding=4, fontsize=11, color=NAVY)
    ax.set_xlim(0, 30000); ax.set_xlabel('pressões'); ax.set_title('91.692 pressões em 5 competições (StatsBomb 360)', color=NAVY, weight='bold')
    ax.get_xaxis().set_visible(False)
    save(fig, 'slide_competitions.png')


# 2) plato de modelos (AUC sob alvo padrao y_10s)
def plateau():
    m = [('Logística', 0.541), ('k-means', 0.551), ('GNN', 0.554), ('Set Transf.', 0.572),
         ('GCN', 0.579), ('DeepSets', 0.581), ('CNN', 0.589), ('CNN multic.', 0.593)]
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    bars = ax.bar([x[0] for x in m], [x[1] for x in m], color=PURPLE)
    ax.axhline(0.59, ls='--', color=TERRA, lw=1.5); ax.text(7.4, 0.595, 'teto $\\approx$0,59', color=TERRA, ha='right', fontsize=11)
    ax.bar_label(bars, labels=['%.3f' % x[1] for x in m], padding=2, fontsize=9.5)
    ax.set_ylim(0.5, 0.62); ax.set_ylabel('AUC (teste)')
    ax.set_title('Alvo padrão (recuperar em $\\leq$10 s) — todas as arquiteturas estacionam', color=NAVY, weight='bold')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    save(fig, 'slide_plateau.png')


# 3) contrapressao: razao CP/regular por alvo + OR controlando zona
def counterpress():
    items = [('$\\leq$10 s', 1.13, True), ('$\\leq$5 s', 1.16, True),
             ('$\\leq$5 s & 5 m\n(bruto)', 1.10, False), ('$\\leq$5 s & 5 m\n(controla zona)', 1.36, True)]
    cols = [TEAL if s else GRAY for *_, s in items]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bars = ax.bar([x[0] for x in items], [x[1] for x in items], color=cols)
    ax.axhline(1.0, ls='--', color=GRAY, lw=1.4); ax.set_ylim(0.9, 1.45)
    ax.bar_label(bars, labels=['%.2f' % x[1] for x in items], padding=3, fontsize=11, weight='bold')
    ax.set_ylabel('razão / OR  (contrapressão vs regular)')
    ax.set_title('Contrapressão: vantagem some no bruto, reaparece controlando a zona', color=NAVY, weight='bold')
    ax.text(0.99, 0.04, 'verde = significativo · cinza = n.s.', transform=ax.transAxes, ha='right', fontsize=9.5, color=GRAY)
    save(fig, 'slide_counterpress.png')


# 4) gatilhos: recuperacao por gatilho (alvo y_5s_5m), baseline 6,1%
def triggers():
    base = 6.1
    t = [('Cobrança lat.', 10.5), ('Correndo p/ trás', 6.8), ('Perto da lat.', 6.3),
         ('Passe lateral', 4.6), ('Isolado', 4.2), ('Recuo', 1.8)]
    cols = [TEAL if v >= base else TERRA for _, v in t]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    bars = ax.bar([x[0] for x in t], [x[1] for x in t], color=cols)
    ax.axhline(base, ls='--', color=NAVY, lw=1.5); ax.text(4.4, base + 0.2, 'baseline 6,1%', color=NAVY, ha='right', fontsize=10)
    ax.bar_label(bars, labels=['%.1f%%' % v for _, v in t], padding=3, fontsize=11)
    ax.set_ylabel('recuperação ($\\leq$5 s & 5 m)'); ax.set_ylim(0, 12)
    ax.set_title('Gatilhos diferem — mas não adicionam sinal preditivo ($\\Delta$AUC $\\approx$0)', color=NAVY, weight='bold')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    save(fig, 'slide_triggers.png')


if __name__ == '__main__':
    competitions(); plateau(); counterpress(); triggers()
    print('figuras de slides em', FIG)
