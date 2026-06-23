# -*- coding: utf-8 -*-
"""make_ladder_fig.py — figura da DOSE-RESPOSTA: AUC (todas as variaveis) e AUC (so geometria)
ao longo da escada de alvos, ordenados do mais frouxo ao mais local. Mostra que a geometria fica
mais preditiva conforme o alvo localiza. Le star_analysis/target_ladder.csv.
Saida: paper/figs/fig_ladder.png. Uso: ./venv1/Scripts/python.exe src/xR_experiments/make_ladder_fig.py
"""
import os
from pathlib import Path
os.chdir(str(Path(__file__).resolve().parent.parent.parent))   # raiz
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY, TEAL, TERRA = '#26215C', '#1D9E75', '#D85A30'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

df = pd.read_csv('star_analysis/target_ladder.csv').set_index('alvo')
order = ['y_10s', 'y_5s', 'y_2act', 'y_5s_9m', 'y_5s_5m']
labels = ['10 s\n(sem raio)', '5 s\n(sem raio)', '2 ações', '5 s · 9 m', '5 s · 5 m']
all_auc = [df.loc[t, 'AUC_all'] for t in order]
geom = [df.loc[t, 'AUC_geom'] for t in order]

fig, ax = plt.subplots(figsize=(7.2, 3.9))
x = range(len(order))
ax.plot(x, all_auc, '-o', color=NAVY, lw=2, label='todas as variáveis')
ax.plot(x, geom, '-s', color=TEAL, lw=2, label='só geometria')
for xi, v in zip(x, all_auc):
    ax.annotate('%.3f' % v, (xi, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=9, color=NAVY)
for xi, v in zip(x, geom):
    ax.annotate('%.3f' % v, (xi, v), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=9, color=TEAL)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylim(0.55, 0.72); ax.set_ylabel('AUC (teste)')
ax.set_title('Quanto mais local o alvo, mais preditiva a geometria', color=NAVY, weight='bold')
ax.annotate('alvo mais local  →', xy=(0.5, -0.26), xycoords='axes fraction', ha='center', fontsize=10, color=TERRA, style='italic')
ax.legend(loc='upper left', frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig('paper/figs/fig_ladder.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('salvo: paper/figs/fig_ladder.png')
