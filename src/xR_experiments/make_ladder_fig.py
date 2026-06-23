# -*- coding: utf-8 -*-
"""make_ladder_fig.py — figura da DOSE-RESPOSTA: AUC (todas as variaveis) e AUC (so geometria).
LINHA PRINCIPAL = eixos naturais de localidade do alvo (tempo -> espaco), 1 variavel por passo.
O alvo por SEQUENCIA (2 acoes), de eixo distinto, e exibido A PARTE como ponto corroborativo.
Mostra que a geometria fica mais preditiva conforme o alvo localiza. Le star_analysis/target_ladder.csv.
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
# LINHA PRINCIPAL: eixos naturais de localidade (tempo -> espaco), 1 variavel por passo
main = ['y_10s', 'y_5s', 'y_5s_9m', 'y_5s_5m']
labels = ['10 s\n(sem raio)', '5 s\n(sem raio)', '5 s · 9 m', '5 s · 5 m']
all_auc = [df.loc[t, 'AUC_all'] for t in main]
geom = [df.loc[t, 'AUC_geom'] for t in main]
x = list(range(len(main)))

fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(x, all_auc, '-o', color=NAVY, lw=2, label='todas as variáveis')
ax.plot(x, geom, '-s', color=TEAL, lw=2, label='só geometria')
for xi, v in zip(x, all_auc):
    ax.annotate('%.3f' % v, (xi, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=9, color=NAVY)
for xi, v in zip(x, geom):
    ax.annotate('%.3f' % v, (xi, v), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=9, color=TEAL)

# y_2act: eixo DIFERENTE (sequencia, sem teto de m/s) -> ponto solto, NAO conectado a linha
x2 = 1.5
a2, g2 = df.loc['y_2act', 'AUC_all'], df.loc['y_2act', 'AUC_geom']
ax.plot([x2], [a2], marker='D', mfc='white', mec=NAVY, mew=1.6, ms=8, ls='none')
ax.plot([x2], [g2], marker='D', mfc='white', mec=TEAL, mew=1.6, ms=8, ls='none',
        label='2 ações (eixo de sequência)')
ax.annotate('2 ações\n(eixo de sequência)', (x2, g2), textcoords='offset points',
            xytext=(10, -27), ha='left', fontsize=8, color=TERRA, style='italic',
            arrowprops=dict(arrowstyle='-', color=TERRA, lw=0.8))

ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlim(-0.4, 3.4)
ax.set_ylim(0.55, 0.72); ax.set_ylabel('AUC (teste)')
ax.set_title('Quanto mais local o alvo, mais preditiva a geometria', color=NAVY, weight='bold')
ax.annotate('alvo mais local  →', xy=(0.5, -0.28), xycoords='axes fraction', ha='center', fontsize=10, color=TERRA, style='italic')
ax.legend(loc='upper left', frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig('paper/figs/fig_ladder.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('salvo: paper/figs/fig_ladder.png')
