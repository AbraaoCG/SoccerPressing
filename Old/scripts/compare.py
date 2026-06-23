# -*- coding: utf-8 -*-
"""compare.py — consolida A/B/C (FIFA WC 2022) numa tabela comparativa. star_analysis/compare.csv."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

OUT = 'star_analysis'
A = pd.read_csv(os.path.join(OUT, 'A_target.csv'))
B = pd.read_csv(os.path.join(OUT, 'B_survival.csv'))
C = pd.read_csv(os.path.join(OUT, 'C_decision.csv'))

a_alvo = A[A['parte'] == 'alvo'].sort_values('auc', ascending=False).iloc[0]
a_abl = A[A['parte'] == 'ablacao']
geom = float(a_abl[a_abl['bloco'] == 'GEOM']['auc'].iloc[0])
allf = float(a_abl[a_abl['bloco'] == 'ALL']['auc'].iloc[0])
poscx = float(a_abl[a_abl['bloco'] == 'POS+CTX']['auc'].iloc[0])

rows = [
    dict(abordagem='A — insight do alvo', prediz='P(perde|contexto)',
         auc_principal=round(float(a_alvo['auc']), 4), metrica='auc(%s)' % a_alvo['alvo'],
         reliab_player='n/a (modelo de contexto)', leve='sim (GBT, ~s)',
         nota='geometria isolada %.3f | marginal +%.3f' % (geom, allf - poscx)),
    dict(abordagem='B — survival + xBLV', prediz='hazard -> P(perde<=5s) + valor',
         auc_principal=round(float(B['auc_contexto'].iloc[0]), 4), metrica='auc(y_5s)',
         reliab_player='%.3f (Spearman, n=%d)' % (B['reliab_splithalf'].iloc[0], B['n_jogadores_rel'].iloc[0]),
         leve='sim (GBT person-period)', nota='= A em predicao; rating de jogador instavel'),
    dict(abordagem='C — contrafactual decisao', prediz='P(reter|opcao) -> regret',
         auc_principal=round(float(C['auc_opcao'].iloc[0]), 4), metrica='auc(reter|opcao)',
         reliab_player='%s (so %d jogador qualifica)' % (str(C['reliab_splithalf'].iloc[0]), C['n_jogadores_rel'].iloc[0]),
         leve='sim, mas dados escassos', nota='so 2798 passes rotulados; metrica nao validavel aqui'),
]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'compare.csv'), index=False)
print('=== COMPARACAO DAS 3 ABORDAGENS (FIFA WC 2022) ===')
print(df.to_string(index=False))
print('\nsalvo em', os.path.join(OUT, 'compare.csv'))
