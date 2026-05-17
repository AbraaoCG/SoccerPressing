# -*- coding: utf-8 -*-
"""Gera o relatorio PDF de analise de variaveis (niveis de teste 1, 2 e 3)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

BASE = os.path.dirname(os.path.abspath(__file__))
VA = os.path.join(BASE, 'variable_analysis')
FIGS = os.path.join(VA, '_report_figs')
os.makedirs(FIGS, exist_ok=True)
OUT = os.path.join(BASE, 'Relatorio_Analise_Variaveis.pdf')

LEVELS = {1: 'Run1_TestLevel1', 2: 'Run1_TestLevel2', 3: 'Run1_TestLevel3'}
S = {lvl: pd.read_csv(os.path.join(VA, d, 'summary.csv')) for lvl, d in LEVELS.items()}
for lvl in S:
    S[lvl] = S[lvl].set_index('feature')

DEC_COLOR = {'INCLUIR': '#2e7d32', 'REVISAR': '#ef6c00', 'EXCLUIR': '#b0bec5'}
GLOBAL_RATE = float(S[1]['taxa_recuperacao_global'].iloc[0])

# ----------------------------------------------------------------------
# FIGURAS
# ----------------------------------------------------------------------

def fig_decisions():
    order = ['INCLUIR', 'REVISAR', 'EXCLUIR']
    counts = {lvl: S[lvl]['decisao'].value_counts() for lvl in (1, 2, 3)}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bottom = np.zeros(3)
    x = np.arange(3)
    for dec in order:
        vals = [int(counts[lvl].get(dec, 0)) for lvl in (1, 2, 3)]
        ax.bar(x, vals, bottom=bottom, label=dec, color=DEC_COLOR[dec], edgecolor='white')
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottom[i] + v / 2, str(v), ha='center', va='center',
                        color='white', fontweight='bold', fontsize=10)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(['Nivel 1\n(univariada)', 'Nivel 2\n(+VIF/Wald/LRT)', 'Nivel 3\n(+permutacao)'])
    ax.set_ylabel('Numero de variaveis')
    ax.set_title('Decisao de inclusao por nivel de teste (21 variaveis)')
    ax.legend(loc='upper right', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGS, 'decisions.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, 7.2, 3.4


def fig_effect():
    s = S[1].copy()
    s['abs_ef'] = s['tamanho_efeito'].abs()
    s = s.sort_values('abs_ef')
    dec2 = S[2]['decisao']
    cols = [DEC_COLOR[dec2[f]] for f in s.index]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y = np.arange(len(s))
    ax.barh(y, s['abs_ef'], color=cols, edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=8)
    for i, (v, t) in enumerate(zip(s['abs_ef'], s['tipo_efeito'])):
        ax.text(v + 0.004, i, '%.3f' % v, va='center', fontsize=7)
    ax.set_xlabel("|tamanho de efeito|  (Cohen d p/ continuas, Cramer's V p/ categoricas)")
    ax.set_title('Forca de associacao univariada com a recuperacao de bola')
    ax.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DEC_COLOR[d]) for d in ['INCLUIR', 'REVISAR', 'EXCLUIR']]
    ax.legend(handles, ['INCLUIR', 'REVISAR', 'EXCLUIR'], title='Decisao (Nivel 2)',
              loc='lower right', fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGS, 'effect.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, 7.2, 6.2


def fig_forest():
    s = S[2].copy()
    ok = (np.isfinite(s['or_ic_inf']) & np.isfinite(s['or_ic_sup'])
          & (s['or_ic_inf'] > 0) & np.isfinite(s['odds_ratio']))
    s = s[ok].sort_values('odds_ratio')
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(s))
    for i, (f, r) in enumerate(s.iterrows()):
        col = DEC_COLOR[r['decisao']]
        ax.plot([r['or_ic_inf'], r['or_ic_sup']], [i, i], color=col, lw=2)
        ax.plot(r['odds_ratio'], i, 'o', color=col, ms=6)
    ax.axvline(1.0, color='#555555', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel('Odds ratio (escala log) com IC 95% - modelo logistico do Nivel 2')
    ax.set_title('Odds ratios no modelo logistico com efeitos fixos de time')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGS, 'forest.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, 7.2, 5.6


def _bd(name, level=3):
    return pd.read_csv(os.path.join(VA, LEVELS[level], 'breakdown_%s.csv' % name))


def fig_cat_breakdown():
    feats = ['had_tackle', 'had_foul', 'action_under_pressure', 'zone']
    titles = ['had_tackle (desarme <=3s)', 'had_foul (falta <=3s)',
              'action_under_pressure', 'zone (terco do campo)']
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, f, t in zip(axes.ravel(), feats, titles):
        d = _bd(f)
        lab = d['faixa'].astype(str)
        rate = d['taxa_recuperacao'].values
        lo = rate - d['ic_inf'].values
        hi = d['ic_sup'].values - rate
        ax.bar(range(len(d)), rate * 100, yerr=[lo * 100, hi * 100],
               color='#1565c0', capsize=3, edgecolor='white')
        ax.axhline(GLOBAL_RATE * 100, color='#c62828', ls='--', lw=1)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(lab, fontsize=7, rotation=20, ha='right')
        ax.set_ylabel('% recuperacao')
        ax.set_title(t, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Taxa de recuperacao por categoria (linha vermelha = media geral %.1f%%)'
                 % (GLOBAL_RATE * 100), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIGS, 'cat_breakdown.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, 7.4, 5.6


def fig_cont_curves():
    feats = ['voronoi_area_carrier', 'dist_to_goal', 'nearest_opponent_dist', 'numerical_superiority']
    titles = ['voronoi_area_carrier (m2)', 'dist_to_goal (m)',
              'nearest_opponent_dist (m)', 'numerical_superiority']
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, f, t in zip(axes.ravel(), feats, titles):
        d = _bd(f)
        x = d['valor_representativo'].values
        rate = d['taxa_recuperacao'].values * 100
        ax.plot(x, rate, '-o', color='#1565c0', ms=4)
        ax.fill_between(x, d['ic_inf'].values * 100, d['ic_sup'].values * 100,
                        color='#1565c0', alpha=0.18)
        ax.axhline(GLOBAL_RATE * 100, color='#c62828', ls='--', lw=1)
        ax.set_title(t, fontsize=9)
        ax.set_ylabel('% recuperacao')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Taxa de recuperacao por faixa da variavel (banda = IC 95%)', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIGS, 'cont_curves.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, 7.4, 5.6


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------
styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=15, spaceBefore=14,
                    spaceAfter=8, textColor=colors.HexColor('#0d2c54'))
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11.5, spaceBefore=10,
                    spaceAfter=5, textColor=colors.HexColor('#1565c0'))
BODY = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.3, leading=13.5,
                      alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle('Bullet', parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle('Cap', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER,
                     textColor=colors.HexColor('#555555'), spaceBefore=3, spaceAfter=10)
TITLE = ParagraphStyle('Title', parent=styles['Title'], fontSize=22, leading=26,
                       textColor=colors.HexColor('#0d2c54'))
SUB = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER,
                     textColor=colors.HexColor('#555555'))
SMALL = ParagraphStyle('Small', parent=styles['Normal'], fontSize=7.6, leading=9)

story = []


def img(figtuple, width_cm=16.0):
    p, w, h = figtuple
    return Image(p, width=width_cm * cm, height=width_cm * cm * h / w)


def para(t, style=BODY):
    story.append(Paragraph(t, style))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BULLET, bulletText='-'))


# ---- Capa ----
story.append(Spacer(1, 3.5 * cm))
para('Relatorio de Analise de Variaveis', TITLE)
story.append(Spacer(1, 0.3 * cm))
para('Modelo de Eficacia do Pressing - UEFA Euro 2020', SUB)
story.append(Spacer(1, 0.2 * cm))
para('Validacao estatistica do input nos niveis de teste 1, 2 e 3', SUB)
story.append(Spacer(1, 1.5 * cm))
cover = [
    ['Dataset', 'UEFA Euro 2020 (StatsBomb) - 51 jogos'],
    ['Unidade de analise', '%d eventos de pressao' % int(S[1]['n_obs'].iloc[0])],
    ['Variavel resposta', 'recovered (recuperacao em ate 10 s)'],
    ['Taxa de recuperacao', '%.2f%%' % (GLOBAL_RATE * 100)],
    ['Variaveis candidatas', '21 (14 continuas, 7 categoricas/binarias)'],
    ['Execucoes analisadas', 'Nivel 1, Nivel 2 e Nivel 3'],
]
t = Table(cover, colWidths=[5 * cm, 10.5 * cm])
t.setStyle(TableStyle([
    ('FONTSIZE', (0, 0), (-1, -1), 9.5),
    ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1565c0')),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 7),
    ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
story.append(t)
story.append(PageBreak())

# ---- 1. Sumario executivo ----
para('1. Sumario Executivo', H1)
para('Este relatorio analisa os resultados da validacao estatistica do input do modelo de '
     'pressing, executada em tres niveis de profundidade de teste. Cada nivel acrescenta '
     'criterios de avaliacao: o <b>Nivel 1</b> aplica apenas a triagem univariada '
     '(Mann-Whitney / qui-quadrado); o <b>Nivel 2</b> acrescenta o diagnostico de '
     'multicolinearidade (VIF) e o modelo logistico com efeitos fixos de time, do qual saem '
     'os testes de Wald e de razao de verossimilhanca (LRT); o <b>Nivel 3</b> acrescenta o '
     'teste de permutacao para variaveis de significancia limitrofe.')
bullets([
    '<b>O nivel de teste muda radicalmente o veredito.</b> A triagem univariada (Nivel 1) '
    'aprova 19 das 21 variaveis; o teste multivariado (Niveis 2 e 3) reduz esse numero para '
    'apenas 8 aprovadas, 3 em revisao e 10 descartadas.',
    '<b>Niveis 2 e 3 produzem decisoes identicas.</b> A permutacao confirmou a unica variavel '
    'limitrofe (had_block, p-permutacao = 0,021) sem alterar nenhuma decisao - logo o teste '
    'decisivo e o LRT multivariado, nao a permutacao.',
    '<b>Significancia estatistica nao e relevancia pratica.</b> Com ~16 mil observacoes, '
    'efeitos minusculos (Cohen d ~ 0,02-0,05) atingem p < 0,05; por isso o Nivel 1 e '
    'permissivo demais e o filtro multivariado e indispensavel.',
    '<b>Multicolinearidade perfeita</b> entre n_teammates_10m, n_opponents_10m e '
    'numerical_superiority (VIF = infinito), pois a terceira e a diferenca exata das duas '
    'primeiras. So uma delas pode entrar no modelo.',
    '<b>Alerta de endogeneidade:</b> had_tackle, had_block e had_foul sao eventos que ocorrem '
    'durante/apos a pressao e praticamente coincidem com o desfecho - inflam o ajuste sem '
    'serem contexto preditivo genuino.',
    '<b>A geometria de Voronoi agrega sinal:</b> voronoi_area_carrier sobrevive a todos os '
    'niveis, confirmando a hipotese do Apendice A do projeto.',
])

# ---- 2. Dados e metodologia ----
para('2. Dados e Metodologia', H1)
para('A unidade de analise e o evento individual de pressao. A variavel resposta <b>recovered</b> '
     'indica se o time pressionante recuperou a bola em ate 10 segundos. Foram avaliadas 21 '
     'variaveis candidatas, organizadas em cinco grupos (features de freeze frame 360, '
     'posicionais, de contexto de jogo, taticas e acoes defensivas concorrentes), mais as '
     'features geometricas de Voronoi. A taxa de recuperacao observada foi de %.2f%%.'
     % (GLOBAL_RATE * 100))
para('Cada variavel passa por uma bateria de testes cuja profundidade e definida pela flag '
     'TEST_LEVEL. A regra de decisao aplicada e: <b>EXCLUIR</b> se nao houver associacao '
     'univariada (Benjamini-Hochberg); <b>REVISAR</b> se VIF >= 5 (multicolinearidade); '
     '<b>EXCLUIR</b> se nao significativa no LRT; <b>INCLUIR</b> caso contrario.')
lvltab = [
    ['Nivel', 'Testes executados', 'Funcao'],
    ['1', 'Mann-Whitney U / Qui-quadrado + correcao BH', 'Triagem univariada (associacao bruta)'],
    ['2', '+ VIF, modelo logistico (Wald), LRT', 'Filtro multivariado e multicolinearidade'],
    ['3', '+ teste de permutacao (1000x)', 'Confirmacao de variaveis limitrofes'],
]
t = Table(lvltab, colWidths=[1.4 * cm, 8.1 * cm, 6 * cm])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d2c54')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.3),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4fa')]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(t)
story.append(Spacer(1, 0.3 * cm))

# ---- 3. Resultados por nivel ----
para('3. Resultados por Nivel de Teste', H1)
story.append(img(fig_decisions(), 14.5))
para('Figura 1. Evolucao das decisoes de inclusao conforme o nivel de teste.', CAP)

n1 = S[1]['decisao'].value_counts()
n2 = S[2]['decisao'].value_counts()
para('3.1 Nivel 1 - triagem univariada', H2)
para('Apenas <b>minute</b> e <b>score_diff</b> sao reprovadas (sem associacao univariada com '
     'a recuperacao). As outras 19 variaveis aparecem como significativas. Esse resultado, '
     'porem, e enganoso: o tamanho amostral (~16 mil pressoes) torna o teste sensivel a '
     'efeitos triviais. Variaveis como pressing_compactness (Cohen d = -0,058) ou '
     'voronoi_area_presser (d = -0,001) sao "significativas" sem nenhuma relevancia pratica.')
para('3.2 Nivel 2 - VIF, Wald e LRT', H2)
para('Ao introduzir o modelo logistico multivariado com efeitos fixos de time, o quadro muda '
     'por completo: das 19 aprovadas no Nivel 1, restam <b>8 INCLUIR</b>, <b>3 REVISAR</b> e '
     '<b>10 EXCLUIR</b>. O LRT e o criterio decisivo - ele mede se a variavel agrega '
     'informacao <i>dado o restante do modelo</i>. Muitas features 360 (n_adversaries_5m, '
     'nearest_opponent_dist, nearest_teammate_dist, pressing_compactness) perdem '
     'significancia porque seu sinal ja e capturado por features correlacionadas. As tres '
     'variaveis de contagem caem em REVISAR por multicolinearidade perfeita (VIF = infinito).')
para('3.3 Nivel 3 - teste de permutacao', H2)
para('O Nivel 3 nao alterou nenhuma decisao em relacao ao Nivel 2. A unica variavel em faixa '
     'limitrofe (had_block, p-BH = 0,024) foi submetida a 1000 permutacoes do rotulo, '
     'obtendo p-permutacao = 0,021 - confirmando a associacao. <b>Conclusao pratica:</b> '
     'para este conjunto de dados, o teste decisivo e o LRT multivariado; a permutacao '
     'funciona como verificacao de robustez, nao como filtro adicional.')

story.append(PageBreak())

# ---- Tabela comparativa ----
para('3.4 Comparativo de decisoes por variavel', H2)
hdr = ['Variavel', 'Grupo', '|Efeito|', 'p-BH', 'VIF', 'p-LRT', 'N1', 'N2', 'N3']
rows = [hdr]
GRP_ABBR = {'Freeze frame 360': '360', 'Posicional': 'Posic.', 'Contexto de jogo': 'Contexto',
            'Tatica': 'Tatica', 'Acoes defensivas concorrentes': 'Acoes def.', 'Voronoi': 'Voronoi'}
feat_order = list(S[1].sort_values('grupo').index)
for f in feat_order:
    r1, r2, r3 = S[1].loc[f], S[2].loc[f], S[3].loc[f]
    vif = r2['VIF']
    vif_s = 'inf' if (pd.notna(vif) and np.isinf(vif)) else ('%.1f' % vif if pd.notna(vif) else '-')
    plrt = r2['p_lrt']
    plrt_s = ('%.3g' % plrt) if pd.notna(plrt) else '-'
    rows.append([f, GRP_ABBR.get(r1['grupo'], r1['grupo']),
                 '%.3f' % abs(r1['tamanho_efeito']),
                 '%.2g' % r1['p_bh'], vif_s, plrt_s,
                 r1['decisao'][:3], r2['decisao'][:3], r3['decisao'][:3]])
t = Table(rows, colWidths=[3.6 * cm, 1.9 * cm, 1.5 * cm, 1.7 * cm, 1.3 * cm, 1.7 * cm,
                           1.25 * cm, 1.25 * cm, 1.25 * cm], repeatRows=1)
tstyle = [
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d2c54')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 7.4),
    ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#dddddd')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
]
for i, f in enumerate(feat_order, start=1):
    for col, lvl in [(6, 1), (7, 2), (8, 3)]:
        dec = S[lvl].loc[f, 'decisao']
        tstyle.append(('TEXTCOLOR', (col, i), (col, i), colors.HexColor(DEC_COLOR[dec])))
        tstyle.append(('FONTNAME', (col, i), (col, i), 'Helvetica-Bold'))
t.setStyle(TableStyle(tstyle))
story.append(t)
para('Tabela 1. N1/N2/N3 = decisao nos niveis 1, 2 e 3 (INC=INCLUIR, REV=REVISAR, EXC=EXCLUIR). '
     '|Efeito| e p-BH vem da triagem univariada; VIF e p-LRT do modelo do Nivel 2.', CAP)

story.append(PageBreak())

# ---- 4. Analise das features ----
para('4. Analise das Features', H1)
para('4.1 Forca de associacao univariada', H2)
para('A Figura 2 ordena as variaveis pelo tamanho de efeito. Tres variaveis se destacam: '
     '<b>had_tackle</b> (Cramer V = 0,307), <b>had_foul</b> (0,163) e '
     '<b>action_under_pressure</b> (0,118). A maioria das features 360 e de Voronoi tem '
     'efeito pequeno (|d| < 0,13). As barras estao coloridas pela decisao final (Nivel 2): '
     'note que varias variaveis cinza (EXCLUIR) tem efeito comparavel a algumas verdes - a '
     'diferenca esta na redundancia detectada pelo modelo multivariado.')
story.append(img(fig_effect(), 13.5))
para('Figura 2. Tamanho de efeito por variavel (cor = decisao no Nivel 2).', CAP)

story.append(PageBreak())
para('4.2 Multicolinearidade', H2)
para('O diagnostico VIF do Nivel 2 revela <b>multicolinearidade perfeita</b> entre tres '
     'features de contagem: n_teammates_10m, n_opponents_10m e numerical_superiority '
     'apresentam VIF = infinito. A causa e estrutural - numerical_superiority e definida '
     'como n_teammates_10m menos n_opponents_10m, ou seja, uma combinacao linear exata das '
     'outras duas. No modelo logistico isso produz coeficientes instaveis (odds ratios com '
     'IC de 0 a infinito) e p-valores sem sentido (p ~ 1,0). <b>Recomendacao:</b> manter '
     'apenas uma das tres - preferencialmente numerical_superiority, que resume a relacao '
     'numerica local em um unico indicador interpretavel.')
para('4.3 Significancia no modelo logistico', H2)
para('A Figura 3 mostra os odds ratios do modelo do Nivel 2. had_tackle tem OR ~ 5,6 (forte '
     'associacao positiva) e had_foul OR ~ 0,07 (forte associacao negativa - a falta '
     'interrompe a jogada). As variaveis continuas aparecem proximas de 1,0 porque o OR e '
     'medido <i>por unidade</i>: dist_to_goal tem OR 0,995 por metro, mas ao longo do '
     'intervalo interquartil (~41 m) o efeito acumulado e relevante (~0,81). O mesmo vale '
     'para voronoi_area_carrier.')
story.append(img(fig_forest(), 13.5))
para('Figura 3. Odds ratios (escala log, IC 95%) do modelo logistico do Nivel 2. '
     'As tres variaveis com VIF infinito foram omitidas por terem IC degenerado.', CAP)

story.append(PageBreak())
para('4.4 Curvas de recuperacao', H2)
para('As figuras abaixo usam as tabelas de breakdown - taxa de recuperacao desagregada por '
     'faixa ou categoria de cada variavel.')
story.append(img(fig_cat_breakdown(), 15.5))
para('Figura 4. Taxa de recuperacao por categoria. had_tackle praticamente triplica a taxa '
     '(66% vs 23%); had_foul quase a zera (3%); sob drible o adversario e mais vulneravel '
     '(42%); o pressing no terco defensivo e o mais produtivo (32%).', CAP)
story.append(img(fig_cont_curves(), 15.5))
para('Figura 5. Taxa de recuperacao por faixa de variaveis continuas. Celulas de Voronoi '
     'menores ao redor do portador (espaco mais fechado) elevam a recuperacao; a relacao '
     'com dist_to_goal tem formato de U; a superioridade numerica local aumenta a '
     'recuperacao de forma monotonica.', CAP)

story.append(PageBreak())

# ---- 5. Achados criticos ----
para('5. Achados Criticos', H1)
para('5.1 Endogeneidade das acoes defensivas concorrentes', H2)
para('had_tackle, had_block e had_foul registram se houve desarme, bloqueio ou falta numa '
     'janela de 3 s <i>apos</i> a pressao. Essa janela se sobrepoe a janela de 10 s que '
     'define a propria recuperacao. Um desarme bem-sucedido frequentemente <i>e</i> a '
     'recuperacao; uma falta interrompe a jogada e impede qualquer recuperacao. Logo, essas '
     'variaveis nao descrevem o <i>contexto</i> da pressao - elas sao quase o proprio '
     'desfecho. Incluidas no modelo, inflam artificialmente o ajuste (had_tackle sozinha '
     'reduz o AIC em ~807 pontos). <b>Recomendacao:</b> trata-las como mediadores/desfechos, '
     'analisados a parte, e nao como preditores de contexto.')
para('5.2 Significancia estatistica versus relevancia pratica', H2)
para('Com 15.958 observacoes, o poder estatistico e altissimo: praticamente qualquer '
     'diferenca nao-nula vira "significativa". A triagem univariada do Nivel 1 aprova 19 '
     'variaveis justamente por isso. O criterio que separa sinal de ruido nao e o p-valor '
     'isolado, mas a combinacao de <b>tamanho de efeito</b> e <b>contribuicao incremental</b> '
     '(LRT) - exatamente o que os Niveis 2 e 3 adicionam.')
para('5.3 Redundancia das features 360', H2)
para('Reprovar no LRT nao significa ser irrelevante - significa ser <i>redundante dado o '
     'resto do modelo</i>. n_adversaries_5m, n_opponents_10m e nearest_opponent_dist medem '
     'facetas correlacionadas da mesma realidade (densidade de adversarios ao redor da '
     'bola). O modelo so precisa de uma delas. A engenharia de features deveria consolidar '
     'esse grupo em um unico indicador de pressao adversaria.')
para('5.4 Valor das features de Voronoi', H2)
para('Entre as tres features geometricas, <b>voronoi_area_carrier</b> e a unica que '
     'sobrevive a todos os niveis (INCLUIR), com p-LRT = 0,003 e reducao de AIC. A area da '
     'celula de Voronoi do portador captura o quao fechado esta o espaco a sua volta - um '
     'sinal que as contagens simples de adversarios nao reproduzem. Isso da suporte '
     'empirico a extensao do Apendice A do projeto. Ja voronoi_area_presser e '
     'voronoi_n_opp_neighbors sao redundantes e podem ficar de fora.')

# ---- 6. Conjunto recomendado ----
para('6. Conjunto de Features Recomendado', H1)
para('Combinando o veredito do pipeline (Niveis 2 e 3) com o alerta de endogeneidade da '
     'Secao 5.1, propoe-se a seguinte especificacao para o modelo de <i>contexto</i> de '
     'pressing:')
rec = [
    ['Variavel', 'Grupo', 'Papel'],
    ['dist_to_goal', 'Posicional', 'Preditor de contexto - INCLUIR'],
    ['zone', 'Posicional', 'Preditor de contexto - INCLUIR'],
    ['action_under_pressure', 'Tatica', 'Preditor de contexto - INCLUIR'],
    ['trigger_type', 'Tatica', 'Preditor de contexto - INCLUIR'],
    ['voronoi_area_carrier', 'Voronoi', 'Preditor geometrico - INCLUIR'],
    ['numerical_superiority', 'Freeze frame 360', 'Manter so esta do trio colinear'],
    ['had_tackle / had_block / had_foul', 'Acoes def.', 'Tratar como desfecho/mediador, nao preditor'],
    ['n_teammates_10m / n_opponents_10m', 'Freeze frame 360', 'Remover (colinearidade com numerical_superiority)'],
    ['minute, score_diff, duration, etc.', 'Diversos', 'Excluir (sem associacao ou redundantes)'],
]
t = Table(rec, colWidths=[5.6 * cm, 3.3 * cm, 6.6 * cm])
ts = [
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d2c54')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#dddddd')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]
for i in range(1, 6):
    ts.append(('TEXTCOLOR', (0, i), (0, i), colors.HexColor('#2e7d32')))
    ts.append(('FONTNAME', (0, i), (0, i), 'Helvetica-Bold'))
t.setStyle(TableStyle(ts))
story.append(t)
story.append(Spacer(1, 0.3 * cm))

# ---- 7. Conclusoes ----
para('7. Conclusoes e Proximos Passos', H1)
bullets([
    'O <b>nivel de teste</b> e determinante: a triagem univariada e otimista demais; o '
    'filtro multivariado (Niveis 2/3) e o que produz um conjunto de features defensavel.',
    'Os Niveis 2 e 3 sao equivalentes em decisao - basta rodar o Nivel 2 para o veredito, '
    'reservando a permutacao (Nivel 3) para auditoria de variaveis limitrofes.',
    'Corrigir a engenharia de features: eliminar a colinearidade perfeita do trio de '
    'contagens e consolidar as features 360 redundantes.',
    'Reposicionar had_tackle/had_block/had_foul como desfechos, evitando vazamento de '
    'informacao no modelo de contexto.',
    'Proximo passo: ajustar o modelo logistico final com o conjunto recomendado da Secao 6, '
    'avaliar AUC por validacao cruzada e seguir para as analises de residuos por jogador e '
    'de contrapressao.',
])

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        title='Relatorio de Analise de Variaveis - Pressing')


def footer(canvas, d):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#999999'))
    canvas.drawString(2.2 * cm, 1.1 * cm,
                      'Analise de Pressing - UEFA Euro 2020 | variable_analysis')
    canvas.drawRightString(A4[0] - 2.2 * cm, 1.1 * cm, 'Pagina %d' % d.page)
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('PDF gerado:', OUT)
