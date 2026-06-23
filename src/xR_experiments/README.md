# `xR_experiments/` — índice do código por categoria

Todo o código Python que gera resultados do artigo **xR** está nesta pasta. Este índice agrupa os
scripts pelo **papel** e pela **seção do artigo** que cada um sustenta, para um revisor localizar
rápido o que olhar. O mapeamento completo *artigo → tabela/figura → saída* está no
[`README.md` da raiz](../../README.md#mapa-artigo--código).

> **Como rodar:** sempre a partir da **raiz do repositório** (cada script faz `chdir` para a raiz
> e grava em `*_analysis/`, `paper/figs/` ou `slides/figs/`). Ex.:
> `./venv1/Scripts/python.exe src/xR_experiments/exp2_target.py`

**Legenda:** ▶ = *entry-point* (rode para reproduzir um resultado) · 📦 = *backbone* (biblioteca
importada pelos outros; **não** roda isolado).

---

## 📦 Backbone / infraestrutura

Não produzem um resultado do artigo sozinhos; são importados pelos *entry-points*.

| Script | Papel | Saída |
|---|---|---|
| `express_data.py` | Backbone: `geom_of` (atributos de geometria), os 5 alvos e o cache | `cache/_express_*` |
| `gbt_util.py` | Fábrica do GBT (XGBoost / HistGBDT) + métricas padronizadas | — |
| `star_data.py` | *Builder* da FIFA WC 2022 (comp 43) com tempo/local da perda | `cache/_star43.pkl` |
| `star_data_all.py` | *Builder* de todas as competições com 360 | `cache/_star_all.pkl` |
| `xt_lite.py` | Grade de valor de ameaça (xT) usada no xBLV/confiabilidade | — |

## ▶ §5 — Evolução experimental

| Script | O que mede | Seção | Saída |
|---|---|---|---|
| `run2b.py` | Modelos de geometria (regras/k-means/GNN/CNN/*stacking*) → o platô ~0,59 | §5.2 | `classify_analysis/` |
| `classify_cnn_voronoi.py` | Seleção de adversários por vizinhos de Voronoi (CNN) | §5.2 | `cnn_voronoi_analysis/` |
| `exp4_soccermap_cnn.py` | CNN multicanal estilo SoccerMap | §5.2 | `express_analysis/exp4.csv` |
| `exp1_xgboost.py` | GBT *vs.* logística no mesmo alvo `y_10s` (o modelo não move o teto) | §5.2 | `express_analysis/exp1.csv` |
| `exp2_target.py` | Varredura de alvos — o pivô (`y_5s_5m`) | §5.3 | `express_analysis/exp2.csv` |
| `target_ladder_ablation.py` | **Dose-resposta:** AUC só-geometria ao longo da escada de alvos | §5.3 | `star_analysis/target_ladder.csv` |
| `label_noise.py` | Ruído de rótulo (76% das recuperações ≤10 s a >5 m/>5 s; mediana 13 m) | §5.3 | `star_analysis/label_noise.csv` |
| `exp6_ablation.py` | Ablação geometria *vs.* posição *vs.* contexto | §5.4 | `express_analysis/exp6_ablation.csv` |
| `control_baserate.py` | Controle: *downsample* da base + IC95% *bootstrap* do *gap* de AUC | §5.5 | `star_analysis/control_baserate.csv` |
| `exp3_crosscomp.py` | Split aleatório *vs.* held-out *vs.* GroupKFold (sem *leakage*) | §5.6 | `express_analysis/exp3.csv` |
| `run_all.py` | Orquestra `exp1..exp5` e consolida o resumo | — | `express_analysis/summary.csv` |

## ▶ §6 — O modelo proposto (xR logístico calibrado)

| Script | O que mede | Seção | Saída |
|---|---|---|---|
| `eval_logit_vs_gbt.py` | Logística *vs.* GBT no alvo proposto (usabilidade × ganho marginal) | §6.1 | `star_analysis/logit_vs_gbt.csv` |
| `eval_usability.py` | Calibração (ECE/*Brier*), PR-AUC e curva de ganho/*lift* | §6.2 | `star_analysis/usability.csv` |
| `B_survival.py` | *Hazard* + xBLV; **confiabilidade split-half** do rating (0,09) | §6.3 | `star_analysis/B_*.csv` |
| `triggers_xR.py` | O tipo de gatilho adiciona sinal? (métricas padronizadas) | §6.4 | `star_analysis/triggers_*.csv` |
| `counterpress_target.py` | A vantagem persiste sob contrapressão e o alvo estrito? | §6.5 | `star_analysis/counterpress_*.csv` |

## ▶ §7 — Aplicação prática

| Script | O que faz | Seção | Saída |
|---|---|---|---|
| `usecases_xR.py` | Treina em 47 jogos, testa em 4 held-out; casos de uso + figura passo a passo | §7 | `star_analysis/usecases.md` · `paper/figs/fig_usecase2.png` |

## ▶ Figuras (artigo + slides)

| Script | Gera | Destino |
|---|---|---|
| `make_figs.py` | Figuras do artigo (ablação, calibração, *lift*, fluxo, caso de uso) | `paper/figs/` |
| `make_ladder_fig.py` | `fig_ladder` (dose-resposta) | `paper/figs/fig_ladder.png` |
| `make_slides_figs.py` | Gráficos dos slides (competições, platô, contrapressão, gatilhos) | `slides/figs/` |
| `make_slides_extra.py` | Slides de *base rate* e de confiabilidade *split-half* | `slides/figs/` |

## ▷ Exploratórios (não citados no texto final)

Sustentam o método/decisões, mas **não** produzem um resultado do texto final — um revisor pode pular.

| Script | O que era | Saída |
|---|---|---|
| `exp5_combined.py` | Pipeline combinado (melhor alvo + cross-comp + *stacking*) | `express_analysis/exp5.csv` |
| `target_y3s10m.py` | Checagem pontual do alvo `y_3s_10m` (logística) | (impressão no console) |
