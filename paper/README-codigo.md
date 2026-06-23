# Código do artigo xR — guia rápido

Este diretório contém **apenas o artigo** (`artigo-xR.tex`, `artigo-xR.bib`, `figs/`). O **código
e os dados** que produzem os resultados ficam no repositório do projeto:

```
.../COE609/Analise_pressao_1/
```
Veja o `README.md` desse repositório para a hierarquia completa.

## Compilar o artigo

```
pdflatex artigo-xR
bibtex   artigo-xR
pdflatex artigo-xR
pdflatex artigo-xR
```
(No Windows com MiKTeX, use o caminho completo do `pdflatex.exe`/`bibtex.exe`.)

## De onde vem cada tabela/figura (artigo → código)

Todo o código Python está em **`src/xR_experiments/`** (exceto o notebook de §5.1 e o C++ de §5.2).

| Seção | Tabela / Figura | Script (em `Analise_pressao_1/`) |
|---|---|---|
| §5.1 Validação do *input* | — | `notebooks/pressing_v1.ipynb` |
| §5.2 Representações + platô | Tab. 1–2 | `src/xR_experiments/run2b.py` · `exp4_soccermap_cnn.py` · `src/cpp/bridge.py` |
| §5.3 Dose-resposta (alvo × geometria) | `fig_ladder` | `src/xR_experiments/target_ladder_ablation.py` · `label_noise.py` · `make_ladder_fig.py` |
| §5.4 Ablação | `fig_ablation` | `src/xR_experiments/exp6_ablation.py` |
| §5.5 Controle de *base rate* | — | `src/xR_experiments/control_baserate.py` |
| §5.6 Cross-competition | — | `src/xR_experiments/exp3_crosscomp.py` |
| §6.1 Logística vs GBT | Tabela 3 | `src/xR_experiments/eval_logit_vs_gbt.py` |
| §6.2 Calibração + ganho | `fig_calibration` · `fig_lift` | `src/xR_experiments/eval_usability.py` |
| §6.3 Confiabilidade do *rating* | Tabela 4 | `src/xR_experiments/B_survival.py` |
| §6.4 Gatilhos | Tabela 5 | `src/xR_experiments/triggers_xR.py` |
| §6.5 Contrapressão | — | `src/xR_experiments/counterpress_target.py` |
| §7 Aplicação | `fig_usecase*` · `fig_flow` | `src/xR_experiments/usecases_xR.py` · `make_figs.py` |

As figuras (`figs/*.png`) são geradas por `src/xR_experiments/make_figs.py` e `make_ladder_fig.py`,
que as gravam diretamente nesta pasta.
