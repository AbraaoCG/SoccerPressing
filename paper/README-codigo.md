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

| Seção | Tabela / Figura | Script (em `Analise_pressao_1/`) |
|---|---|---|
| §4.2 Representações | Tabela 1 | (descrição) |
| §4.2 Platô ~0,59 | Tabela 2 | `src/geometry_models/run2b.py`, `src/cpp/bridge.py` |
| §4.3 Pivô do alvo | Figura 1 | `src/xR_experiments/exp2_target.py` |
| §4.4 Ablação | Figura 2 | `src/xR_experiments/exp6_ablation.py` |
| §4.5 Cross-competition | (texto) | `src/xR_experiments/exp3_crosscomp.py` |
| §5.1 Logística vs GBT | Tabela 3 | `src/xR_paper/eval_logit_vs_gbt.py` |
| §5.2 Calibração + lift | Figuras 3–4 | `src/xR_paper/eval_usability.py` |
| §5.3 Confiabilidade | Tabela 4 | `src/xR_paper/B_survival.py` |
| §6 Aplicação | Figura 5 | `src/xR_paper/make_figs.py` |

**As 5 figuras** (`figs/*.png`) são geradas por `src/xR_paper/make_figs.py`, que as grava
diretamente nesta pasta.
