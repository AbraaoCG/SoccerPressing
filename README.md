# xR — Recuperação Esperada sob Pressão

Código e dados do artigo **"xR — Recuperação Esperada sob Pressão: um modelo logístico calibrado
com alvo espacialmente restrito para análise tática no futebol"** (SBPO 2026).

O xR é a **probabilidade calibrada de a equipe que pressiona recuperar a bola** (do lado do portador,
o risco de **perder** a posse), estimada a partir do StatsBomb-360. O achado central: a **definição
do alvo** — e não a arquitetura do modelo — é a maior alavanca preditiva; restringir o sucesso no
tempo e no espaço (recuperar em ≤5 s e ≤5 m) eleva a AUC de 0,59 para 0,70.

> O artigo LaTeX (fontes, figuras e PDF compilado) fica em **`paper/`**, dentro deste repositório.
> Veja *Como gerar o artigo (PDF)* abaixo.

---

## Hierarquia do projeto (mapeada às seções do artigo)

```
Analise_pressao_1/
├── README.md                     (este arquivo)
├── requirements.txt
│
├── notebooks/                            # análises exploratórias (Jupyter)
│   ├── pressing_v1.ipynb                 → §4.1  Validação estatística do input
│   └── classify_pressing2.ipynb          → §4.2  Geometria de adversários (conceito)
│
├── src/                                  # pipeline (rodar a partir da raiz; cada script faz chdir)
│   ├── geometry_models/                  → §4.2  O platô ~0,59 (a arquitetura não move o teto)
│   │   ├── run2b.py                       — todos os modelos de geometria (regras/k-means/GNN/CNN/stacking)
│   │   └── classify_cnn_voronoi.py        — seleção por vizinhos de Voronoi (CNN)
│   │
│   ├── xR_experiments/                   → §4.3–4.5  O pivô do alvo, ablação e cross-competition
│   │   ├── express_data.py                — backbone: features tabulares + raster + 5 alvos (cache)
│   │   ├── gbt_util.py                    — fábrica do GBT (XGBoost / HistGBDT) + métricas
│   │   ├── exp1_xgboost.py                — GBT vs logística (mesmo alvo y_10s)
│   │   ├── exp2_target.py                 — varredura de alvos (o pivô: y_5s_5m → 0,70)
│   │   ├── exp3_crosscomp.py              — split aleatório vs held-out vs GroupKFold (sem leakage)
│   │   ├── exp4_soccermap_cnn.py          — CNN multicanal estilo SoccerMap
│   │   ├── exp6_ablation.py               — ablação geometria vs posição vs contexto
│   │   ├── exp5_combined.py               — (exploratório) melhor alvo + cross-comp + stacking
│   │   └── run_all.py                     — consolida express_analysis/summary.csv
│   │
│   ├── xR_paper/                         → §5–6  Modelo proposto, calibração e aplicação
│   │   ├── star_data.py / star_data_all.py — builders (FIFA WC 2022 / todas as competições) p/ §5.3
│   │   ├── xt_lite.py                      — grade de valor (xT) usada na confiabilidade
│   │   ├── B_survival.py                   — hazard + xBLV; confiabilidade do rating (0,09) §5.3
│   │   ├── eval_logit_vs_gbt.py            — Tabela 2 (logística vs GBT no alvo proposto)
│   │   ├── eval_usability.py               — calibração, PR-AUC, lift/triagem (§5.2)
│   │   └── make_figs.py                    — gera as 5 figuras do artigo (→ paper/figs/)
│   │
│   └── cpp/                              → §4.2  GNN/CNN em C++ (alta performance)
│       ├── bridge.py                       — ponte Python↔C++ (prepara dados, roda exe, AUC)
│       ├── train.cpp / nn.hpp              — GNN manual (Eigen)
│       ├── cnn_mlpack.cpp                  — CNN via mlpack 4.x
│       └── build.sh                        — compilação (MSYS2 UCRT64)
│
├── variable_analysis/  classify_analysis/  cnn_voronoi_analysis/   # tabelas geradas (ignoradas no git)
├── express_analysis/   star_analysis/                              #   "
│
├── paper/                               # artigo LaTeX — autocontido (ver "Como gerar o artigo")
│   ├── artigo-xR.tex / artigo-xR.bib     — texto e bibliografia
│   ├── sbpo-template.sty / sbpo.bst       — template SBPO 2026 (+ logo)
│   ├── figs/   artigo-xR.pdf              — 5 figuras (de make_figs.py) + PDF compilado
│   └── README-codigo.md                  — mapa artigo→código (companheiro)
│
├── Old/                                  # NÃO citado no artigo (arquivado)
│   ├── notebooks/classify_pressing.ipynb      — 1ª versão da geometria (substituída pela v2)
│   ├── scripts/{classify_geom_label,gerar_relatorio}.py
│   ├── scripts/{A_target,C_decision,compare}.py  — abordagens exploratórias A e C
│   ├── geom_label_analysis/                   — experimento de classificação de rótulo
│   └── variable_analysis/                     — relatório PDF + figuras do PDF
│
├── StatsBomb_2/      # dados StatsBomb open-data (ignorado no git)
├── cache/            # caches .pkl/.npz (ignorado no git)
└── venv1/            # ambiente Python (ignorado no git)
```

## Mapa artigo ↔ código

Cada tabela/figura do artigo é gerada por um script deste repositório:

| Seção do artigo | Tabela / Figura | Código que gera | Saída |
|---|---|---|---|
| §4.1 Validação do input | — | `notebooks/pressing_v1.ipynb` | `variable_analysis/Run1_*` |
| §4.2 Representações de entrada | **Tabela 1** | (descrição; dados de `run2b`/`exp4`) | — |
| §4.2 O platô ~0,59 | **Tabela 2** | `src/geometry_models/run2b.py` · `src/xR_experiments/exp4_soccermap_cnn.py` · `src/cpp/bridge.py` | `classify_analysis/` · `cnn_voronoi_analysis/` |
| §4.3 O pivô do alvo | **Figura 1** | `src/xR_experiments/exp2_target.py` | `express_analysis/exp2.csv` |
| §4.4 Ablação | **Figura 2** | `src/xR_experiments/exp6_ablation.py` | `express_analysis/exp6_ablation.csv` |
| §4.5 Cross-competition | (texto) | `src/xR_experiments/exp3_crosscomp.py` | `express_analysis/exp3.csv` |
| §5.1 Logística vs GBT | **Tabela 3** | `src/xR_paper/eval_logit_vs_gbt.py` | `star_analysis/logit_vs_gbt.csv` |
| §5.2 Calibração + lift | **Figuras 3–4** | `src/xR_paper/eval_usability.py` · `make_figs.py` | `star_analysis/usability.csv` |
| §5.3 Confiabilidade do rating | **Tabela 4** | `src/xR_paper/B_survival.py` (+ `star_data_all.py`) | `star_analysis/B_*.csv` |
| §6 Aplicação (fluxo do xR) | **Figura 5** | `src/xR_paper/make_figs.py` (`fig_flow`) | `paper/figs/fig_flow.png` |

> Todas as **5 figuras** do artigo são geradas por `src/xR_paper/make_figs.py`, que as grava em
> **`paper/figs/`**, junto do `artigo-xR.tex`.

## Resultados (tabelas geradas)

As pastas `*_analysis/` contêm os CSVs que alimentam as tabelas e figuras do artigo (gerados; não
versionados): `variable_analysis/` (testes de hipótese), `classify_analysis/` (platô),
`cnn_voronoi_analysis/` (Voronoi), `express_analysis/` (alvo/ablação/cross-comp) e `star_analysis/`
(calibração, usabilidade, confiabilidade).

## Como gerar o artigo (PDF)

O artigo está em **`paper/`** e é **autocontido** — só precisa de uma distribuição LaTeX instalada
(**MiKTeX** ou **TeX Live**; *não* incluída no repositório). As figuras já vêm em `paper/figs/`;
para regenerá-las dos dados, rode antes `./venv1/Scripts/python.exe src/xR_paper/make_figs.py`.

```
cd paper
pdflatex artigo-xR
bibtex   artigo-xR
pdflatex artigo-xR
pdflatex artigo-xR
```

Gera `paper/artigo-xR.pdf` (11 páginas). No Windows com MiKTeX, use o caminho completo dos
executáveis (ex.: `…\MiKTeX\miktex\bin\x64\pdflatex.exe`) ou adicione-o ao `PATH`.

## Como rodar

Pré-requisitos: `venv1/` (Python 3.14 com `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`,
`shapely`, `torch`, `xgboost`, `matplotlib`); dados em `StatsBomb_2/data/`. Para a parte C++:
MSYS2 UCRT64 (`g++`, Eigen, mlpack, armadillo).

```
# pipeline xR (a partir da raiz; cada script faz chdir para a raiz)
./venv1/Scripts/python.exe src/xR_experiments/express_data.py     # constrói o cache (91k pressões)
./venv1/Scripts/python.exe src/xR_experiments/run_all.py          # exp1..exp5 + summary
./venv1/Scripts/python.exe src/xR_experiments/exp6_ablation.py    # ablação
./venv1/Scripts/python.exe src/xR_paper/eval_logit_vs_gbt.py      # Tabela 2
./venv1/Scripts/python.exe src/xR_paper/eval_usability.py         # calibração + lift
./venv1/Scripts/python.exe src/xR_paper/make_figs.py              # figuras do artigo

# C++ (após compilar src/cpp via build.sh)
./venv1/Scripts/python.exe src/cpp/bridge.py prepare --select voronoi
./venv1/Scripts/python.exe src/cpp/bridge.py run --model gnn
```
