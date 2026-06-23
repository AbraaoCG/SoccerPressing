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
├── src/                                  # todo o código (rodar a partir da raiz; cada script faz chdir)
│   ├── xR_experiments/                   # TODOS os scripts Python que geram resultados do artigo
│   │   ├── express_data.py · gbt_util.py      — backbone (features/5 alvos/cache) + fábrica do GBT
│   │   ├── exp1..exp6 · run_all.py            — varredura de alvos, ablação, cross-comp, CNN, platô
│   │   ├── target_ladder_ablation.py          — dose-resposta: AUC só-geometria por alvo (fig_ladder)
│   │   ├── label_noise.py · control_baserate.py — ruído de rótulo (76%) e controle de base rate
│   │   ├── run2b.py · classify_cnn_voronoi.py — modelos de geometria (platô) e seleção por Voronoi
│   │   ├── star_data*.py · xt_lite.py · B_survival.py — builders + confiabilidade do rating (0,09)
│   │   ├── eval_logit_vs_gbt.py · eval_usability.py   — logística vs GBT, calibração, lift
│   │   ├── triggers_xR.py · counterpress_target.py    — análises complementares (gatilhos, contrapressão)
│   │   ├── usecases_xR.py                     — exemplos de uso (held-out) + figura passo a passo
│   │   └── make_figs.py · make_ladder_fig.py · make_slides_*.py — figuras do artigo e dos slides
│   │
│   └── cpp/                              # GNN/CNN em C++ — gera a linha "GNN Voronoi" do platô
│       └── bridge.py · train.cpp · nn.hpp · cnn_mlpack.cpp · build.sh
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

Todo o código Python que gera resultados do artigo está em **`src/xR_experiments/`** (caminhos
abaixo relativos a essa pasta, salvo indicação). Exceções por natureza: o notebook de §5.1 e o
pipeline C++ de §5.2 (linha "GNN Voronoi" do platô). Para um **índice dos scripts agrupados por
categoria e papel** (backbone *vs.* entry-point), veja
[`src/xR_experiments/README.md`](src/xR_experiments/README.md).

| Seção do artigo | Tabela / Figura | Código que gera | Saída |
|---|---|---|---|
| §5.1 Validação do *input* | — | `notebooks/pressing_v1.ipynb` | `variable_analysis/` |
| §5.2 Representações + platô | Tab. 1–2 | `run2b.py` · `exp4_soccermap_cnn.py` · `src/cpp/bridge.py` | `classify_analysis/` · `cnn_voronoi_analysis/` |
| §5.3 Dose-resposta (alvo × geometria) | `fig_ladder` | `target_ladder_ablation.py` · `label_noise.py` · `make_ladder_fig.py` | `star_analysis/target_ladder.csv` · `label_noise.csv` |
| §5.4 Ablação | `fig_ablation` | `exp6_ablation.py` · `make_figs.py` | `express_analysis/exp6_ablation.csv` |
| §5.5 Controle de *base rate* + *bootstrap* | — | `control_baserate.py` | `star_analysis/control_baserate.csv` |
| §5.6 Generalização entre competições | — | `exp3_crosscomp.py` | `express_analysis/exp3.csv` |
| §6.1 Logística *vs.* GBT | Tabela 3 | `eval_logit_vs_gbt.py` | `star_analysis/logit_vs_gbt.csv` |
| §6.2 Calibração + curva de ganho | `fig_calibration` · `fig_lift` | `eval_usability.py` · `make_figs.py` | `star_analysis/usability.csv` |
| §6.3 Confiabilidade do *rating* | Tabela 4 | `B_survival.py` (+ `star_data_all.py`) | `star_analysis/B_*.csv` |
| §6.4 Gatilhos (*triggers*) | Tabela 5 | `triggers_xR.py` | `star_analysis/triggers_*.csv` |
| §6.5 Contrapressão | — | `counterpress_target.py` | `star_analysis/counterpress_*.csv` |
| §7 Aplicação prática | `fig_usecase` · `fig_usecase2` · `fig_flow` | `usecases_xR.py` · `make_figs.py` | `star_analysis/usecases.md` · `paper/figs/` |

> As figuras do artigo são geradas por `src/xR_experiments/make_figs.py` e `make_ladder_fig.py`,
> que as gravam em **`paper/figs/`**, junto do `artigo-xR.tex`. As dos slides: `make_slides_figs.py`
> e `make_slides_extra.py` (→ `slides/figs/`).

## Resultados (tabelas geradas)

As pastas `*_analysis/` contêm os CSVs que alimentam as tabelas e figuras do artigo (gerados; não
versionados): `variable_analysis/` (testes de hipótese), `classify_analysis/` (platô),
`cnn_voronoi_analysis/` (Voronoi), `express_analysis/` (alvo/ablação/cross-comp) e `star_analysis/`
(calibração, usabilidade, confiabilidade).

## Como gerar o artigo (PDF)

O artigo está em **`paper/`** e é **autocontido** — só precisa de uma distribuição LaTeX instalada
(**MiKTeX** ou **TeX Live**; *não* incluída no repositório). As figuras já vêm em `paper/figs/`;
para regenerá-las dos dados, rode antes `./venv1/Scripts/python.exe src/xR_experiments/make_figs.py`.

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
./venv1/Scripts/python.exe src/xR_experiments/eval_logit_vs_gbt.py   # Tabela 3 (logística vs GBT)
./venv1/Scripts/python.exe src/xR_experiments/eval_usability.py      # calibração + lift
./venv1/Scripts/python.exe src/xR_experiments/make_figs.py           # figuras do artigo

# C++ (após compilar src/cpp via build.sh)
./venv1/Scripts/python.exe src/cpp/bridge.py prepare --select voronoi
./venv1/Scripts/python.exe src/cpp/bridge.py run --model gnn
```
