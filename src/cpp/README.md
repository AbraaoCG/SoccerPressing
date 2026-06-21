# algorithms/ — GNN e CNN em C++ (treino de alta performance)

Implementação em **C++** dos modelos de geometria do pressing (GNN e CNN) que preveem
`recovered` (recuperação da bola em ≤10 s). O Python continua responsável por **ler/preparar os
dados** (StatsBomb Euro 2020); o C++ faz o **treino/inferência**.

## Arquitetura da integração (linha de comando + arquivos binários)

Escolhida por robustez no toolchain Windows/MinGW (sem pybind11/ABI):

```
bridge.py  --prepare-->  io/*.bin   --(exe C++ lê)-->  treina  -->  io/pred_*.bin  --(Python lê)-->  AUC
```

1. `bridge.py prepare` carrega a Euro 2020, monta para cada pressão:
   - **CNN:** imagem raster 1×48×48 (adversários reorientados em torno do portador);
   - **GNN:** features de nó `[K,3]` + adjacência `[K,K]` + máscara `[K]` (K=14 adversários);
   - rótulo `recovered`; faz split treino/teste estratificado; grava em `io/` no formato binário.
2. `bridge.py build --model gnn|cnn` compila o C++ com g++.
3. `bridge.py run --model gnn|cnn` executa o `.exe`, lê as predições e reporta **AUC** e acurácia.

### Formato binário (`io/*.bin`)
`int32 ndim` · `int32 dim[ndim]` · `float32 data[prod(dims)]` (row-major). Lido/escrito por
`read_tensor`/`write_tensor` em `nn.hpp` e por `bridge.py`.

## Build (MSYS2 / MinGW UCRT64)

Dependências:
- **Eigen** (header-only) — para a GNN.
- **mlpack 4.x** + **Armadillo** + **OpenBLAS** + **ensmallen** — para a CNN.

```
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-eigen3 \
          mingw-w64-ucrt-x86_64-mlpack \
          mingw-w64-ucrt-x86_64-armadillo
```

Compilar (a partir desta pasta):
```
bash build.sh
```
ou:
```
g++ -O3 -march=native -std=c++17 -I /ucrt64/include/eigen3 \
    train.cpp cnn_mlpack.cpp -o train.exe -larmadillo -lopenblas
```
Depois: `python bridge.py prepare && python bridge.py run --model cnn|gnn`.

## Arquivos
- `bridge.py`       — preparação de dados, export binário, build e execução + métricas (Python/venv1).
- `nn.hpp`          — Eigen: IO binário, camadas (Linear/Conv via im2col), Adam, BCE — usado pela GNN.
- `train.cpp`       — `--model cnn|gnn`: GNN manual (Eigen, backward completo). Dispatcher.
- `cnn_mlpack.cpp`  — CNN via **mlpack 4.x** (`FFN` + `Convolution` + `MeanPooling` + `Linear` +
  `LogSoftMax` + `NegativeLogLikelihood`; otimizador `ens::Adam`).
- `build.sh`        — comando de compilação.
- `io/`             — dados de troca (gerado; ignorado no git).
