#!/bin/sh
# Compila train.cpp (GNN manual, Eigen) + cnn_mlpack.cpp (CNN via mlpack).
# Pre-requisitos no MSYS2 UCRT64:
#   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-eigen3 \
#             mingw-w64-ucrt-x86_64-mlpack mingw-w64-ucrt-x86_64-armadillo
EIGEN="${EIGEN_INC:-/ucrt64/include/eigen3}"
echo "Eigen: $EIGEN | mlpack/armadillo: /ucrt64/include"
g++ -O3 -march=native -std=c++17 \
    -I"$EIGEN" \
    train.cpp cnn_mlpack.cpp \
    -o train.exe \
    -larmadillo -lopenblas \
    && echo "OK -> train.exe"
