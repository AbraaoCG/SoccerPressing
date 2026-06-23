// cnn_mlpack.cpp — treina a CNN com mlpack 4.x (substitui a CNN manual de train.cpp).
// API: train_cnn_mlpack(io, epochs) — le io/cnn_*.bin, ytr.bin; grava io/pred_cnn.bin.
//
// Arquitetura: Conv(8,3x3,pad1) -> ReLU -> Conv(16,3x3,pad1) -> ReLU -> MeanPooling(48x48)
//              -> Linear(2) -> LogSoftMax (treino com NegativeLogLikelihood).
// Adam (ensmallen), input 48x48x1, labels {0,1}.

#include "nn.hpp"  // Tensor, read_tensor, write_tensor
#include <mlpack.hpp>
#include <ensmallen.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace mlpack;

void train_cnn_mlpack(const std::string& io, int epochs) {
    Tensor Xtr = read_tensor(io + "/cnn_Xtr.bin");
    Tensor Xte = read_tensor(io + "/cnn_Xte.bin");
    Tensor ytr = read_tensor(io + "/ytr.bin");
    int N = Xtr.shape[0], H = Xtr.shape[2], W = Xtr.shape[3];
    int Nt = Xte.shape[0];
    int FEAT = H * W;

    // converte para arma::mat (rows = W*H*C, cols = amostras)
    arma::mat X(FEAT, N), Xt(FEAT, Nt);
    for (int i = 0; i < N; ++i)
        for (int p = 0; p < FEAT; ++p) X(p, i) = (double)Xtr.data[(long long)i * FEAT + p];
    for (int i = 0; i < Nt; ++i)
        for (int p = 0; p < FEAT; ++p) Xt(p, i) = (double)Xte.data[(long long)i * FEAT + p];
    arma::mat y(1, N);
    for (int i = 0; i < N; ++i) y(0, i) = ytr.data[i] > 0.5f ? 1.0 : 0.0;

    // FFN com NLL + LogSoftMax (padrao mlpack para classificacao)
    FFN<NegativeLogLikelihood, HeInitialization> model;
    model.Add<Convolution>(8, 3, 3, 1, 1, 1, 1);     // maps=8, k=3x3, stride=1x1, pad=1x1
    model.Add<ReLU>();
    model.Add<Convolution>(16, 3, 3, 1, 1, 1, 1);
    model.Add<ReLU>();
    model.Add<MeanPooling>((size_t)W, (size_t)H, 1, 1);  // GAP: kernel cobre tudo
    model.Add<Linear>(2);
    model.Add<LogSoftMax>();
    model.InputDimensions() = std::vector<size_t>{(size_t)W, (size_t)H, 1};

    // Adam: (lr, batch, b1, b2, eps, maxIter, tol, shuffle)
    ens::Adam opt(1e-3, 128, 0.9, 0.999, 1e-8, (size_t)epochs * (size_t)N, 1e-8, true);
    std::cout << "  cnn(mlpack): treino - " << epochs << " epocas em " << N << " amostras\n";
    model.Train(X, y, opt, ens::PrintLoss(), ens::ProgressBar());

    // Predicoes: preds = log-probabilidades (2 x Nt); P(classe=1) = exp(linha 1)
    arma::mat preds;
    model.Predict(Xt, preds);
    std::vector<float> out(Nt);
    for (int i = 0; i < Nt; ++i) out[i] = (float)std::exp(preds(1, i));
    write_tensor(io + "/pred_cnn.bin", out, {Nt});
    std::cout << "cnn(mlpack): " << Nt << " predicoes salvas\n";
}
