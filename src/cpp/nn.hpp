// nn.hpp — utilidades de rede neural (Eigen): IO binário, Adam, init, ativações, im2col.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <stdexcept>

using Eigen::MatrixXf;
using Eigen::VectorXf;

// ---------------- IO binário (mesmo formato de bridge.py) ----------------
// layout do arquivo: int32 ndim | int32 dims[ndim] | float32 data[prod(dims)]  (row-major)
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    long long numel() const { long long n = 1; for (int d : shape) n *= d; return n; }
};

inline Tensor read_tensor(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("nao consegui abrir " + path);
    int ndim = 0; f.read((char*)&ndim, 4);
    Tensor t; t.shape.resize(ndim);
    f.read((char*)t.shape.data(), 4 * ndim);
    long long n = t.numel(); t.data.resize(n);
    f.read((char*)t.data.data(), sizeof(float) * n);
    return t;
}

inline void write_tensor(const std::string& path, const std::vector<float>& data,
                         const std::vector<int>& shape) {
    std::ofstream f(path, std::ios::binary);
    int ndim = (int)shape.size();
    f.write((char*)&ndim, 4);
    f.write((char*)shape.data(), 4 * (size_t)ndim);
    f.write((char*)data.data(), sizeof(float) * data.size());
}

// ---------------- inicialização ----------------
inline std::mt19937& rng() { static std::mt19937 g(42); return g; }

inline MatrixXf he_init(int rows, int cols) {  // rows = fan_in
    std::normal_distribution<float> nd(0.f, std::sqrt(2.f / std::max(1, rows)));
    MatrixXf W(rows, cols);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) W(i, j) = nd(rng());
    return W;
}

// ---------------- Adam (um por parâmetro) ----------------
struct Adam {
    MatrixXf m, v; int t = 0;
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 1e-4f;
    void init(int r, int c) { m = MatrixXf::Zero(r, c); v = MatrixXf::Zero(r, c); t = 0; }
    void step(MatrixXf& W, const MatrixXf& grad) {
        MatrixXf g = grad + wd * W;
        ++t;
        m = b1 * m + (1.f - b1) * g;
        v = (b2 * v.array() + (1.f - b2) * g.array().square()).matrix();
        float c1 = 1.f - std::pow(b1, t), c2 = 1.f - std::pow(b2, t);
        W.array() -= lr * (m.array() / c1) / (((v.array() / c2).sqrt()) + eps);
    }
};

// ---------------- ativações ----------------
inline MatrixXf relu(const MatrixXf& x) { return x.cwiseMax(0.f); }
inline MatrixXf relu_grad(const MatrixXf& x) { return (x.array() > 0.f).cast<float>(); }
inline float sigmoidf(float z) { return 1.f / (1.f + std::exp(-z)); }

// BCE com logits: retorna gradiente dL/dlogit = sigmoid(logit) - y (médio no batch é feito fora).
inline VectorXf bce_grad(const VectorXf& logits, const VectorXf& y) {
    VectorXf g(logits.size());
    for (int i = 0; i < logits.size(); ++i) g[i] = sigmoidf(logits[i]) - y[i];
    return g;
}
inline float bce_loss(const VectorXf& logits, const VectorXf& y) {
    float s = 0.f;
    for (int i = 0; i < logits.size(); ++i) {
        float z = logits[i];
        // log(1+exp(-|z|)) + max(z,0) - z*y  (estável)
        s += std::log1p(std::exp(-std::abs(z))) + std::max(z, 0.f) - z * y[i];
    }
    return s / logits.size();
}

// ---------------- im2col / col2im (conv 3x3, stride 1, pad 1; mantém H,W) ----------------
// Entrada de uma imagem: ptr para [C,H,W] row-major. Saída cols: [H*W, C*9].
inline MatrixXf im2col(const float* img, int C, int H, int W) {
    MatrixXf cols(H * W, C * 9);
    cols.setZero();
    int col = 0;
    for (int c = 0; c < C; ++c)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                for (int y = 0; y < H; ++y) {
                    int yy = y + dy;
                    for (int x = 0; x < W; ++x) {
                        int xx = x + dx;
                        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                            cols(y * W + x, col) = img[(c * H + yy) * W + xx];
                    }
                }
                ++col;
            }
    return cols;
}

// Acumula gradiente de cols [H*W, C*9] de volta para a imagem [C,H,W] (out += ...).
inline void col2im(const MatrixXf& dcols, int C, int H, int W, float* out) {
    int col = 0;
    for (int c = 0; c < C; ++c)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                for (int y = 0; y < H; ++y) {
                    int yy = y + dy;
                    for (int x = 0; x < W; ++x) {
                        int xx = x + dx;
                        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                            out[(c * H + yy) * W + xx] += dcols(y * W + x, col);
                    }
                }
                ++col;
            }
}
