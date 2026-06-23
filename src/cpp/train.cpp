// train.cpp — treina GNN ou CNN (C++/Eigen) para prever recovered.
//   ./train.exe --model cnn|gnn --io <dir> [--epochs N]
// Lê io/{model}_*.bin (de bridge.py), treina, grava io/pred_{model}.bin (sigmoide no teste).
//
// Convencoes: camadas ACUMULAM gradiente por amostra e dao 1 passo Adam por mini-batch
// (zero() -> varias backward() -> step()), uniforme para Linear/Conv/GNN.
#include "nn.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

using Eigen::MatrixXf;
using Eigen::VectorXf;
typedef long long ll;

// implementado em cnn_mlpack.cpp
void train_cnn_mlpack(const std::string& io, int epochs);

// ----------------------------- camada densa -----------------------------
struct Linear {
    MatrixXf W, b, dW, db, Xc; Adam aW, ab;
    void init(int in, int out, float lr) {
        W = he_init(in, out); b = MatrixXf::Zero(1, out);
        aW.init(in, out); aW.lr = lr; ab.init(1, out); ab.lr = lr; zero();
    }
    void zero() { dW = MatrixXf::Zero(W.rows(), W.cols()); db = MatrixXf::Zero(1, W.cols()); }
    MatrixXf forward(const MatrixXf& X) { Xc = X; MatrixXf Y = X * W; Y.rowwise() += b.row(0); return Y; }
    MatrixXf backward(const MatrixXf& dY) { dW += Xc.transpose() * dY; db += dY.colwise().sum(); return dY * W.transpose(); }
    void step() { aW.step(W, dW); ab.step(b, db); zero(); }
};

// ----------------------------- conv 3x3 (im2col) -----------------------------
struct Conv {
    int Cin, Cout; MatrixXf W, b, dW, db; Adam aW, ab;
    void init(int cin, int cout, float lr) {
        Cin = cin; Cout = cout; W = he_init(cin * 9, cout); b = MatrixXf::Zero(1, cout);
        aW.init(cin * 9, cout); aW.lr = lr; ab.init(1, cout); ab.lr = lr; zero();
    }
    void zero() { dW = MatrixXf::Zero(Cin * 9, Cout); db = MatrixXf::Zero(1, Cout); }
    void step() { aW.step(W, dW); ab.step(b, db); zero(); }
};

static std::vector<float> chw_from_act(const MatrixXf& act, int C, int HW) {
    std::vector<float> buf((size_t)C * HW);
    for (int c = 0; c < C; ++c) for (int p = 0; p < HW; ++p) buf[(size_t)c * HW + p] = act(p, c);
    return buf;
}

// ----------------------------- CNN -----------------------------
// conv(1->C1) relu  conv(C1->C2) relu  GAP  Linear(C2->1)
struct CNNCache { MatrixXf cols1, pre1, cols2, pre2; };
struct CNN {
    int H, W, C1 = 8, C2 = 16; Conv c1, c2; Linear fc;
    void init(int h, int w, float lr) { H = h; W = w; c1.init(1, C1, lr); c2.init(C1, C2, lr); fc.init(C2, 1, lr); }
    MatrixXf features(const float* data, const std::vector<int>& idx, std::vector<CNNCache>* cache) {
        int B = (int)idx.size(), HW = H * W; MatrixXf feat(B, C2);
        if (cache) cache->resize(B);
        for (int bi = 0; bi < B; ++bi) {
            const float* img = data + (ll)idx[bi] * HW;
            MatrixXf cols1 = im2col(img, 1, H, W);
            MatrixXf pre1 = cols1 * c1.W; pre1.rowwise() += c1.b.row(0);
            MatrixXf act1 = relu(pre1);
            std::vector<float> buf1 = chw_from_act(act1, C1, HW);
            MatrixXf cols2 = im2col(buf1.data(), C1, H, W);
            MatrixXf pre2 = cols2 * c2.W; pre2.rowwise() += c2.b.row(0);
            MatrixXf act2 = relu(pre2);
            feat.row(bi) = act2.colwise().mean();           // global average pooling
            if (cache) (*cache)[bi] = {cols1, pre1, cols2, pre2};
        }
        return feat;
    }
    VectorXf logits(const float* data, const std::vector<int>& idx) {
        return fc.forward(features(data, idx, nullptr)).col(0);
    }
    void train_batch(const float* data, const std::vector<int>& idx, const VectorXf& y) {
        int B = (int)idx.size(), HW = H * W;
        std::vector<CNNCache> cache;
        MatrixXf feat = features(data, idx, &cache);
        MatrixXf z = fc.forward(feat);                       // [B,1]
        VectorXf g = bce_grad(z.col(0), y) / (float)B;       // dL/dlogit
        MatrixXf dfeat = fc.backward(g);                     // [B,C2]
        for (int bi = 0; bi < B; ++bi) {
            CNNCache& cc = cache[bi];
            MatrixXf dact2(HW, C2);
            for (int c = 0; c < C2; ++c) dact2.col(c).setConstant(dfeat(bi, c) / (float)HW);
            MatrixXf dpre2 = dact2.cwiseProduct(relu_grad(cc.pre2));
            c2.dW += cc.cols2.transpose() * dpre2; c2.db += dpre2.colwise().sum();
            MatrixXf dcols2 = dpre2 * c2.W.transpose();      // [HW, C1*9]
            std::vector<float> dbuf1((size_t)C1 * HW, 0.f);
            col2im(dcols2, C1, H, W, dbuf1.data());
            MatrixXf dact1(HW, C1);
            for (int c = 0; c < C1; ++c) for (int p = 0; p < HW; ++p) dact1(p, c) = dbuf1[(size_t)c * HW + p];
            MatrixXf dpre1 = dact1.cwiseProduct(relu_grad(cc.pre1));
            c1.dW += cc.cols1.transpose() * dpre1; c1.db += dpre1.colwise().sum();
        }
        fc.step(); c1.step(); c2.step();
    }
};

// ----------------------------- GNN (DenseGCN) -----------------------------
// gconv(F->h) relu  gconv(h->h) relu  pool[mean;max]  Linear(2h->h) relu Linear(h->1)
struct GCache { MatrixXf X, A; VectorXf mask; MatrixXf H1pre, H1, H2pre, H2, l1pre, a1; std::vector<int> argmax; float nv; int K; };
struct GNN {
    int F = 3, h = 16; MatrixXf Wg1, Wg2, dWg1, dWg2; Adam aWg1, aWg2; Linear l1, l2;
    void init(int Fin, float lr) {
        F = Fin; Wg1 = he_init(F, h); Wg2 = he_init(h, h);
        aWg1.init(F, h); aWg1.lr = lr; aWg2.init(h, h); aWg2.lr = lr;
        l1.init(2 * h, h, lr); l2.init(h, 1, lr); zerog();
    }
    void zerog() { dWg1 = MatrixXf::Zero(F, h); dWg2 = MatrixXf::Zero(h, h); }
    static MatrixXf mask_rows(const MatrixXf& M, const VectorXf& mask) { return (M.array().colwise() * mask.array()).matrix(); }
    float forward(const MatrixXf& X, const MatrixXf& A, const VectorXf& mask, GCache& c) {
        c.X = X; c.A = A; c.mask = mask; c.K = (int)X.rows();
        c.H1pre = A * (X * Wg1); c.H1 = mask_rows(relu(c.H1pre), mask);
        c.H2pre = A * (c.H1 * Wg2); c.H2 = mask_rows(relu(c.H2pre), mask);
        c.nv = std::max(1.f, mask.sum());
        VectorXf mean = c.H2.colwise().sum().transpose() / c.nv;     // H2 ja zerado fora da mascara
        VectorXf mx(h); c.argmax.assign(h, 0);
        for (int j = 0; j < h; ++j) {
            float best = -1e30f; int bi = 0;
            for (int i = 0; i < c.K; ++i) if (mask[i] > 0 && c.H2(i, j) > best) { best = c.H2(i, j); bi = i; }
            mx[j] = (best < -1e29f ? 0.f : best); c.argmax[j] = bi;
        }
        VectorXf pooled(2 * h); pooled << mean, mx;
        c.l1pre = l1.forward(pooled.transpose());                    // [1,h]; l1.Xc = pooled^T
        c.a1 = relu(c.l1pre);
        MatrixXf z = l2.forward(c.a1);                               // l2.Xc = a1
        return z(0, 0);
    }
    void backward(GCache& c, float g) {
        MatrixXf da1 = l2.backward(MatrixXf::Constant(1, 1, g));      // [1,h]
        MatrixXf da1pre = da1.cwiseProduct(relu_grad(c.l1pre));
        MatrixXf dpooled = l1.backward(da1pre);                       // [1,2h]
        VectorXf dmean = dpooled.block(0, 0, 1, h).transpose();
        VectorXf dmax = dpooled.block(0, h, 1, h).transpose();
        MatrixXf dH2 = MatrixXf::Zero(c.K, h);
        for (int i = 0; i < c.K; ++i) if (c.mask[i] > 0) dH2.row(i) += (dmean / c.nv).transpose();
        for (int j = 0; j < h; ++j) dH2(c.argmax[j], j) += dmax[j];
        MatrixXf dH2pre = mask_rows(dH2.cwiseProduct(relu_grad(c.H2pre)), c.mask);
        MatrixXf dP1 = c.A.transpose() * dH2pre;                     // [K,h]
        dWg2 += c.H1.transpose() * dP1;
        MatrixXf dH1 = dP1 * Wg2.transpose();
        MatrixXf dH1pre = mask_rows(dH1.cwiseProduct(relu_grad(c.H1pre)), c.mask);
        MatrixXf dP0 = c.A.transpose() * dH1pre;
        dWg1 += c.X.transpose() * dP0;
    }
    void step() { aWg1.step(Wg1, dWg1); aWg2.step(Wg2, dWg2); zerog(); l1.step(); l2.step(); }
};

// ----------------------------- helpers de dados -----------------------------
static std::vector<int> shuffled(int n) {
    std::vector<int> v(n); std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), rng()); return v;
}

void train_cnn(const std::string& io, int epochs) {
    Tensor Xtr = read_tensor(io + "/cnn_Xtr.bin"), Xte = read_tensor(io + "/cnn_Xte.bin");
    Tensor ytr = read_tensor(io + "/ytr.bin");
    int N = Xtr.shape[0], H = Xtr.shape[2], W = Xtr.shape[3];
    VectorXf y = Eigen::Map<VectorXf>(ytr.data.data(), N);
    CNN net; net.init(H, W, 1e-3f); int bs = 128;
    std::vector<int> all(N); std::iota(all.begin(), all.end(), 0);
    for (int ep = 0; ep < epochs; ++ep) {
        auto order = shuffled(N);
        for (int s = 0; s < N; s += bs) {
            std::vector<int> idx(order.begin() + s, order.begin() + std::min(N, s + bs));
            VectorXf yb(idx.size()); for (size_t i = 0; i < idx.size(); ++i) yb[i] = y[idx[i]];
            net.train_batch(Xtr.data.data(), idx, yb);
        }
        std::cout << "  cnn epoca " << ep + 1 << "/" << epochs
                  << " loss " << bce_loss(net.logits(Xtr.data.data(), all), y) << "\n";
    }
    int M = Xte.shape[0]; std::vector<int> ai(M); std::iota(ai.begin(), ai.end(), 0);
    VectorXf zt = net.logits(Xte.data.data(), ai);
    std::vector<float> pred(M); for (int i = 0; i < M; ++i) pred[i] = sigmoidf(zt[i]);
    write_tensor(io + "/pred_cnn.bin", pred, {M});
    std::cout << "cnn: " << M << " predicoes salvas\n";
}

void train_gnn(const std::string& io, int epochs) {
    Tensor Xt = read_tensor(io + "/gnn_Xtr.bin"), At = read_tensor(io + "/gnn_Atr.bin"),
           Mt = read_tensor(io + "/gnn_Mtr.bin"), yt = read_tensor(io + "/ytr.bin");
    Tensor Xe = read_tensor(io + "/gnn_Xte.bin"), Ae = read_tensor(io + "/gnn_Ate.bin"),
           Me = read_tensor(io + "/gnn_Mte.bin");
    int N = Xt.shape[0], K = Xt.shape[1], F = Xt.shape[2];
    auto gX = [&](const Tensor& T, int i) { MatrixXf X(K, F); for (int r = 0; r < K; ++r) for (int c = 0; c < F; ++c) X(r, c) = T.data[((ll)i * K + r) * F + c]; return X; };
    auto gA = [&](const Tensor& T, int i) { MatrixXf A(K, K); for (int r = 0; r < K; ++r) for (int c = 0; c < K; ++c) A(r, c) = T.data[((ll)i * K + r) * K + c]; return A; };
    auto gM = [&](const Tensor& T, int i) { VectorXf m(K); for (int r = 0; r < K; ++r) m[r] = T.data[(ll)i * K + r]; return m; };
    VectorXf y = Eigen::Map<VectorXf>(yt.data.data(), N);
    GNN net; net.init(F, 1e-3f); int bs = 128;
    for (int ep = 0; ep < epochs; ++ep) {
        auto order = shuffled(N); double loss = 0;
        for (int s = 0; s < N; s += bs) {
            int e = std::min(N, s + bs);
            for (int t = s; t < e; ++t) {
                int i = order[t]; GCache c;
                float z = net.forward(gX(Xt, i), gA(At, i), gM(Mt, i), c);
                float p = sigmoidf(z);
                loss += -(y[i] * std::log(p + 1e-9f) + (1 - y[i]) * std::log(1 - p + 1e-9f));
                net.backward(c, (p - y[i]) / (e - s));
            }
            net.step();
        }
        std::cout << "  gnn epoca " << ep + 1 << "/" << epochs << " loss " << loss / N << "\n";
    }
    int M = Xe.shape[0]; std::vector<float> pred(M);
    for (int i = 0; i < M; ++i) { GCache c; pred[i] = sigmoidf(net.forward(gX(Xe, i), gA(Ae, i), gM(Me, i), c)); }
    write_tensor(io + "/pred_gnn.bin", pred, {M});
    std::cout << "gnn: " << M << " predicoes salvas\n";
}

int main(int argc, char** argv) {
    std::string model = "cnn", io = "io"; int epochs = 15;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--io" && i + 1 < argc) io = argv[++i];
        else if (a == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
    }
    std::cout << "modelo=" << model << " io=" << io << " epocas=" << epochs << "\n";
    try {
        if (model == "cnn") train_cnn_mlpack(io, epochs);
        else train_gnn(io, epochs);
    } catch (const std::exception& ex) { std::cerr << "ERRO: " << ex.what() << "\n"; return 1; }
    return 0;
}
