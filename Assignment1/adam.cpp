#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

extern "C" {
using tensor_handle = void*;

const char* tensor_last_error();
std::int64_t tensor_size(tensor_handle h);
int tensor_read_data(tensor_handle h, double* out, std::int64_t out_len);
int tensor_read_grad(tensor_handle h, double* out, std::int64_t out_len);
int tensor_has_grad(tensor_handle h);
int tensor_write_data(tensor_handle h, const double* in, std::int64_t in_len);

// Adam update (L2 weight decay via gradient term):
//  m = beta1*m + (1-beta1)*g
//  v = beta2*v + (1-beta2)*g^2
//  mhat = m / (1-beta1^t)
//  vhat = v / (1-beta2^t)
//  w = w - lr * mhat / (sqrt(vhat) + eps)
// If weight_decay > 0, uses L2 term via g += weight_decay*w.
int adam_update(
    tensor_handle param,
    tensor_handle m,
    tensor_handle v,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    std::int64_t t
);
}

static void require(bool cond, const char* msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

int adam_update(
    tensor_handle param,
    tensor_handle m,
    tensor_handle v,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    std::int64_t t
) {
    try {
        require(param != nullptr, "param must be non-null");
        require(m != nullptr, "m must be non-null");
        require(v != nullptr, "v must be non-null");
        require(lr > 0.0, "lr must be > 0");
        require(beta1 >= 0.0 && beta1 < 1.0, "beta1 must be in [0,1)");
        require(beta2 >= 0.0 && beta2 < 1.0, "beta2 must be in [0,1)");
        require(eps > 0.0, "eps must be > 0");
        require(weight_decay >= 0.0, "weight_decay must be >= 0");
        require(t >= 1, "t must be >= 1");

        const std::int64_t n = tensor_size(param);
        require(n == tensor_size(m), "param and m must have same numel");
        require(n == tensor_size(v), "param and v must have same numel");
        require(n >= 0, "invalid numel");
        if (n == 0) {
            return 0;
        }

        require(tensor_has_grad(param) == 1, "param must have grad (call backward first)");

        std::vector<double> w(static_cast<std::size_t>(n));
        std::vector<double> mvec(static_cast<std::size_t>(n));
        std::vector<double> vvec(static_cast<std::size_t>(n));
        std::vector<double> g(static_cast<std::size_t>(n));

        require(tensor_read_data(param, w.data(), n) == 0, tensor_last_error());
        require(tensor_read_data(m, mvec.data(), n) == 0, tensor_last_error());
        require(tensor_read_data(v, vvec.data(), n) == 0, tensor_last_error());
        require(tensor_read_grad(param, g.data(), n) == 0, tensor_last_error());

        const double one_minus_b1 = 1.0 - beta1;
        const double one_minus_b2 = 1.0 - beta2;
        const double b1t = std::pow(beta1, static_cast<double>(t));
        const double b2t = std::pow(beta2, static_cast<double>(t));
        const double inv_bias1 = 1.0 / (1.0 - b1t);
        const double inv_bias2 = 1.0 / (1.0 - b2t);

        for (std::int64_t i = 0; i < n; ++i) {
            double gi = g[static_cast<std::size_t>(i)];
            if (weight_decay > 0.0) {
                gi += weight_decay * w[static_cast<std::size_t>(i)];
            }

            double mi = mvec[static_cast<std::size_t>(i)];
            double vi = vvec[static_cast<std::size_t>(i)];

            mi = beta1 * mi + one_minus_b1 * gi;
            vi = beta2 * vi + one_minus_b2 * (gi * gi);

            mvec[static_cast<std::size_t>(i)] = mi;
            vvec[static_cast<std::size_t>(i)] = vi;

            const double mhat = mi * inv_bias1;
            const double vhat = vi * inv_bias2;
            w[static_cast<std::size_t>(i)] -= lr * mhat / (std::sqrt(vhat) + eps);
        }

        require(tensor_write_data(param, w.data(), n) == 0, tensor_last_error());
        require(tensor_write_data(m, mvec.data(), n) == 0, tensor_last_error());
        require(tensor_write_data(v, vvec.data(), n) == 0, tensor_last_error());
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}
