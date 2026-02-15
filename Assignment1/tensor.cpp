#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

static thread_local std::string g_last_error;

static void set_last_error(const std::string& msg) { g_last_error = msg; }
static void set_last_error(const char* msg) { g_last_error = (msg ? msg : ""); }

static std::int64_t prod(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1), std::multiplies<>());
}

static std::vector<std::int64_t> compute_strides(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    std::vector<std::int64_t> strides(shape.size(), 1);
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<std::size_t>(i)] =
            strides[static_cast<std::size_t>(i + 1)] * shape[static_cast<std::size_t>(i + 1)];
    }
    return strides;
}

struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

struct Tensor : std::enable_shared_from_this<Tensor> {
    std::vector<std::int64_t> shape;
    std::int64_t size = 1;
    std::vector<double> data;
    std::vector<std::int64_t> strides;
    bool requires_grad = false;
    std::vector<double> grad; // empty if !requires_grad

    std::vector<std::weak_ptr<Tensor>> prev;
    std::string op;
    std::function<void()> backward_fn;
    std::string name;

    Tensor(std::vector<double> d, std::vector<std::int64_t> s, bool req_grad)
        : shape(std::move(s)),
          size(prod(shape)),
          data(std::move(d)),
          strides(compute_strides(shape)),
          requires_grad(req_grad),
          grad(req_grad ? static_cast<std::size_t>(size) : 0, 0.0),
          backward_fn([]() {}) {
        if (static_cast<std::int64_t>(data.size()) != size) {
            throw std::runtime_error("Data size mismatch");
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    TensorPtr named(const std::string& n) {
        name = n;
        return shared_from_this();
    }

    void zero_grad() {
        if (!requires_grad) {
            return;
        }
        if (grad.size() != static_cast<std::size_t>(size)) {
            grad.assign(static_cast<std::size_t>(size), 0.0);
        } else {
            std::fill(grad.begin(), grad.end(), 0.0);
        }
    }

    void accum_grad(const std::vector<double>& g) {
        if (!requires_grad) {
            return;
        }
        if (static_cast<std::int64_t>(g.size()) != size) {
            throw std::runtime_error("Gradient size mismatch");
        }
        if (grad.size() != static_cast<std::size_t>(size)) {
            grad.assign(static_cast<std::size_t>(size), 0.0);
        }
        for (std::int64_t i = 0; i < size; ++i) {
            grad[static_cast<std::size_t>(i)] += g[static_cast<std::size_t>(i)];
        }
    }

    std::int64_t index(const std::vector<std::int64_t>& idx) const {
        if (idx.size() != shape.size()) {
            throw std::runtime_error("Incorrect number of indices");
        }
        std::int64_t flat = 0;
        for (std::size_t k = 0; k < idx.size(); ++k) {
            const auto i = idx[k];
            const auto s = shape[k];
            const auto st = strides[k];
            if (i < 0 || i >= s) {
                throw std::runtime_error("Index out of bounds");
            }
            flat += i * st;
        }
        return flat;
    }

    double get(const std::vector<std::int64_t>& idx) const { return data[static_cast<std::size_t>(index(idx))]; }
    void set(const std::vector<std::int64_t>& idx, double v) { data[static_cast<std::size_t>(index(idx))] = v; }

    double item() const {
        if (size != 1) {
            throw std::runtime_error("item() only valid for size==1");
        }
        return data[0];
    }

    TensorPtr detach() const {
        return std::make_shared<Tensor>(data, shape, false);
    }

    void backward(const TensorPtr& grad_tensor = nullptr, const double* grad_scalar = nullptr) {
        if (!requires_grad) {
            throw std::runtime_error("backward() called on tensor that does not require gradients");
        }

        std::vector<double> grad_data;
        if (!grad_tensor && !grad_scalar) {
            if (size != 1) {
                throw std::runtime_error("grad must be specified for non-scalar outputs");
            }
            grad_data = {1.0};
        } else if (grad_tensor) {
            if (grad_tensor->size != size) {
                throw std::runtime_error("grad tensor must match output size");
            }
            grad_data = grad_tensor->data;
        } else {
            if (size != 1) {
                throw std::runtime_error("numeric grad only supported for scalar outputs");
            }
            grad_data = {static_cast<double>(*grad_scalar)};
        }

        std::vector<TensorPtr> topo;
        topo.reserve(1024);
        std::unordered_set<const Tensor*> visited;
        visited.reserve(2048);

        std::function<void(const TensorPtr&)> build = [&](const TensorPtr& v) {
            if (!v) {
                return;
            }
            if (visited.insert(v.get()).second) {
                for (const auto& wp : v->prev) {
                    if (auto sp = wp.lock()) {
                        build(sp);
                    }
                }
                topo.push_back(v);
            }
        };

        build(shared_from_this());
        accum_grad(grad_data);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward_fn();
        }
    }
};

static TensorPtr make_tensor(std::vector<double> d, const std::vector<std::int64_t>& s, bool req_grad) {
    return std::make_shared<Tensor>(std::move(d), s, req_grad);
}

static TensorPtr zeros(const std::vector<std::int64_t>& shape, bool requires_grad) {
    const auto sz = prod(shape);
    return make_tensor(std::vector<double>(static_cast<std::size_t>(sz), 0.0), shape, requires_grad);
}

static TensorPtr ones(const std::vector<std::int64_t>& shape, bool requires_grad) {
    const auto sz = prod(shape);
    return make_tensor(std::vector<double>(static_cast<std::size_t>(sz), 1.0), shape, requires_grad);
}

static TensorPtr randn(
    const std::vector<std::int64_t>& shape,
    double mean,
    double stdev,
    bool requires_grad,
    const std::int64_t* seed_opt) {
    const auto sz = prod(shape);
    std::vector<double> d(static_cast<std::size_t>(sz));
    std::mt19937 rng;
    if (seed_opt) {
        rng.seed(static_cast<std::mt19937::result_type>(*seed_opt));
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    std::normal_distribution<double> dist(mean, stdev);
    for (auto& v : d) {
        v = dist(rng);
    }
    return make_tensor(std::move(d), shape, requires_grad);
}

static void require_same_shape(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch");
    }
}

static TensorPtr reshape(const TensorPtr& a, const std::vector<std::int64_t>& new_shape) {
    if (std::find(new_shape.begin(), new_shape.end(), static_cast<std::int64_t>(-1)) != new_shape.end()) {
        throw std::runtime_error("-1 reshape not supported");
    }
    if (prod(new_shape) != a->size) {
        throw std::runtime_error("New shape size mismatch");
    }
    const bool req = a->requires_grad;
    auto out = make_tensor(a->data, new_shape, req);
    out->prev = {a};
    out->op = "reshape";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            a->accum_grad(out_sp->grad);
        }
    };
    return out;
}

static TensorPtr transpose2d(const TensorPtr& a) {
    if (a->shape.size() != 2) {
        throw std::runtime_error("transpose expects 2D tensor");
    }
    const std::int64_t m = a->shape[0];
    const std::int64_t n = a->shape[1];
    auto out = zeros({n, m}, a->requires_grad);
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            out->data[static_cast<std::size_t>(j * out->strides[0] + i * out->strides[1])] =
                a->data[static_cast<std::size_t>(i * a->strides[0] + j * a->strides[1])];
        }
    }
    out->prev = {a};
    out->op = "transpose";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp, m, n]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!a->requires_grad) {
            return;
        }
        std::vector<double> grad_self(static_cast<std::size_t>(a->size), 0.0);
        for (std::int64_t i = 0; i < m; ++i) {
            for (std::int64_t j = 0; j < n; ++j) {
                const auto out_idx = j * out_sp->strides[0] + i * out_sp->strides[1];
                const auto self_idx = i * a->strides[0] + j * a->strides[1];
                grad_self[static_cast<std::size_t>(self_idx)] += out_sp->grad[static_cast<std::size_t>(out_idx)];
            }
        }
        a->accum_grad(grad_self);
    };
    return out;
}

static TensorPtr sum_all(const TensorPtr& a) {
    double s = 0.0;
    for (double v : a->data) {
        s += v;
    }
    auto out = make_tensor(std::vector<double>{s}, {}, a->requires_grad);
    out->prev = {a};
    out->op = "sum";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            std::vector<double> g(static_cast<std::size_t>(a->size), out_sp->grad[0]);
            a->accum_grad(g);
        }
    };
    return out;
}

static TensorPtr neg(const TensorPtr& a) {
    std::vector<double> d = a->data;
    for (auto& v : d) {
        v = -v;
    }
    auto out = make_tensor(std::move(d), a->shape, a->requires_grad);
    out->prev = {a};
    out->op = "neg";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!a->requires_grad) {
            return;
        }
        std::vector<double> g(out_sp->grad.size());
        for (std::size_t i = 0; i < g.size(); ++i) {
            g[i] = -out_sp->grad[i];
        }
        a->accum_grad(g);
    };
    return out;
}

static TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape == b->shape) {
        std::vector<double> d(static_cast<std::size_t>(a->size));
        for (std::int64_t i = 0; i < a->size; ++i) {
            d[static_cast<std::size_t>(i)] =
                a->data[static_cast<std::size_t>(i)] + b->data[static_cast<std::size_t>(i)];
        }
        const bool req = a->requires_grad || b->requires_grad;
        auto out = make_tensor(std::move(d), a->shape, req);
        out->prev = {a, b};
        out->op = "+";
        std::weak_ptr<Tensor> out_wp = out;
        out->backward_fn = [a, b, out_wp]() {
            auto out_sp = out_wp.lock();
            if (!out_sp) {
                return;
            }
            if (a->requires_grad) {
                a->accum_grad(out_sp->grad);
            }
            if (b->requires_grad) {
                b->accum_grad(out_sp->grad);
            }
        };
        return out;
    }

    // Minimal broadcasting for dense bias: (N,C) + (1,C) or (1,C) + (N,C).
    const TensorPtr* x = &a;
    const TensorPtr* y = &b;
    bool swapped = false;
    if (a->shape.size() == 2 && b->shape.size() == 2 && a->shape[0] == 1 && b->shape[0] > 1 && a->shape[1] == b->shape[1]) {
        x = &b;
        y = &a;
        swapped = true;
    }

    if ((*x)->shape.size() == 2 && (*y)->shape.size() == 2 && (*y)->shape[0] == 1 && (*x)->shape[1] == (*y)->shape[1]) {
        const std::int64_t N = (*x)->shape[0];
        const std::int64_t C = (*x)->shape[1];
        std::vector<double> d(static_cast<std::size_t>(N * C));
        for (std::int64_t n = 0; n < N; ++n) {
            for (std::int64_t c = 0; c < C; ++c) {
                const std::size_t xi = static_cast<std::size_t>(n * (*x)->strides[0] + c * (*x)->strides[1]);
                const std::size_t yi = static_cast<std::size_t>(0 * (*y)->strides[0] + c * (*y)->strides[1]);
                d[static_cast<std::size_t>(n * C + c)] = (*x)->data[xi] + (*y)->data[yi];
            }
        }

        const bool req = a->requires_grad || b->requires_grad;
        auto out = make_tensor(std::move(d), (*x)->shape, req);
        out->prev = {a, b};
        out->op = "+bcast";
        std::weak_ptr<Tensor> out_wp = out;
        out->backward_fn = [a, b, out_wp, swapped, N, C]() {
            auto out_sp = out_wp.lock();
            if (!out_sp) {
                return;
            }
            // grad for the (N,C) input is direct
            if (a->requires_grad) {
                // If swapped, 'a' might be the (1,C) bias; handle below.
                if (!(swapped && a->shape.size() == 2 && a->shape[0] == 1)) {
                    a->accum_grad(out_sp->grad);
                }
            }
            if (b->requires_grad) {
                if (!( !swapped && b->shape.size() == 2 && b->shape[0] == 1)) {
                    b->accum_grad(out_sp->grad);
                }
            }

            // grad for the (1,C) broadcasted input is row-sum
            auto accum_bias = [&](const TensorPtr& bias) {
                if (!bias->requires_grad) {
                    return;
                }
                std::vector<double> gb(static_cast<std::size_t>(bias->size), 0.0);
                for (std::int64_t n = 0; n < N; ++n) {
                    for (std::int64_t c = 0; c < C; ++c) {
                        const std::size_t oi = static_cast<std::size_t>(n * out_sp->strides[0] + c * out_sp->strides[1]);
                        const std::size_t bi = static_cast<std::size_t>(0 * bias->strides[0] + c * bias->strides[1]);
                        gb[bi] += out_sp->grad[oi];
                    }
                }
                bias->accum_grad(gb);
            };

            if (a->shape.size() == 2 && a->shape[0] == 1) {
                accum_bias(a);
            }
            if (b->shape.size() == 2 && b->shape[0] == 1) {
                accum_bias(b);
            }
        };
        return out;
    }

    throw std::runtime_error("Shape mismatch");
}

static TensorPtr add_scalar(const TensorPtr& a, double c) {
    std::vector<double> d = a->data;
    for (auto& v : d) {
        v += c;
    }
    auto out = make_tensor(std::move(d), a->shape, a->requires_grad);
    out->prev = {a};
    out->op = "+c";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            a->accum_grad(out_sp->grad);
        }
    };
    return out;
}

static TensorPtr sub(const TensorPtr& a, const TensorPtr& b) {
    require_same_shape(a, b);
    std::vector<double> d(static_cast<std::size_t>(a->size));
    for (std::int64_t i = 0; i < a->size; ++i) {
        d[static_cast<std::size_t>(i)] = a->data[static_cast<std::size_t>(i)] - b->data[static_cast<std::size_t>(i)];
    }
    const bool req = a->requires_grad || b->requires_grad;
    auto out = make_tensor(std::move(d), a->shape, req);
    out->prev = {a, b};
    out->op = "-";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, b, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            a->accum_grad(out_sp->grad);
        }
        if (b->requires_grad) {
            std::vector<double> gb(out_sp->grad.size());
            for (std::size_t i = 0; i < gb.size(); ++i) {
                gb[i] = -out_sp->grad[i];
            }
            b->accum_grad(gb);
        }
    };
    return out;
}

static TensorPtr sub_scalar(const TensorPtr& a, double c) {
    std::vector<double> d = a->data;
    for (auto& v : d) {
        v -= c;
    }
    auto out = make_tensor(std::move(d), a->shape, a->requires_grad);
    out->prev = {a};
    out->op = "-c";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            a->accum_grad(out_sp->grad);
        }
    };
    return out;
}

static TensorPtr mul(const TensorPtr& a, const TensorPtr& b) {
    require_same_shape(a, b);
    std::vector<double> d(static_cast<std::size_t>(a->size));
    for (std::int64_t i = 0; i < a->size; ++i) {
        d[static_cast<std::size_t>(i)] = a->data[static_cast<std::size_t>(i)] * b->data[static_cast<std::size_t>(i)];
    }
    const bool req = a->requires_grad || b->requires_grad;
    auto out = make_tensor(std::move(d), a->shape, req);
    out->prev = {a, b};
    out->op = "*";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, b, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            std::vector<double> ga(static_cast<std::size_t>(a->size));
            for (std::int64_t i = 0; i < a->size; ++i) {
                ga[static_cast<std::size_t>(i)] = b->data[static_cast<std::size_t>(i)] * out_sp->grad[static_cast<std::size_t>(i)];
            }
            a->accum_grad(ga);
        }
        if (b->requires_grad) {
            std::vector<double> gb(static_cast<std::size_t>(b->size));
            for (std::int64_t i = 0; i < b->size; ++i) {
                gb[static_cast<std::size_t>(i)] = a->data[static_cast<std::size_t>(i)] * out_sp->grad[static_cast<std::size_t>(i)];
            }
            b->accum_grad(gb);
        }
    };
    return out;
}

static TensorPtr mul_scalar(const TensorPtr& a, double c) {
    std::vector<double> d = a->data;
    for (auto& v : d) {
        v *= c;
    }
    auto out = make_tensor(std::move(d), a->shape, a->requires_grad);
    out->prev = {a};
    out->op = "*c";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp, c]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            std::vector<double> ga(out_sp->grad.size());
            for (std::size_t i = 0; i < ga.size(); ++i) {
                ga[i] = c * out_sp->grad[i];
            }
            a->accum_grad(ga);
        }
    };
    return out;
}

static TensorPtr div_tt(const TensorPtr& a, const TensorPtr& b) {
    require_same_shape(a, b);
    std::vector<double> d(static_cast<std::size_t>(a->size));
    for (std::int64_t i = 0; i < a->size; ++i) {
        d[static_cast<std::size_t>(i)] = a->data[static_cast<std::size_t>(i)] / b->data[static_cast<std::size_t>(i)];
    }
    const bool req = a->requires_grad || b->requires_grad;
    auto out = make_tensor(std::move(d), a->shape, req);
    out->prev = {a, b};
    out->op = "/";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, b, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            std::vector<double> ga(static_cast<std::size_t>(a->size));
            for (std::int64_t i = 0; i < a->size; ++i) {
                const double bb = b->data[static_cast<std::size_t>(i)];
                ga[static_cast<std::size_t>(i)] = (1.0 / bb) * out_sp->grad[static_cast<std::size_t>(i)];
            }
            a->accum_grad(ga);
        }
        if (b->requires_grad) {
            std::vector<double> gb(static_cast<std::size_t>(b->size));
            for (std::int64_t i = 0; i < b->size; ++i) {
                const double aa = a->data[static_cast<std::size_t>(i)];
                const double bb = b->data[static_cast<std::size_t>(i)];
                gb[static_cast<std::size_t>(i)] = (-aa / (bb * bb)) * out_sp->grad[static_cast<std::size_t>(i)];
            }
            b->accum_grad(gb);
        }
    };
    return out;
}

static TensorPtr div_scalar(const TensorPtr& a, double c) {
    std::vector<double> d = a->data;
    for (auto& v : d) {
        v /= c;
    }
    auto out = make_tensor(std::move(d), a->shape, a->requires_grad);
    out->prev = {a};
    out->op = "/c";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp, c]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            std::vector<double> ga(out_sp->grad.size());
            const double inv = 1.0 / c;
            for (std::size_t i = 0; i < ga.size(); ++i) {
                ga[i] = inv * out_sp->grad[i];
            }
            a->accum_grad(ga);
        }
    };
    return out;
}

static TensorPtr pow_scalar(const TensorPtr& a, double p) {
    std::vector<double> d = a->data;
    for (auto& v : d) {
        v = std::pow(v, p);
    }
    auto out = make_tensor(std::move(d), a->shape, a->requires_grad);
    out->prev = {a};
    out->op = "pow";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, out_wp, p]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!a->requires_grad) {
            return;
        }
        std::vector<double> ga(static_cast<std::size_t>(a->size));
        for (std::int64_t i = 0; i < a->size; ++i) {
            const double aa = a->data[static_cast<std::size_t>(i)];
            ga[static_cast<std::size_t>(i)] = (p * std::pow(aa, p - 1.0)) * out_sp->grad[static_cast<std::size_t>(i)];
        }
        a->accum_grad(ga);
    };
    return out;
}

static TensorPtr matmul2d(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("matmul expects 2D tensors");
    }
    const std::int64_t m = a->shape[0];
    const std::int64_t n = a->shape[1];
    const std::int64_t n2 = b->shape[0];
    const std::int64_t p = b->shape[1];
    if (n != n2) {
        throw std::runtime_error("Inner dimensions mismatch");
    }

    const bool req = a->requires_grad || b->requires_grad;
    auto out = zeros({m, p}, req);
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t k = 0; k < n; ++k) {
            const double av = a->data[static_cast<std::size_t>(i * a->strides[0] + k * a->strides[1])];
            const auto base = i * out->strides[0];
            const auto bbase = k * b->strides[0];
            for (std::int64_t j = 0; j < p; ++j) {
                out->data[static_cast<std::size_t>(base + j * out->strides[1])] +=
                    av * b->data[static_cast<std::size_t>(bbase + j * b->strides[1])];
            }
        }
    }
    out->prev = {a, b};
    out->op = "matmul";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [a, b, out_wp, m, n, p]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (a->requires_grad) {
            std::vector<double> grad_a(static_cast<std::size_t>(a->size), 0.0);
            for (std::int64_t i = 0; i < m; ++i) {
                for (std::int64_t k = 0; k < n; ++k) {
                    double s = 0.0;
                    for (std::int64_t j = 0; j < p; ++j) {
                        s += out_sp->grad[static_cast<std::size_t>(i * out_sp->strides[0] + j * out_sp->strides[1])] *
                             b->data[static_cast<std::size_t>(k * b->strides[0] + j * b->strides[1])];
                    }
                    grad_a[static_cast<std::size_t>(i * a->strides[0] + k * a->strides[1])] += s;
                }
            }
            a->accum_grad(grad_a);
        }
        if (b->requires_grad) {
            std::vector<double> grad_b(static_cast<std::size_t>(b->size), 0.0);
            for (std::int64_t k = 0; k < n; ++k) {
                for (std::int64_t j = 0; j < p; ++j) {
                    double s = 0.0;
                    for (std::int64_t i = 0; i < m; ++i) {
                        s += a->data[static_cast<std::size_t>(i * a->strides[0] + k * a->strides[1])] *
                             out_sp->grad[static_cast<std::size_t>(i * out_sp->strides[0] + j * out_sp->strides[1])];
                    }
                    grad_b[static_cast<std::size_t>(k * b->strides[0] + j * b->strides[1])] += s;
                }
            }
            b->accum_grad(grad_b);
        }
    };
    return out;
}

static TensorPtr relu(const TensorPtr& x) {
    std::vector<double> d = x->data;
    for (auto& v : d) {
        v = (v > 0.0) ? v : 0.0;
    }
    auto out = make_tensor(std::move(d), x->shape, x->requires_grad);
    out->prev = {x};
    out->op = "relu";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [x, out_wp]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!x->requires_grad) {
            return;
        }
        std::vector<double> gx(static_cast<std::size_t>(x->size));
        for (std::int64_t i = 0; i < x->size; ++i) {
            const double xv = x->data[static_cast<std::size_t>(i)];
            const double g = out_sp->grad[static_cast<std::size_t>(i)];
            gx[static_cast<std::size_t>(i)] = (xv > 0.0) ? g : 0.0;
        }
        x->accum_grad(gx);
    };
    return out;
}

static TensorPtr dropout(const TensorPtr& x, double p, const std::int64_t* seed_opt, bool training) {
    if (!(p >= 0.0 && p < 1.0)) {
        throw std::runtime_error("dropout p must be in [0,1)");
    }
    const double keep = 1.0 - p;
    if (!training || p == 0.0) {
        // Identity op (keeps graph / grad passthrough).
        auto out = make_tensor(x->data, x->shape, x->requires_grad);
        out->prev = {x};
        out->op = "dropout";
        std::weak_ptr<Tensor> out_wp = out;
        out->backward_fn = [x, out_wp]() {
            auto out_sp = out_wp.lock();
            if (!out_sp) {
                return;
            }
            if (!x->requires_grad) {
                return;
            }
            x->accum_grad(out_sp->grad);
        };
        return out;
    }

    std::mt19937_64 rng;
    if (seed_opt) {
        rng.seed(static_cast<std::uint64_t>(*seed_opt));
    } else {
        std::random_device rd;
        rng.seed((static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd()));
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<double> mask(static_cast<std::size_t>(x->size));
    std::vector<double> d(static_cast<std::size_t>(x->size));
    const double inv_keep = 1.0 / keep;
    for (std::int64_t i = 0; i < x->size; ++i) {
        const double m = (unif(rng) < keep) ? 1.0 : 0.0;
        mask[static_cast<std::size_t>(i)] = m;
        d[static_cast<std::size_t>(i)] = x->data[static_cast<std::size_t>(i)] * m * inv_keep;
    }

    auto out = make_tensor(std::move(d), x->shape, x->requires_grad);
    out->prev = {x};
    out->op = "dropout";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [x, out_wp, mask = std::move(mask), inv_keep]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!x->requires_grad) {
            return;
        }
        std::vector<double> gx(static_cast<std::size_t>(x->size));
        for (std::int64_t i = 0; i < x->size; ++i) {
            gx[static_cast<std::size_t>(i)] = out_sp->grad[static_cast<std::size_t>(i)] * mask[static_cast<std::size_t>(i)] * inv_keep;
        }
        x->accum_grad(gx);
    };
    return out;
}

static TensorPtr batchnorm1d(
    const TensorPtr& x,
    const TensorPtr& gamma,
    const TensorPtr& beta,
    const TensorPtr& running_mean,
    const TensorPtr& running_var,
    double momentum,
    double eps,
    bool training
) {
    if (x->shape.size() != 2) {
        throw std::runtime_error("batchnorm1d expects x as (N,F)");
    }
    const std::int64_t N = x->shape[0];
    const std::int64_t F = x->shape[1];
    if (gamma->shape != std::vector<std::int64_t>{F} || beta->shape != std::vector<std::int64_t>{F}) {
        throw std::runtime_error("batchnorm1d expects gamma/beta shape (F,)");
    }
    if (running_mean && running_mean->shape != std::vector<std::int64_t>{F}) {
        throw std::runtime_error("batchnorm1d expects running_mean shape (F,)");
    }
    if (running_var && running_var->shape != std::vector<std::int64_t>{F}) {
        throw std::runtime_error("batchnorm1d expects running_var shape (F,)");
    }
    if (!(momentum >= 0.0 && momentum <= 1.0)) {
        throw std::runtime_error("batchnorm momentum must be in [0,1]");
    }
    if (!(eps > 0.0)) {
        throw std::runtime_error("batchnorm eps must be > 0");
    }

    std::vector<double> mean(static_cast<std::size_t>(F), 0.0);
    std::vector<double> var(static_cast<std::size_t>(F), 0.0);

    if (training || !running_mean || !running_var) {
        // Compute batch stats.
        const double invN = (N > 0) ? (1.0 / static_cast<double>(N)) : 0.0;
        for (std::int64_t n = 0; n < N; ++n) {
            const std::int64_t base = n * x->strides[0];
            for (std::int64_t f = 0; f < F; ++f) {
                mean[static_cast<std::size_t>(f)] += x->data[static_cast<std::size_t>(base + f * x->strides[1])];
            }
        }
        for (std::int64_t f = 0; f < F; ++f) {
            mean[static_cast<std::size_t>(f)] *= invN;
        }
        for (std::int64_t n = 0; n < N; ++n) {
            const std::int64_t base = n * x->strides[0];
            for (std::int64_t f = 0; f < F; ++f) {
                const double xm = x->data[static_cast<std::size_t>(base + f * x->strides[1])] - mean[static_cast<std::size_t>(f)];
                var[static_cast<std::size_t>(f)] += xm * xm;
            }
        }
        for (std::int64_t f = 0; f < F; ++f) {
            var[static_cast<std::size_t>(f)] *= invN;
        }
        // Update running stats if provided.
        if (training && running_mean && running_var) {
            for (std::int64_t f = 0; f < F; ++f) {
                running_mean->data[static_cast<std::size_t>(f)] =
                    momentum * running_mean->data[static_cast<std::size_t>(f)] + (1.0 - momentum) * mean[static_cast<std::size_t>(f)];
                running_var->data[static_cast<std::size_t>(f)] =
                    momentum * running_var->data[static_cast<std::size_t>(f)] + (1.0 - momentum) * var[static_cast<std::size_t>(f)];
            }
        }
    } else {
        // Inference stats.
        mean = running_mean->data;
        var = running_var->data;
    }

    std::vector<double> invstd(static_cast<std::size_t>(F));
    for (std::int64_t f = 0; f < F; ++f) {
        invstd[static_cast<std::size_t>(f)] = 1.0 / std::sqrt(var[static_cast<std::size_t>(f)] + eps);
    }

    const bool req = x->requires_grad || gamma->requires_grad || beta->requires_grad;
    std::vector<double> y(static_cast<std::size_t>(x->size));
    std::vector<double> xhat(static_cast<std::size_t>(x->size));
    for (std::int64_t n = 0; n < N; ++n) {
        const std::int64_t base = n * x->strides[0];
        for (std::int64_t f = 0; f < F; ++f) {
            const std::int64_t idx = base + f * x->strides[1];
            const double xn = (x->data[static_cast<std::size_t>(idx)] - mean[static_cast<std::size_t>(f)]) * invstd[static_cast<std::size_t>(f)];
            xhat[static_cast<std::size_t>(idx)] = xn;
            y[static_cast<std::size_t>(idx)] = xn * gamma->data[static_cast<std::size_t>(f)] + beta->data[static_cast<std::size_t>(f)];
        }
    }

    auto out = make_tensor(std::move(y), x->shape, req);
    out->prev = {x, gamma, beta};
    out->op = "batchnorm1d";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [x, gamma, beta, out_wp, xhat = std::move(xhat), invstd, N, F]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        const double m = (N > 0) ? static_cast<double>(N) : 1.0;

        if (beta->requires_grad) {
            std::vector<double> db(static_cast<std::size_t>(F), 0.0);
            for (std::int64_t n = 0; n < N; ++n) {
                const std::int64_t base = n * out_sp->strides[0];
                for (std::int64_t f = 0; f < F; ++f) {
                    db[static_cast<std::size_t>(f)] += out_sp->grad[static_cast<std::size_t>(base + f * out_sp->strides[1])];
                }
            }
            beta->accum_grad(db);
        }

        std::vector<double> dgamma(static_cast<std::size_t>(F), 0.0);
        if (gamma->requires_grad || x->requires_grad) {
            for (std::int64_t n = 0; n < N; ++n) {
                const std::int64_t base = n * out_sp->strides[0];
                for (std::int64_t f = 0; f < F; ++f) {
                    const std::int64_t idx = base + f * out_sp->strides[1];
                    dgamma[static_cast<std::size_t>(f)] += out_sp->grad[static_cast<std::size_t>(idx)] * xhat[static_cast<std::size_t>(idx)];
                }
            }
            if (gamma->requires_grad) {
                gamma->accum_grad(dgamma);
            }
        }

        if (x->requires_grad) {
            // dx = (1/m) * gamma * invstd * (m*dy - sum(dy) - xhat*sum(dy*xhat))
            std::vector<double> sum_dy(static_cast<std::size_t>(F), 0.0);
            for (std::int64_t n = 0; n < N; ++n) {
                const std::int64_t base = n * out_sp->strides[0];
                for (std::int64_t f = 0; f < F; ++f) {
                    sum_dy[static_cast<std::size_t>(f)] += out_sp->grad[static_cast<std::size_t>(base + f * out_sp->strides[1])];
                }
            }

            std::vector<double> gx(static_cast<std::size_t>(x->size), 0.0);
            for (std::int64_t n = 0; n < N; ++n) {
                const std::int64_t base = n * out_sp->strides[0];
                for (std::int64_t f = 0; f < F; ++f) {
                    const std::int64_t idx = base + f * out_sp->strides[1];
                    const double dy = out_sp->grad[static_cast<std::size_t>(idx)];
                    const double g = gamma->data[static_cast<std::size_t>(f)];
                    const double invs = invstd[static_cast<std::size_t>(f)];
                    gx[static_cast<std::size_t>(idx)] =
                        (g * invs / m) * (m * dy - sum_dy[static_cast<std::size_t>(f)] - xhat[static_cast<std::size_t>(idx)] * dgamma[static_cast<std::size_t>(f)]);
                }
            }
            x->accum_grad(gx);
        }
    };
    return out;
}

static TensorPtr batchnorm2d(
    const TensorPtr& x,
    const TensorPtr& gamma,
    const TensorPtr& beta,
    const TensorPtr& running_mean,
    const TensorPtr& running_var,
    double momentum,
    double eps,
    bool training
) {
    if (x->shape.size() != 4) {
        throw std::runtime_error("batchnorm2d expects x as (N,C,H,W)");
    }
    const std::int64_t N = x->shape[0];
    const std::int64_t C = x->shape[1];
    const std::int64_t H = x->shape[2];
    const std::int64_t W = x->shape[3];
    if (gamma->shape != std::vector<std::int64_t>{C} || beta->shape != std::vector<std::int64_t>{C}) {
        throw std::runtime_error("batchnorm2d expects gamma/beta shape (C,)");
    }
    if (running_mean && running_mean->shape != std::vector<std::int64_t>{C}) {
        throw std::runtime_error("batchnorm2d expects running_mean shape (C,)");
    }
    if (running_var && running_var->shape != std::vector<std::int64_t>{C}) {
        throw std::runtime_error("batchnorm2d expects running_var shape (C,)");
    }
    if (!(momentum >= 0.0 && momentum <= 1.0)) {
        throw std::runtime_error("batchnorm momentum must be in [0,1]");
    }
    if (!(eps > 0.0)) {
        throw std::runtime_error("batchnorm eps must be > 0");
    }

    const double m = static_cast<double>(std::max<std::int64_t>(1, N * H * W));
    std::vector<double> mean(static_cast<std::size_t>(C), 0.0);
    std::vector<double> var(static_cast<std::size_t>(C), 0.0);

    const auto xs0 = x->strides[0];
    const auto xs1 = x->strides[1];
    const auto xs2 = x->strides[2];
    const auto xs3 = x->strides[3];

    if (training || !running_mean || !running_var) {
        // Compute per-channel mean.
        for (std::int64_t n = 0; n < N; ++n) {
            const std::int64_t nb = n * xs0;
            for (std::int64_t c = 0; c < C; ++c) {
                const std::int64_t cb = nb + c * xs1;
                double acc = 0.0;
                for (std::int64_t h = 0; h < H; ++h) {
                    const std::int64_t hb = cb + h * xs2;
                    for (std::int64_t w = 0; w < W; ++w) {
                        acc += x->data[static_cast<std::size_t>(hb + w * xs3)];
                    }
                }
                mean[static_cast<std::size_t>(c)] += acc;
            }
        }
        for (std::int64_t c = 0; c < C; ++c) {
            mean[static_cast<std::size_t>(c)] /= m;
        }
        // Compute per-channel variance.
        for (std::int64_t n = 0; n < N; ++n) {
            const std::int64_t nb = n * xs0;
            for (std::int64_t c = 0; c < C; ++c) {
                const std::int64_t cb = nb + c * xs1;
                double acc = 0.0;
                const double mu = mean[static_cast<std::size_t>(c)];
                for (std::int64_t h = 0; h < H; ++h) {
                    const std::int64_t hb = cb + h * xs2;
                    for (std::int64_t w = 0; w < W; ++w) {
                        const double xm = x->data[static_cast<std::size_t>(hb + w * xs3)] - mu;
                        acc += xm * xm;
                    }
                }
                var[static_cast<std::size_t>(c)] += acc;
            }
        }
        for (std::int64_t c = 0; c < C; ++c) {
            var[static_cast<std::size_t>(c)] /= m;
        }
        if (training && running_mean && running_var) {
            for (std::int64_t c = 0; c < C; ++c) {
                running_mean->data[static_cast<std::size_t>(c)] =
                    momentum * running_mean->data[static_cast<std::size_t>(c)] + (1.0 - momentum) * mean[static_cast<std::size_t>(c)];
                running_var->data[static_cast<std::size_t>(c)] =
                    momentum * running_var->data[static_cast<std::size_t>(c)] + (1.0 - momentum) * var[static_cast<std::size_t>(c)];
            }
        }
    } else {
        mean = running_mean->data;
        var = running_var->data;
    }

    std::vector<double> invstd(static_cast<std::size_t>(C));
    for (std::int64_t c = 0; c < C; ++c) {
        invstd[static_cast<std::size_t>(c)] = 1.0 / std::sqrt(var[static_cast<std::size_t>(c)] + eps);
    }

    const bool req = x->requires_grad || gamma->requires_grad || beta->requires_grad;
    std::vector<double> y(static_cast<std::size_t>(x->size));
    std::vector<double> xhat(static_cast<std::size_t>(x->size));
    for (std::int64_t n = 0; n < N; ++n) {
        const std::int64_t nb = n * xs0;
        for (std::int64_t c = 0; c < C; ++c) {
            const std::int64_t cb = nb + c * xs1;
            const double mu = mean[static_cast<std::size_t>(c)];
            const double invs = invstd[static_cast<std::size_t>(c)];
            const double g = gamma->data[static_cast<std::size_t>(c)];
            const double be = beta->data[static_cast<std::size_t>(c)];
            for (std::int64_t h = 0; h < H; ++h) {
                const std::int64_t hb = cb + h * xs2;
                for (std::int64_t w = 0; w < W; ++w) {
                    const std::int64_t idx = hb + w * xs3;
                    const double xn = (x->data[static_cast<std::size_t>(idx)] - mu) * invs;
                    xhat[static_cast<std::size_t>(idx)] = xn;
                    y[static_cast<std::size_t>(idx)] = xn * g + be;
                }
            }
        }
    }

    auto out = make_tensor(std::move(y), x->shape, req);
    out->prev = {x, gamma, beta};
    out->op = "batchnorm2d";
    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [x, gamma, beta, out_wp, xhat = std::move(xhat), invstd, N, C, H, W, xs0, xs1, xs2, xs3]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        const double m = static_cast<double>(std::max<std::int64_t>(1, N * H * W));

        std::vector<double> sum_dy(static_cast<std::size_t>(C), 0.0);
        std::vector<double> sum_dy_xhat(static_cast<std::size_t>(C), 0.0);

        // Accumulate sums.
        for (std::int64_t n = 0; n < N; ++n) {
            const std::int64_t nb = n * out_sp->strides[0];
            for (std::int64_t c = 0; c < C; ++c) {
                const std::int64_t cb = nb + c * out_sp->strides[1];
                double s1 = 0.0;
                double s2 = 0.0;
                for (std::int64_t h = 0; h < H; ++h) {
                    const std::int64_t hb = cb + h * out_sp->strides[2];
                    for (std::int64_t w = 0; w < W; ++w) {
                        const std::int64_t idx = hb + w * out_sp->strides[3];
                        const double dy = out_sp->grad[static_cast<std::size_t>(idx)];
                        s1 += dy;
                        s2 += dy * xhat[static_cast<std::size_t>(idx)];
                    }
                }
                sum_dy[static_cast<std::size_t>(c)] += s1;
                sum_dy_xhat[static_cast<std::size_t>(c)] += s2;
            }
        }

        if (beta->requires_grad) {
            beta->accum_grad(sum_dy);
        }
        if (gamma->requires_grad) {
            gamma->accum_grad(sum_dy_xhat);
        }

        if (x->requires_grad) {
            std::vector<double> gx(static_cast<std::size_t>(x->size), 0.0);
            for (std::int64_t n = 0; n < N; ++n) {
                const std::int64_t nb_out = n * out_sp->strides[0];
                const std::int64_t nb_x = n * xs0;
                for (std::int64_t c = 0; c < C; ++c) {
                    const std::int64_t cb_out = nb_out + c * out_sp->strides[1];
                    const std::int64_t cb_x = nb_x + c * xs1;
                    const double g = gamma->data[static_cast<std::size_t>(c)];
                    const double invs = invstd[static_cast<std::size_t>(c)];
                    const double sdy = sum_dy[static_cast<std::size_t>(c)];
                    const double sdyxh = sum_dy_xhat[static_cast<std::size_t>(c)];
                    for (std::int64_t h = 0; h < H; ++h) {
                        const std::int64_t hb_out = cb_out + h * out_sp->strides[2];
                        const std::int64_t hb_x = cb_x + h * xs2;
                        for (std::int64_t w = 0; w < W; ++w) {
                            const std::int64_t idx_out = hb_out + w * out_sp->strides[3];
                            const std::int64_t idx_x = hb_x + w * xs3;
                            const double dy = out_sp->grad[static_cast<std::size_t>(idx_out)];
                            const double xh = xhat[static_cast<std::size_t>(idx_out)];
                            gx[static_cast<std::size_t>(idx_x)] =
                                (g * invs / m) * (m * dy - sdy - xh * sdyxh);
                        }
                    }
                }
            }
            x->accum_grad(gx);
        }
    };
    return out;
}

static TensorPtr conv2d(const TensorPtr& x, const TensorPtr& w, const TensorPtr& b, std::int64_t stride, std::int64_t padding) {
    if (x->shape.size() != 4 || w->shape.size() != 4) {
        throw std::runtime_error("conv2d expects x as (N,C,H,W) and w as (O,C,KH,KW)");
    }
    const std::int64_t N = x->shape[0];
    const std::int64_t C_in = x->shape[1];
    const std::int64_t H = x->shape[2];
    const std::int64_t W = x->shape[3];
    const std::int64_t C_out = w->shape[0];
    const std::int64_t C_in2 = w->shape[1];
    const std::int64_t KH = w->shape[2];
    const std::int64_t KW = w->shape[3];
    if (C_in != C_in2) {
        throw std::runtime_error("conv2d channel mismatch");
    }
    if (b && b->shape != std::vector<std::int64_t>{C_out}) {
        throw std::runtime_error("bias must have shape (C_out,)");
    }
    if (stride <= 0 || padding < 0) {
        throw std::runtime_error("stride must be >0 and padding >=0");
    }

    const std::int64_t H_out = (H + 2 * padding - KH) / stride + 1;
    const std::int64_t W_out = (W + 2 * padding - KW) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("Output spatial size is non-positive; check padding/stride/kernel");
    }

    const bool req = x->requires_grad || w->requires_grad || (b && b->requires_grad);
    auto out = zeros({N, C_out, H_out, W_out}, req);
    out->op = "conv2d";
    out->prev.clear();
    out->prev.push_back(x);
    out->prev.push_back(w);
    if (b) {
        out->prev.push_back(b);
    }

    const auto xs0 = x->strides[0];
    const auto xs1 = x->strides[1];
    const auto xs2 = x->strides[2];
    const auto xs3 = x->strides[3];

    const auto ws0 = w->strides[0];
    const auto ws1 = w->strides[1];
    const auto ws2 = w->strides[2];
    const auto ws3 = w->strides[3];

    const auto os0 = out->strides[0];
    const auto os1 = out->strides[1];
    const auto os2 = out->strides[2];
    const auto os3 = out->strides[3];

    // conv2d (no BLAS/GEMM). Optional padding buffer.
    const double* xdata = x->data.data();
    const double* wdata = w->data.data();
    const double* bdata = (b ? b->data.data() : nullptr);

    const bool use_padding_buffer = (padding > 0);
    const std::int64_t H_pad = H + 2 * padding;
    const std::int64_t W_pad = W + 2 * padding;

    auto conv_one_n = [&](std::int64_t n, const double* xn, std::int64_t xs0e, std::int64_t xs1e, std::int64_t xs2e, std::int64_t xs3e, std::int64_t He, std::int64_t We) {
        auto conv_one_co = [&](std::int64_t co) {
            const double bias_val = (bdata ? bdata[static_cast<std::size_t>(co)] : 0.0);
            for (std::int64_t ho = 0; ho < H_out; ++ho) {
                const std::int64_t hi0 = ho * stride;
                for (std::int64_t wo = 0; wo < W_out; ++wo) {
                    const std::int64_t wi0 = wo * stride;
                    double acc = bias_val;
                    for (std::int64_t ci = 0; ci < C_in; ++ci) {
                        const std::int64_t x_base = /*n already in xn*/ ci * xs1e;
                        const std::int64_t w_base = co * ws0 + ci * ws1;
                        for (std::int64_t kh = 0; kh < KH; ++kh) {
                            const std::int64_t x_row = x_base + (hi0 + kh) * xs2e + wi0 * xs3e;
                            const std::int64_t w_row = w_base + kh * ws2;

                            if (xs3e == 1 && ws3 == 1) {
                                const double* xp = xn + static_cast<std::size_t>(x_row);
                                const double* wp = wdata + static_cast<std::size_t>(w_row);
                                for (std::int64_t kw = 0; kw < KW; ++kw) {
                                    acc += xp[static_cast<std::size_t>(kw)] * wp[static_cast<std::size_t>(kw)];
                                }
                            } else {
                                for (std::int64_t kw = 0; kw < KW; ++kw) {
                                    acc += xn[static_cast<std::size_t>(x_row + kw * xs3e)] *
                                           wdata[static_cast<std::size_t>(w_row + kw * ws3)];
                                }
                            }
                        }
                    }
                    out->data[static_cast<std::size_t>(n * os0 + co * os1 + ho * os2 + wo * os3)] = acc;
                }
            }
        };

        for (std::int64_t co = 0; co < C_out; ++co) {
            conv_one_co(co);
        }
    };

    if (!use_padding_buffer) {
        // No padding: directly convolve over x.
        for (std::int64_t n = 0; n < N; ++n) {
            const double* xn = xdata + static_cast<std::size_t>(n * xs0);
            conv_one_n(n, xn, xs0, xs1, xs2, xs3, H, W);
        }
    } else {
        // Padding: build a zero-padded contiguous buffer per image and convolve.
        const std::int64_t xs0p = C_in * H_pad * W_pad;
        const std::int64_t xs1p = H_pad * W_pad;
        const std::int64_t xs2p = W_pad;
        const std::int64_t xs3p = 1;

        std::vector<double> xpad(static_cast<std::size_t>(xs0p), 0.0);
        for (std::int64_t n = 0; n < N; ++n) {
            std::fill(xpad.begin(), xpad.end(), 0.0);
            for (std::int64_t ci = 0; ci < C_in; ++ci) {
                for (std::int64_t h = 0; h < H; ++h) {
                    const double* src = xdata + static_cast<std::size_t>(n * xs0 + ci * xs1 + h * xs2);
                    double* dst = xpad.data() + static_cast<std::size_t>(ci * xs1p + (h + padding) * xs2p + padding);
                    if (xs3 == 1) {
                        std::copy(src, src + static_cast<std::size_t>(W), dst);
                    } else {
                        for (std::int64_t wj = 0; wj < W; ++wj) {
                            dst[static_cast<std::size_t>(wj)] = src[static_cast<std::size_t>(wj * xs3)];
                        }
                    }
                }
            }

            const double* xn = xpad.data();
            conv_one_n(n, xn, xs0p, xs1p, xs2p, xs3p, H_pad, W_pad);
        }
    }

    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [x, w, b, out_wp, N, C_in, H, W, C_out, KH, KW, stride, padding,
                        xs0, xs1, xs2, xs3, ws0, ws1, ws2, ws3, os0, os1, os2, os3, H_out, W_out]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }

        // dx
        if (x->requires_grad) {
            std::vector<double> dx(static_cast<std::size_t>(x->size), 0.0);
            if (stride == 1 && padding == 0) {
                for (std::int64_t n = 0; n < N; ++n) {
                    for (std::int64_t co = 0; co < C_out; ++co) {
                        for (std::int64_t ho = 0; ho < H_out; ++ho) {
                            for (std::int64_t wo = 0; wo < W_out; ++wo) {
                                const double go = out_sp->grad[static_cast<std::size_t>(n * os0 + co * os1 + ho * os2 + wo * os3)];
                                if (go == 0.0) {
                                    continue;
                                }
                                for (std::int64_t ci = 0; ci < C_in; ++ci) {
                                    const std::int64_t x_base = n * xs0 + ci * xs1 + ho * xs2 + wo * xs3;
                                    const std::int64_t w_base = co * ws0 + ci * ws1;
                                    for (std::int64_t kh = 0; kh < KH; ++kh) {
                                        const std::int64_t x_row = x_base + kh * xs2;
                                        const std::int64_t w_row = w_base + kh * ws2;
                                        for (std::int64_t kw = 0; kw < KW; ++kw) {
                                            dx[static_cast<std::size_t>(x_row + kw * xs3)] +=
                                                w->data[static_cast<std::size_t>(w_row + kw * ws3)] * go;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for (std::int64_t n = 0; n < N; ++n) {
                    for (std::int64_t co = 0; co < C_out; ++co) {
                        for (std::int64_t ho = 0; ho < H_out; ++ho) {
                            const std::int64_t hi0 = ho * stride - padding;
                            const std::int64_t kh_start = std::max<std::int64_t>(0, -hi0);
                            const std::int64_t kh_end = std::min<std::int64_t>(KH, H - hi0);
                            for (std::int64_t wo = 0; wo < W_out; ++wo) {
                                const std::int64_t wi0 = wo * stride - padding;
                                const std::int64_t kw_start = std::max<std::int64_t>(0, -wi0);
                                const std::int64_t kw_end = std::min<std::int64_t>(KW, W - wi0);
                                const double go = out_sp->grad[static_cast<std::size_t>(n * os0 + co * os1 + ho * os2 + wo * os3)];
                                if (go == 0.0) {
                                    continue;
                                }
                                for (std::int64_t ci = 0; ci < C_in; ++ci) {
                                    const std::int64_t x_base = n * xs0 + ci * xs1;
                                    const std::int64_t w_base = co * ws0 + ci * ws1;
                                    for (std::int64_t kh = kh_start; kh < kh_end; ++kh) {
                                        const std::int64_t hi = hi0 + kh;
                                        const std::int64_t x_row = x_base + hi * xs2;
                                        const std::int64_t w_row = w_base + kh * ws2;
                                        for (std::int64_t kw = kw_start; kw < kw_end; ++kw) {
                                            const std::int64_t wi = wi0 + kw;
                                            dx[static_cast<std::size_t>(x_row + wi * xs3)] +=
                                                w->data[static_cast<std::size_t>(w_row + kw * ws3)] * go;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            x->accum_grad(dx);
        }

        // dw
        if (w->requires_grad) {
            std::vector<double> dw(static_cast<std::size_t>(w->size), 0.0);

            for (std::int64_t co = 0; co < C_out; ++co) {
                for (std::int64_t ci = 0; ci < C_in; ++ci) {
                    for (std::int64_t kh = 0; kh < KH; ++kh) {
                        for (std::int64_t kw = 0; kw < KW; ++kw) {
                            double acc = 0.0;
                            for (std::int64_t n = 0; n < N; ++n) {
                                for (std::int64_t ho = 0; ho < H_out; ++ho) {
                                    const std::int64_t hi = ho * stride - padding + kh;
                                    if (hi < 0 || hi >= H) {
                                        continue;
                                    }
                                    const std::int64_t x_row = n * xs0 + ci * xs1 + hi * xs2;
                                    const std::int64_t o_row = n * os0 + co * os1 + ho * os2;
                                    for (std::int64_t wo = 0; wo < W_out; ++wo) {
                                        const std::int64_t wi = wo * stride - padding + kw;
                                        if (wi < 0 || wi >= W) {
                                            continue;
                                        }
                                        const double go = out_sp->grad[static_cast<std::size_t>(o_row + wo * os3)];
                                        acc += x->data[static_cast<std::size_t>(x_row + wi * xs3)] * go;
                                    }
                                }
                            }
                            const std::int64_t idx = co * ws0 + ci * ws1 + kh * ws2 + kw * ws3;
                            dw[static_cast<std::size_t>(idx)] = acc;
                        }
                    }
                }
            }
            w->accum_grad(dw);
        }

        // db
        if (b && b->requires_grad) {
            std::vector<double> db(static_cast<std::size_t>(b->size), 0.0);

            for (std::int64_t co = 0; co < C_out; ++co) {
                double s = 0.0;
                for (std::int64_t n = 0; n < N; ++n) {
                    const std::int64_t base = n * os0 + co * os1;
                    for (std::int64_t ho = 0; ho < H_out; ++ho) {
                        const std::int64_t row = base + ho * os2;
                        for (std::int64_t wo = 0; wo < W_out; ++wo) {
                            s += out_sp->grad[static_cast<std::size_t>(row + wo * os3)];
                        }
                    }
                }
                db[static_cast<std::size_t>(co)] = s;
            }
            b->accum_grad(db);
        }
    };

    return out;
}

static TensorPtr maxpool2d(const TensorPtr& x, std::int64_t kernel, const std::int64_t* stride_opt) {
    if (x->shape.size() != 4) {
        throw std::runtime_error("maxpool2d expects x as (N,C,H,W)");
    }
    std::int64_t stride = (stride_opt ? *stride_opt : kernel);
    if (kernel <= 0 || stride <= 0) {
        throw std::runtime_error("kernel/stride must be > 0");
    }
    const std::int64_t N = x->shape[0];
    const std::int64_t C = x->shape[1];
    const std::int64_t H = x->shape[2];
    const std::int64_t W = x->shape[3];
    const std::int64_t H_out = (H - kernel) / stride + 1;
    const std::int64_t W_out = (W - kernel) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("Output spatial size is non-positive");
    }

    auto out = zeros({N, C, H_out, W_out}, x->requires_grad);
    out->op = "maxpool2d";
    out->prev = {x};

    const auto xs0 = x->strides[0];
    const auto xs1 = x->strides[1];
    const auto xs2 = x->strides[2];
    const auto xs3 = x->strides[3];
    const auto os0 = out->strides[0];
    const auto os1 = out->strides[1];
    const auto os2 = out->strides[2];
    const auto os3 = out->strides[3];

    std::vector<std::int64_t> argmax(static_cast<std::size_t>(out->size), 0);
    std::int64_t idx = 0;
    for (std::int64_t n = 0; n < N; ++n) {
        for (std::int64_t c = 0; c < C; ++c) {
            for (std::int64_t ho = 0; ho < H_out; ++ho) {
                const std::int64_t hi0 = ho * stride;
                for (std::int64_t wo = 0; wo < W_out; ++wo) {
                    const std::int64_t wi0 = wo * stride;
                    double best_val = -std::numeric_limits<double>::infinity();
                    std::int64_t best_flat = 0;
                    for (std::int64_t kh = 0; kh < kernel; ++kh) {
                        const std::int64_t hi = hi0 + kh;
                        for (std::int64_t kw = 0; kw < kernel; ++kw) {
                            const std::int64_t wi = wi0 + kw;
                            const std::int64_t flat = n * xs0 + c * xs1 + hi * xs2 + wi * xs3;
                            const double v = x->data[static_cast<std::size_t>(flat)];
                            if (v > best_val) {
                                best_val = v;
                                best_flat = flat;
                            }
                        }
                    }
                    out->data[static_cast<std::size_t>(n * os0 + c * os1 + ho * os2 + wo * os3)] = best_val;
                    argmax[static_cast<std::size_t>(idx)] = best_flat;
                    ++idx;
                }
            }
        }
    }

    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [x, out_wp, argmax = std::move(argmax)]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (x->requires_grad) {
            std::vector<double> dx(static_cast<std::size_t>(x->size), 0.0);
            for (std::int64_t i = 0; i < out_sp->size; ++i) {
                dx[static_cast<std::size_t>(argmax[static_cast<std::size_t>(i)])] +=
                    out_sp->grad[static_cast<std::size_t>(i)];
            }
            x->accum_grad(dx);
        }
    };

    return out;
}

static TensorPtr mse_loss(const TensorPtr& pred, const TensorPtr& target) {
    require_same_shape(pred, target);
    auto diff = sub(pred, target);
    auto sq = mul(diff, diff);
    auto s = sum_all(sq);
    auto out = mul_scalar(s, 1.0 / static_cast<double>(pred->size));
    return out;
}

static TensorPtr softmax2d(const TensorPtr& x) {
    if (x->shape.size() != 2) {
        throw std::runtime_error("softmax expects (N,C)");
    }
    const std::int64_t N = x->shape[0];
    const std::int64_t C = x->shape[1];
    auto out = zeros({N, C}, x->requires_grad);
    out->op = "softmax";
    out->prev = {x};

    // stable softmax row-wise
    for (std::int64_t n = 0; n < N; ++n) {
        double m = -std::numeric_limits<double>::infinity();
        for (std::int64_t c = 0; c < C; ++c) {
            const double v = x->data[static_cast<std::size_t>(n * x->strides[0] + c * x->strides[1])];
            if (v > m) {
                m = v;
            }
        }
        double s = 0.0;
        for (std::int64_t c = 0; c < C; ++c) {
            const double v = x->data[static_cast<std::size_t>(n * x->strides[0] + c * x->strides[1])];
            const double e = std::exp(v - m);
            out->data[static_cast<std::size_t>(n * out->strides[0] + c * out->strides[1])] = e;
            s += e;
        }
        const double invs = 1.0 / s;
        for (std::int64_t c = 0; c < C; ++c) {
            out->data[static_cast<std::size_t>(n * out->strides[0] + c * out->strides[1])] *= invs;
        }
    }

    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [x, out_wp, N, C]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!x->requires_grad) {
            return;
        }
        // JVP: dx = p * (g - sum(g*p)) row-wise
        std::vector<double> dx(static_cast<std::size_t>(x->size), 0.0);
        for (std::int64_t n = 0; n < N; ++n) {
            double dot = 0.0;
            for (std::int64_t c = 0; c < C; ++c) {
                const auto idx = static_cast<std::size_t>(n * out_sp->strides[0] + c * out_sp->strides[1]);
                dot += out_sp->grad[idx] * out_sp->data[idx];
            }
            for (std::int64_t c = 0; c < C; ++c) {
                const auto oidx = static_cast<std::size_t>(n * out_sp->strides[0] + c * out_sp->strides[1]);
                const double p = out_sp->data[oidx];
                const double g = out_sp->grad[oidx];
                dx[static_cast<std::size_t>(n * x->strides[0] + c * x->strides[1])] += p * (g - dot);
            }
        }
        x->accum_grad(dx);
    };

    return out;
}

static TensorPtr categorical_crossentropy_probs(const TensorPtr& probs, const std::vector<std::int64_t>& targets, double eps) {
    if (probs->shape.size() != 2) {
        throw std::runtime_error("categorical_crossentropy expects probs (N,C)");
    }
    const std::int64_t N = probs->shape[0];
    const std::int64_t C = probs->shape[1];
    if (static_cast<std::int64_t>(targets.size()) != N) {
        throw std::runtime_error("targets length must equal N");
    }
    if (!(eps > 0.0)) {
        throw std::runtime_error("eps must be > 0");
    }

    double loss_val = 0.0;
    for (std::int64_t n = 0; n < N; ++n) {
        const auto t = targets[static_cast<std::size_t>(n)];
        if (t < 0 || t >= C) {
            throw std::runtime_error("target out of range");
        }
        const double p = probs->data[static_cast<std::size_t>(n * probs->strides[0] + t * probs->strides[1])];
        const double pc = (p > eps ? p : eps);
        loss_val += -std::log(pc);
    }
    loss_val /= static_cast<double>(N);

    auto out = make_tensor(std::vector<double>{loss_val}, {}, probs->requires_grad);
    out->op = "categorical_crossentropy";
    out->prev = {probs};

    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [probs, out_wp, targets, eps, N, C]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!probs->requires_grad) {
            return;
        }
        if (out_sp->grad.empty()) {
            return;
        }
        const double g = out_sp->grad[0];
        std::vector<double> dprobs(static_cast<std::size_t>(probs->size), 0.0);
        const double invN = 1.0 / static_cast<double>(N);
        for (std::int64_t n = 0; n < N; ++n) {
            const std::int64_t t = targets[static_cast<std::size_t>(n)];
            const std::size_t idx = static_cast<std::size_t>(n * probs->strides[0] + t * probs->strides[1]);
            const double p = probs->data[idx];
            if (p <= eps) {
                continue;
            }
            dprobs[idx] += (-invN / p) * g;
        }
        probs->accum_grad(dprobs);
    };

    return out;
}

static TensorPtr categorical_crossentropy_logits(const TensorPtr& logits, const std::vector<std::int64_t>& targets) {
    if (logits->shape.size() != 2) {
        throw std::runtime_error("categorical_crossentropy_logits expects logits (N,C)");
    }
    const std::int64_t N = logits->shape[0];
    const std::int64_t C = logits->shape[1];
    if (static_cast<std::int64_t>(targets.size()) != N) {
        throw std::runtime_error("targets length must equal N");
    }

    double loss_val = 0.0;
    for (std::int64_t n = 0; n < N; ++n) {
        const auto t = targets[static_cast<std::size_t>(n)];
        if (t < 0 || t >= C) {
            throw std::runtime_error("target out of range");
        }
        double m = -std::numeric_limits<double>::infinity();
        for (std::int64_t c = 0; c < C; ++c) {
            const double v = logits->data[static_cast<std::size_t>(n * logits->strides[0] + c * logits->strides[1])];
            if (v > m) {
                m = v;
            }
        }
        double sumexp = 0.0;
        for (std::int64_t c = 0; c < C; ++c) {
            const double v = logits->data[static_cast<std::size_t>(n * logits->strides[0] + c * logits->strides[1])];
            sumexp += std::exp(v - m);
        }
        const double logsumexp = m + std::log(sumexp);
        const double zt = logits->data[static_cast<std::size_t>(n * logits->strides[0] + t * logits->strides[1])];
        loss_val += -(zt - logsumexp);
    }
    loss_val /= static_cast<double>(N);

    auto out = make_tensor(std::vector<double>{loss_val}, {}, logits->requires_grad);
    out->op = "categorical_crossentropy_logits";
    out->prev = {logits};

    std::weak_ptr<Tensor> out_wp = out;
    out->backward_fn = [logits, out_wp, targets, N, C]() {
        auto out_sp = out_wp.lock();
        if (!out_sp) {
            return;
        }
        if (!logits->requires_grad) {
            return;
        }
        if (out_sp->grad.empty()) {
            return;
        }
        const double g = out_sp->grad[0];
        const double invN = 1.0 / static_cast<double>(N);

        std::vector<double> dlogits(static_cast<std::size_t>(logits->size), 0.0);
        for (std::int64_t n = 0; n < N; ++n) {
            const std::int64_t t = targets[static_cast<std::size_t>(n)];
            double m = -std::numeric_limits<double>::infinity();
            for (std::int64_t c = 0; c < C; ++c) {
                const double v = logits->data[static_cast<std::size_t>(n * logits->strides[0] + c * logits->strides[1])];
                if (v > m) {
                    m = v;
                }
            }
            double sumexp = 0.0;
            for (std::int64_t c = 0; c < C; ++c) {
                const double v = logits->data[static_cast<std::size_t>(n * logits->strides[0] + c * logits->strides[1])];
                sumexp += std::exp(v - m);
            }
            const double invsumexp = 1.0 / sumexp;
            for (std::int64_t c = 0; c < C; ++c) {
                const double v = logits->data[static_cast<std::size_t>(n * logits->strides[0] + c * logits->strides[1])];
                const double p = std::exp(v - m) * invsumexp;
                const double y = (c == t) ? 1.0 : 0.0;
                const std::size_t idx = static_cast<std::size_t>(n * logits->strides[0] + c * logits->strides[1]);
                dlogits[idx] += (p - y) * invN * g;
            }
        }
        logits->accum_grad(dlogits);
    };

    return out;
}

// ---- C ABI for ctypes ----

using tensor_handle = void*;

static TensorPtr& handle_ref(tensor_handle h) {
    if (!h) {
        throw std::runtime_error("null tensor handle");
    }
    return *reinterpret_cast<TensorPtr*>(h);
}

static tensor_handle make_handle(TensorPtr t) {
    return reinterpret_cast<tensor_handle>(new TensorPtr(std::move(t)));
}

} // namespace

extern "C" {

const char* tensor_last_error() {
    return g_last_error.c_str();
}

tensor_handle tensor_create(const double* data, std::int64_t data_len, const std::int64_t* shape, std::int64_t shape_len, int requires_grad) {
    try {
        if (!data || data_len < 0) {
            throw std::runtime_error("data must be non-null and data_len >= 0");
        }
        std::vector<std::int64_t> s;
        s.reserve(static_cast<std::size_t>(shape_len));
        for (std::int64_t i = 0; i < shape_len; ++i) {
            s.push_back(shape ? shape[i] : 0);
        }
        std::vector<double> d(static_cast<std::size_t>(data_len));
        std::memcpy(d.data(), data, static_cast<std::size_t>(data_len) * sizeof(double));
        auto t = make_tensor(std::move(d), s, requires_grad != 0);
        g_last_error.clear();
        return make_handle(std::move(t));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

void tensor_free(tensor_handle h) {
    try {
        if (!h) {
            return;
        }
        delete reinterpret_cast<TensorPtr*>(h);
        g_last_error.clear();
    } catch (const std::exception& e) {
        set_last_error(e.what());
    }
}

std::int64_t tensor_ndim(tensor_handle h) {
    try {
        return static_cast<std::int64_t>(handle_ref(h)->shape.size());
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

std::int64_t tensor_size(tensor_handle h) {
    try {
        return handle_ref(h)->size;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

int tensor_shape(tensor_handle h, std::int64_t* out_shape, std::int64_t out_len) {
    try {
        const auto& s = handle_ref(h)->shape;
        if (!out_shape || out_len != static_cast<std::int64_t>(s.size())) {
            throw std::runtime_error("out_shape must be non-null and out_len must equal ndim");
        }
        for (std::int64_t i = 0; i < out_len; ++i) {
            out_shape[i] = s[static_cast<std::size_t>(i)];
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

int tensor_read_data(tensor_handle h, double* out, std::int64_t out_len) {
    try {
        const auto& d = handle_ref(h)->data;
        if (!out || out_len != static_cast<std::int64_t>(d.size())) {
            throw std::runtime_error("out must be non-null and out_len must equal tensor.size");
        }
        std::memcpy(out, d.data(), static_cast<std::size_t>(out_len) * sizeof(double));
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

int tensor_read_grad(tensor_handle h, double* out, std::int64_t out_len) {
    try {
        const auto& g = handle_ref(h)->grad;
        if (g.empty()) {
            throw std::runtime_error("tensor has no grad (requires_grad=False)");
        }
        if (!out || out_len != static_cast<std::int64_t>(g.size())) {
            throw std::runtime_error("out must be non-null and out_len must equal tensor.size");
        }
        std::memcpy(out, g.data(), static_cast<std::size_t>(out_len) * sizeof(double));
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

int tensor_has_grad(tensor_handle h) {
    try {
        const auto& g = handle_ref(h)->grad;
        g_last_error.clear();
        return g.empty() ? 0 : 1;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return 0;
    }
}

int tensor_write_data(tensor_handle h, const double* in, std::int64_t in_len) {
    try {
        if (!in) {
            throw std::runtime_error("in must be non-null");
        }
        auto& d = handle_ref(h)->data;
        if (in_len != static_cast<std::int64_t>(d.size())) {
            throw std::runtime_error("in_len must equal tensor.size");
        }
        std::memcpy(d.data(), in, static_cast<std::size_t>(in_len) * sizeof(double));
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

void tensor_zero_grad(tensor_handle h) {
    try {
        handle_ref(h)->zero_grad();
        g_last_error.clear();
    } catch (const std::exception& e) {
        set_last_error(e.what());
    }
}

int tensor_backward(tensor_handle h) {
    try {
        handle_ref(h)->backward(nullptr, nullptr);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

int tensor_backward_scalar(tensor_handle h, double grad) {
    try {
        handle_ref(h)->backward(nullptr, &grad);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

int tensor_backward_with_tensor(tensor_handle h, tensor_handle grad_tensor) {
    try {
        handle_ref(h)->backward(handle_ref(grad_tensor), nullptr);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    }
}

tensor_handle tensor_detach(tensor_handle h) {
    try {
        auto out = handle_ref(h)->detach();
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_zeros(const std::int64_t* shape, std::int64_t shape_len, int requires_grad) {
    try {
        std::vector<std::int64_t> s(static_cast<std::size_t>(shape_len));
        for (std::int64_t i = 0; i < shape_len; ++i) {
            s[static_cast<std::size_t>(i)] = shape[i];
        }
        auto out = zeros(s, requires_grad != 0);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_ones(const std::int64_t* shape, std::int64_t shape_len, int requires_grad) {
    try {
        std::vector<std::int64_t> s(static_cast<std::size_t>(shape_len));
        for (std::int64_t i = 0; i < shape_len; ++i) {
            s[static_cast<std::size_t>(i)] = shape[i];
        }
        auto out = ones(s, requires_grad != 0);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_randn(const std::int64_t* shape, std::int64_t shape_len, double mean, double stdev, int requires_grad, const std::int64_t* seed_opt) {
    try {
        std::vector<std::int64_t> s(static_cast<std::size_t>(shape_len));
        for (std::int64_t i = 0; i < shape_len; ++i) {
            s[static_cast<std::size_t>(i)] = shape[i];
        }
        auto out = randn(s, mean, stdev, requires_grad != 0, seed_opt);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_reshape(tensor_handle h, const std::int64_t* new_shape, std::int64_t new_shape_len) {
    try {
        std::vector<std::int64_t> s(static_cast<std::size_t>(new_shape_len));
        for (std::int64_t i = 0; i < new_shape_len; ++i) {
            s[static_cast<std::size_t>(i)] = new_shape[i];
        }
        auto out = reshape(handle_ref(h), s);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_transpose(tensor_handle h) {
    try {
        auto out = transpose2d(handle_ref(h));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_sum(tensor_handle h) {
    try {
        auto out = sum_all(handle_ref(h));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_neg(tensor_handle h) {
    try {
        auto out = neg(handle_ref(h));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_add(tensor_handle a, tensor_handle b) {
    try {
        auto out = add(handle_ref(a), handle_ref(b));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_add_scalar(tensor_handle a, double c) {
    try {
        auto out = add_scalar(handle_ref(a), c);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_sub(tensor_handle a, tensor_handle b) {
    try {
        auto out = sub(handle_ref(a), handle_ref(b));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_sub_scalar(tensor_handle a, double c) {
    try {
        auto out = sub_scalar(handle_ref(a), c);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_mul(tensor_handle a, tensor_handle b) {
    try {
        auto out = mul(handle_ref(a), handle_ref(b));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_mul_scalar(tensor_handle a, double c) {
    try {
        auto out = mul_scalar(handle_ref(a), c);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_div(tensor_handle a, tensor_handle b) {
    try {
        auto out = div_tt(handle_ref(a), handle_ref(b));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_div_scalar(tensor_handle a, double c) {
    try {
        auto out = div_scalar(handle_ref(a), c);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_pow(tensor_handle a, double p) {
    try {
        auto out = pow_scalar(handle_ref(a), p);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_matmul(tensor_handle a, tensor_handle b) {
    try {
        auto out = matmul2d(handle_ref(a), handle_ref(b));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_relu(tensor_handle x) {
    try {
        auto out = relu(handle_ref(x));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_dropout(tensor_handle x, double p, const std::int64_t* seed_opt, int training) {
    try {
        auto out = dropout(handle_ref(x), p, seed_opt, training != 0);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_batchnorm1d(
    tensor_handle x,
    tensor_handle gamma,
    tensor_handle beta,
    tensor_handle running_mean,
    tensor_handle running_var,
    double momentum,
    double eps,
    int training
) {
    try {
        TensorPtr rm;
        TensorPtr rv;
        if (running_mean) {
            rm = handle_ref(running_mean);
        }
        if (running_var) {
            rv = handle_ref(running_var);
        }
        auto out = batchnorm1d(handle_ref(x), handle_ref(gamma), handle_ref(beta), rm, rv, momentum, eps, training != 0);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_batchnorm2d(
    tensor_handle x,
    tensor_handle gamma,
    tensor_handle beta,
    tensor_handle running_mean,
    tensor_handle running_var,
    double momentum,
    double eps,
    int training
) {
    try {
        TensorPtr rm;
        TensorPtr rv;
        if (running_mean) {
            rm = handle_ref(running_mean);
        }
        if (running_var) {
            rv = handle_ref(running_var);
        }
        auto out = batchnorm2d(handle_ref(x), handle_ref(gamma), handle_ref(beta), rm, rv, momentum, eps, training != 0);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_conv2d(tensor_handle x, tensor_handle w, tensor_handle b, std::int64_t stride, std::int64_t padding) {
    try {
        TensorPtr bias;
        if (b) {
            bias = handle_ref(b);
        }
        auto out = conv2d(handle_ref(x), handle_ref(w), bias, stride, padding);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_maxpool2d(tensor_handle x, std::int64_t kernel, const std::int64_t* stride_opt) {
    try {
        auto out = maxpool2d(handle_ref(x), kernel, stride_opt);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_mse_loss(tensor_handle pred, tensor_handle target) {
    try {
        auto out = mse_loss(handle_ref(pred), handle_ref(target));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_softmax2d(tensor_handle x) {
    try {
        auto out = softmax2d(handle_ref(x));
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_categorical_crossentropy(tensor_handle probs, const std::int64_t* targets, std::int64_t targets_len, double eps) {
    try {
        if (!targets || targets_len < 0) {
            throw std::runtime_error("targets must be non-null and targets_len >= 0");
        }
        std::vector<std::int64_t> t(static_cast<std::size_t>(targets_len));
        for (std::int64_t i = 0; i < targets_len; ++i) {
            t[static_cast<std::size_t>(i)] = targets[i];
        }
        auto out = categorical_crossentropy_probs(handle_ref(probs), t, eps);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

tensor_handle tensor_categorical_crossentropy_logits(tensor_handle logits, const std::int64_t* targets, std::int64_t targets_len) {
    try {
        if (!targets || targets_len < 0) {
            throw std::runtime_error("targets must be non-null and targets_len >= 0");
        }
        std::vector<std::int64_t> t(static_cast<std::size_t>(targets_len));
        for (std::int64_t i = 0; i < targets_len; ++i) {
            t[static_cast<std::size_t>(i)] = targets[i];
        }
        auto out = categorical_crossentropy_logits(handle_ref(logits), t);
        g_last_error.clear();
        return make_handle(std::move(out));
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    }
}

} // extern "C"