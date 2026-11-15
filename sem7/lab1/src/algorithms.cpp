#include "algorithms.hpp"
#include "simd.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matmul {
namespace {

void naive(ConstView a, ConstView b, MutableView c) {
    for (std::size_t i = 0; i < a.n; ++i)
        for (std::size_t j = 0; j < a.n; ++j) {
            float sum = 0;
            for (std::size_t k = 0; k < a.n; ++k) sum += a(i, k) * b(k, j);
            c(i, j) = sum;
        }
}

void transposed(ConstView a, ConstView b, MutableView c) {
    Matrix bt(b.n);
    transpose(b, bt.view());
    const auto t = bt.view();
    for (std::size_t i = 0; i < a.n; ++i)
        for (std::size_t j = 0; j < a.n; ++j) {
            float sum = 0;
            for (std::size_t k = 0; k < a.n; ++k) sum += a(i, k) * t(j, k);
            c(i, j) = sum;
        }
}

// Each task owns an entire output tile for all k, so no output reductions/locks.
void blocked_tile(ConstView a, ConstView b, MutableView c, std::size_t block,
                  std::size_t ib, std::size_t ie, std::size_t jb, std::size_t je) {
    for (std::size_t i = ib; i < ie; ++i)
        for (std::size_t j = jb; j < je; ++j) c(i, j) = 0;
    for (std::size_t kk = 0; kk < a.n; kk += block)
        for (std::size_t i = ib; i < ie; ++i)
            for (std::size_t k = kk; k < std::min(kk + block, a.n); ++k) {
                const float value = a(i, k);
                for (std::size_t j = jb; j < je; ++j) c(i, j) += value * b(k, j);
            }
}

void tiled_product(ConstView a, ConstView b, MutableView c, std::size_t block,
                   bool vectorized, ThreadPool* pool) {
    const auto tiles = 1 + (a.n - 1) / block;
    // Allocate/transposition only for the SIMD version, inside the measured call.
    // This matrix is shared read-only by tile tasks.
    auto bt = vectorized ? std::make_unique<Matrix>(a.n) : nullptr;
    if (bt) transpose(b, bt->view());
    const auto task = [&](std::size_t index) {
        const auto ib = (index / tiles) * block;
        const auto jb = (index % tiles) * block;
        const auto ie = ib + std::min(block, a.n - ib);
        const auto je = jb + std::min(block, a.n - jb);
        if (vectorized) simd_tile(a, bt->view(), c, ib, ie, jb, je, block);
        else blocked_tile(a, b, c, block, ib, ie, jb, je);
    };
    if (pool) pool->parallel_for(tiles * tiles, task);
    else for (std::size_t tile = 0; tile < tiles * tiles; ++tile) task(tile);
}

void linear_combination(ConstView a, ConstView b, MutableView out, float sign) {
    for (std::size_t i = 0; i < out.n; ++i)
        for (std::size_t j = 0; j < out.n; ++j) out(i, j) = a(i, j) + sign * b(i, j);
}

void copy(ConstView in, MutableView out) {
    for (std::size_t i = 0; i < out.n; ++i)
        for (std::size_t j = 0; j < out.n; ++j) out(i, j) = in(i, j);
}

void strassen_recursive(ConstView a, ConstView b, MutableView c,
                        const Config& config, bool vectorized, ThreadPool* pool) {
    if (a.n <= config.cutoff) {
        tiled_product(a, b, c, config.block, vectorized, pool);
        return;
    }
    const auto h = a.n / 2;
    const std::array aa{a.sub(0, 0, h), a.sub(0, h, h), a.sub(h, 0, h), a.sub(h, h, h)};
    const std::array bb{b.sub(0, 0, h), b.sub(0, h, h), b.sub(h, 0, h), b.sub(h, h, h)};
    std::array products{Matrix(h), Matrix(h), Matrix(h), Matrix(h), Matrix(h), Matrix(h), Matrix(h)};
    const auto product = [&](std::size_t index) {
        Matrix x(h), y(h);
        switch (index) {
        case 0: // M1 = (A11+A22)(B11+B22)
            linear_combination(aa[0], aa[3], x.view(), 1);
            linear_combination(bb[0], bb[3], y.view(), 1);
            break;
        case 1: // M2 = (A21+A22)B11
            linear_combination(aa[2], aa[3], x.view(), 1); copy(bb[0], y.view());
            break;
        case 2: // M3 = A11(B12-B22)
            copy(aa[0], x.view()); linear_combination(bb[1], bb[3], y.view(), -1);
            break;
        case 3: // M4 = A22(B21-B11)
            copy(aa[3], x.view()); linear_combination(bb[2], bb[0], y.view(), -1);
            break;
        case 4: // M5 = (A11+A12)B22
            linear_combination(aa[0], aa[1], x.view(), 1); copy(bb[3], y.view());
            break;
        case 5: // M6 = (A21-A11)(B11+B12)
            linear_combination(aa[2], aa[0], x.view(), -1);
            linear_combination(bb[0], bb[1], y.view(), 1);
            break;
        case 6: // M7 = (A12-A22)(B21+B22)
            linear_combination(aa[1], aa[3], x.view(), -1);
            linear_combination(bb[2], bb[3], y.view(), 1);
            break;
        }
        // Parallelism only at the top level: never wait on this pool from workers.
        strassen_recursive(x.view(), y.view(), products[index].view(), config, vectorized, nullptr);
    };
    if (pool) pool->parallel_for(products.size(), product);
    else for (std::size_t index = 0; index < products.size(); ++index) product(index);

    for (std::size_t i = 0; i < h; ++i)
        for (std::size_t j = 0; j < h; ++j) {
            const float m1 = products[0](i, j), m2 = products[1](i, j), m3 = products[2](i, j);
            const float m4 = products[3](i, j), m5 = products[4](i, j), m6 = products[5](i, j);
            const float m7 = products[6](i, j);
            c(i, j) = m1 + m4 - m5 + m7;
            c(i, j + h) = m3 + m5;
            c(i + h, j) = m2 + m4;
            c(i + h, j + h) = m1 - m2 + m3 + m6;
        }
}

void strassen(ConstView a, ConstView b, MutableView c,
               const Config& config, bool vectorized, ThreadPool* pool) {
    // No padding for leaf calls. Larger non-power-of-two matrices are zero padded.
    if (a.n <= config.cutoff) {
        tiled_product(a, b, c, config.block, vectorized, pool);
        return;
    }
    std::size_t padded = 1;
    while (padded < a.n) {
        if (padded > std::numeric_limits<std::size_t>::max() / 2)
            throw std::length_error("Strassen padding overflows size_t");
        padded *= 2;
    }
    if (padded == a.n) {
        strassen_recursive(a, b, c, config, vectorized, pool);
        return;
    }
    Matrix ap(padded), bp(padded), cp(padded);
    for (std::size_t i = 0; i < a.n; ++i)
        for (std::size_t j = 0; j < a.n; ++j) {
            ap(i, j) = a(i, j); bp(i, j) = b(i, j);
        }
    strassen_recursive(ap.view(), bp.view(), cp.view(), config, vectorized, pool);
    copy(cp.view().sub(0, 0, a.n), c);
}
}  // namespace

std::string_view name(Algorithm algorithm) {
    switch (algorithm) {
    case Algorithm::naive: return "naive";
    case Algorithm::transposed: return "transposed";
    case Algorithm::blocked: return "blocked";
    case Algorithm::strassen: return "strassen";
    case Algorithm::parallel: return "parallel";
    case Algorithm::simd: return "simd";
    case Algorithm::combined: return "combined";
    }
    throw std::invalid_argument("Unknown algorithm");
}
Algorithm parse_algorithm(std::string_view text) {
    for (const auto algorithm : algorithms) if (name(algorithm) == text) return algorithm;
    throw std::invalid_argument("Unknown algorithm: " + std::string(text));
}
bool is_parallel(Algorithm algorithm) {
    return algorithm == Algorithm::parallel || algorithm == Algorithm::combined;
}

void multiply(Algorithm algorithm, const Matrix& a, const Matrix& b, Matrix& c,
              const Config& config, ThreadPool& pool) {
    if (a.size() != b.size() || a.size() != c.size())
        throw std::invalid_argument("Matrix sizes must match");
    if (&c == &a || &c == &b) throw std::invalid_argument("Output must not alias inputs");
    if (config.block == 0 || config.cutoff == 0)
        throw std::invalid_argument("Block size and Strassen cutoff must be positive");
    // Clamp block to n to keep additions bounded even with huge user parameters.
    const Config bounded{std::min(config.block, a.size()), config.cutoff};
    switch (algorithm) {
    case Algorithm::naive: naive(a.view(), b.view(), c.view()); break;
    case Algorithm::transposed: transposed(a.view(), b.view(), c.view()); break;
    case Algorithm::blocked:
        tiled_product(a.view(), b.view(), c.view(), bounded.block, false, nullptr); break;
    case Algorithm::strassen: strassen(a.view(), b.view(), c.view(), bounded, false, nullptr); break;
    case Algorithm::parallel:
        tiled_product(a.view(), b.view(), c.view(), bounded.block, false, &pool); break;
    case Algorithm::simd:
        // A full k range here isolates transpose+SIMD from cache blocking.
        { Matrix bt(b.size()); transpose(b.view(), bt.view());
          simd_tile(a.view(), bt.view(), c.view(), 0, a.size(), 0, a.size(), a.size()); }
        break;
    case Algorithm::combined: strassen(a.view(), b.view(), c.view(), bounded, true, &pool); break;
    default: throw std::invalid_argument("Unknown algorithm");
    }
}
}  // namespace matmul
