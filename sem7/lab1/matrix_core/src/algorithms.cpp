#include "algorithms.hpp"
#include "simd.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace matmul {
namespace {

void naive(ConstView a, ConstView b, MutableView c) {
    for (std::size_t i = 0; i < a.n; ++i) {
        for (std::size_t j = 0; j < a.n; ++j) {
            float sum = 0;
            for (std::size_t k = 0; k < a.n; ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
}

void transposed(ConstView a, ConstView b, MutableView c) {
    Matrix bt(b.n);
    transpose(b, bt.view());
    const auto t = bt.view();
    for (std::size_t i = 0; i < a.n; ++i) {
        for (std::size_t j = 0; j < a.n; ++j) {
            float sum = 0;
            for (std::size_t k = 0; k < a.n; ++k) {
                sum += a(i, k) * t(j, k);
            }
            c(i, j) = sum;
        }
    }
}

// Each task owns an entire output tile for all k, so no output reductions/locks.
void blocked_tile(ConstView a,
                  ConstView b,
                  MutableView c,
                  std::size_t block,
                  std::size_t row_begin,
                  std::size_t row_end,
                  std::size_t col_begin,
                  std::size_t col_end) {
    for (auto i = row_begin; i < row_end; ++i) {
        for (auto j = col_begin; j < col_end; ++j) {
            c(i, j) = 0;
        }
    }
    for (std::size_t k_begin = 0; k_begin < a.n; k_begin += block) {
        const auto k_end = std::min(k_begin + block, a.n);
        for (auto i = row_begin; i < row_end; ++i) {
            for (auto k = k_begin; k < k_end; ++k) {
                const float value = a(i, k);
                for (auto j = col_begin; j < col_end; ++j) {
                    c(i, j) += value * b(k, j);
                }
            }
        }
    }
}

void multiply_tile(ConstView a,
                   ConstView b,
                   MutableView c,
                   std::size_t block,
                   std::size_t tiles_per_side,
                   bool vectorized,
                   std::size_t index) {
    const auto row_begin = (index / tiles_per_side) * block;
    const auto col_begin = (index % tiles_per_side) * block;
    const auto row_end = row_begin + std::min(block, a.n - row_begin);
    const auto col_end = col_begin + std::min(block, a.n - col_begin);
    if (vectorized) {
        simd_block_tile(a, b, c, {row_begin, row_end, col_begin, col_end}, block);
    } else {
        blocked_tile(a, b, c, block, row_begin, row_end, col_begin, col_end);
    }
}

void tiled_product(
    ConstView a, ConstView b, MutableView c, std::size_t block, bool vectorized, ThreadPool* pool) {
    const auto tiles_per_side = 1 + (a.n - 1) / block;
    const auto tile_count = tiles_per_side * tiles_per_side;
    const auto task = [&a, &b, &c, block, tiles_per_side, vectorized](std::size_t index) {
        multiply_tile(a, b, c, block, tiles_per_side, vectorized, index);
    };
    if (pool) {
        pool->parallel_for(tile_count, task);
    } else {
        for (std::size_t tile = 0; tile < tile_count; ++tile) {
            task(tile);
        }
    }
}

void linear_combination(ConstView a, ConstView b, MutableView out, float sign) {
    for (std::size_t i = 0; i < out.n; ++i) {
        for (std::size_t j = 0; j < out.n; ++j) {
            out(i, j) = a(i, j) + sign * b(i, j);
        }
    }
}

void copy(ConstView in, MutableView out) {
    for (std::size_t i = 0; i < out.n; ++i) {
        for (std::size_t j = 0; j < out.n; ++j) {
            out(i, j) = in(i, j);
        }
    }
}

void strassen_factors(std::size_t index,
                      const std::array<ConstView, 4>& a,
                      const std::array<ConstView, 4>& b,
                      MutableView left,
                      MutableView right) {
    const auto& [a11, a12, a21, a22] = a;
    const auto& [b11, b12, b21, b22] = b;

    switch (index) {
    case 0: // M1 = (A11+A22)(B11+B22)
        linear_combination(a11, a22, left, 1);
        linear_combination(b11, b22, right, 1);
        break;
    case 1: // M2 = (A21+A22)B11
        linear_combination(a21, a22, left, 1);
        copy(b11, right);
        break;
    case 2: // M3 = A11(B12-B22)
        copy(a11, left);
        linear_combination(b12, b22, right, -1);
        break;
    case 3: // M4 = A22(B21-B11)
        copy(a22, left);
        linear_combination(b21, b11, right, -1);
        break;
    case 4: // M5 = (A11+A12)B22
        linear_combination(a11, a12, left, 1);
        copy(b22, right);
        break;
    case 5: // M6 = (A21-A11)(B11+B12)
        linear_combination(a21, a11, left, -1);
        linear_combination(b11, b12, right, 1);
        break;
    case 6: // M7 = (A12-A22)(B21+B22)
        linear_combination(a12, a22, left, -1);
        linear_combination(b21, b22, right, 1);
        break;
    }
}

void strassen_recursive(ConstView a,
                        ConstView b,
                        MutableView c,
                        const Config& config,
                        bool vectorized,
                        ThreadPool* pool);

void strassen_product(std::size_t index,
                      const std::array<ConstView, 4>& a,
                      const std::array<ConstView, 4>& b,
                      MutableView result,
                      const Config& config,
                      bool vectorized) {
    Matrix left(result.n);
    Matrix right(result.n);
    strassen_factors(index, a, b, left.view(), right.view());
    // Only the top level uses the pool. Workers never wait for other pool jobs.
    strassen_recursive(left.view(), right.view(), result, config, vectorized, nullptr);
}

void strassen_recursive(ConstView a,
                        ConstView b,
                        MutableView c,
                        const Config& config,
                        bool vectorized,
                        ThreadPool* pool) {
    if (a.n <= config.cutoff) {
        tiled_product(a, b, c, config.block, vectorized, pool);
        return;
    }
    const auto h = a.n / 2;
    const std::array a_quadrants{
        a.sub(0, 0, h),
        a.sub(0, h, h),
        a.sub(h, 0, h),
        a.sub(h, h, h),
    };
    const std::array b_quadrants{
        b.sub(0, 0, h),
        b.sub(0, h, h),
        b.sub(h, 0, h),
        b.sub(h, h, h),
    };
    std::array products{
        Matrix(h),
        Matrix(h),
        Matrix(h),
        Matrix(h),
        Matrix(h),
        Matrix(h),
        Matrix(h),
    };
    const auto product =
        [&a_quadrants, &b_quadrants, &products, &config, vectorized](std::size_t index) {
            strassen_product(
                index, a_quadrants, b_quadrants, products[index].view(), config, vectorized);
        };
    if (pool) {
        pool->parallel_for(products.size(), product);
    } else {
        for (std::size_t index = 0; index < products.size(); ++index) {
            product(index);
        }
    }

    for (std::size_t i = 0; i < h; ++i) {
        for (std::size_t j = 0; j < h; ++j) {
            const float m1 = products[0](i, j);
            const float m2 = products[1](i, j);
            const float m3 = products[2](i, j);
            const float m4 = products[3](i, j);
            const float m5 = products[4](i, j);
            const float m6 = products[5](i, j);
            const float m7 = products[6](i, j);
            c(i, j) = m1 + m4 - m5 + m7;
            c(i, j + h) = m3 + m5;
            c(i + h, j) = m2 + m4;
            c(i + h, j + h) = m1 - m2 + m3 + m6;
        }
    }
}

void strassen(ConstView a,
              ConstView b,
              MutableView c,
              const Config& config,
              bool vectorized,
              ThreadPool* pool) {
    // No padding for leaf calls. Larger non-power-of-two matrices are zero padded.
    if (a.n <= config.cutoff) {
        tiled_product(a, b, c, config.block, vectorized, pool);
        return;
    }
    std::size_t padded = 1;
    while (padded < a.n) {
        if (padded > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::length_error("Strassen padding overflows size_t");
        }
        padded *= 2;
    }
    if (padded == a.n) {
        strassen_recursive(a, b, c, config, vectorized, pool);
        return;
    }
    Matrix ap(padded);
    Matrix bp(padded);
    Matrix cp(padded);
    for (std::size_t i = 0; i < a.n; ++i) {
        for (std::size_t j = 0; j < a.n; ++j) {
            ap(i, j) = a(i, j);
            bp(i, j) = b(i, j);
        }
    }
    strassen_recursive(ap.view(), bp.view(), cp.view(), config, vectorized, pool);
    copy(cp.view().sub(0, 0, a.n), c);
}
} // namespace

std::string_view name(Algorithm algorithm) {
    switch (algorithm) {
    case Algorithm::naive:
        return "naive";
    case Algorithm::transposed:
        return "transposed";
    case Algorithm::blocked:
        return "blocked";
    case Algorithm::strassen:
        return "strassen";
    case Algorithm::parallel:
        return "parallel";
    case Algorithm::simd:
        return "simd";
    case Algorithm::combined:
        return "combined";
    case Algorithm::combined_no_strassen:
        return "combined-no-strassen";
    }
    throw std::invalid_argument("Unknown algorithm");
}

Algorithm parse_algorithm(std::string_view text) {
    for (const auto algorithm : algorithms) {
        if (name(algorithm) == text) {
            return algorithm;
        }
    }
    throw std::invalid_argument("Unknown algorithm: " + std::string(text));
}

bool is_parallel(Algorithm algorithm) {
    return algorithm == Algorithm::parallel || algorithm == Algorithm::combined ||
           algorithm == Algorithm::combined_no_strassen;
}

void multiply(Algorithm algorithm,
              const Matrix& a,
              const Matrix& b,
              Matrix& c,
              const Config& config,
              ThreadPool& pool) {
    if (a.size() != b.size() || a.size() != c.size()) {
        throw std::invalid_argument("Matrix sizes must match");
    }
    if (&c == &a || &c == &b) {
        throw std::invalid_argument("Output must not alias inputs");
    }
    if (config.block == 0 || config.cutoff == 0) {
        throw std::invalid_argument("Block size and Strassen cutoff must be positive");
    }
    // Clamp block to n to keep additions bounded even with huge user parameters.
    const Config bounded{std::min(config.block, a.size()), config.cutoff};
    switch (algorithm) {
    case Algorithm::naive:
        naive(a.view(), b.view(), c.view());
        break;
    case Algorithm::transposed:
        transposed(a.view(), b.view(), c.view());
        break;
    case Algorithm::blocked:
        tiled_product(a.view(), b.view(), c.view(), bounded.block, false, nullptr);
        break;
    case Algorithm::strassen:
        strassen(a.view(), b.view(), c.view(), bounded, false, nullptr);
        break;
    case Algorithm::parallel:
        tiled_product(a.view(), b.view(), c.view(), bounded.block, false, &pool);
        break;
    case Algorithm::simd: {
        // A full k range here isolates transpose+SIMD from cache blocking.
        Matrix bt(b.size());
        transpose(b.view(), bt.view());
        simd_tile(a.view(), bt.view(), c.view(), 0, a.size(), 0, a.size(), a.size());
        break;
    }
    case Algorithm::combined:
        strassen(a.view(), b.view(), c.view(), bounded, true, &pool);
        break;
    case Algorithm::combined_no_strassen:
        tiled_product(a.view(), b.view(), c.view(), bounded.block, true, &pool);
        break;
    default:
        throw std::invalid_argument("Unknown algorithm");
    }
}
} // namespace matmul
