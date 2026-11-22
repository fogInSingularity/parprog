#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>

namespace matmul {
namespace {
std::size_t elements(std::size_t n) {
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / sizeof(float) / n) {
        throw std::invalid_argument("Matrix size must be positive and fit in address space");
    }
    return n * n;
}
}

Matrix::Matrix(std::size_t n) : n_(n), data_(hwy::AllocateAligned<float>(elements(n))) {
    if (!data_) {
        throw std::bad_alloc();
    }
    fill(0);
}

void Matrix::fill(float value) {
    std::fill_n(data_.get(), n_ * n_, value);
}

std::string_view name(InputRange range) {
    switch (range) {
    case InputRange::positive:
        return "positive";
    case InputRange::signed_unit:
        return "signed";
    }
    throw std::invalid_argument("Unknown input range");
}

void Matrix::randomize(std::uint32_t seed, InputRange range) {
    if (range != InputRange::positive && range != InputRange::signed_unit) {
        throw std::invalid_argument("Unknown input range");
    }
    std::mt19937 generator(seed);
    for (std::size_t i = 0; i < n_ * n_; ++i) {
        if (range == InputRange::positive) {
            data_[i] = 1.0f + static_cast<float>(generator() % 1025) / 1024.0f;
        } else {
            data_[i] = static_cast<float>(static_cast<int>(generator() % 2049) - 1024) / 1024.0f;
        }
    }
}

void transpose(ConstView in, MutableView out) {
    constexpr std::size_t tile = 32;
    for (std::size_t ii = 0; ii < in.n; ii += tile) {
        for (std::size_t jj = 0; jj < in.n; jj += tile) {
            for (std::size_t i = ii; i < std::min(ii + tile, in.n); ++i) {
                for (std::size_t j = jj; j < std::min(jj + tile, in.n); ++j) {
                    out(j, i) = in(i, j);
                }
            }
        }
    }
}

std::vector<double> reference_product(const Matrix& a, const Matrix& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Reference: incompatible sizes");
    }
    const auto n = a.size();
    std::vector<double> result(n * n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                result[i * n + j] += static_cast<double>(a(i, k)) * b(k, j);
            }
        }
    }
    return result;
}

Error compare(const Matrix& actual,
              const std::vector<double>& expected,
              double absolute_tolerance,
              double relative_tolerance) {
    const auto n = actual.size();
    if (expected.size() != n * n) {
        throw std::invalid_argument("Comparison: incompatible sizes");
    }
    if (!std::isfinite(absolute_tolerance) || !std::isfinite(relative_tolerance) ||
        absolute_tolerance < 0 || relative_tolerance < 0) {
        throw std::invalid_argument("Comparison tolerances must be finite and nonnegative");
    }
    Error error;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double value = actual(i, j);
            const double ref = expected[i * n + j];
            const bool finite = std::isfinite(value) && std::isfinite(ref);
            const double diff =
                finite ? std::abs(value - ref) : std::numeric_limits<double>::infinity();
            const double limit = absolute_tolerance + relative_tolerance * std::abs(ref);
            double scaled = 0;
            if (!finite || (limit == 0 && diff != 0)) {
                scaled = std::numeric_limits<double>::infinity();
            } else if (limit > 0) {
                scaled = diff / limit;
            }
            if (!finite || diff > limit) {
                error.passed = false;
                ++error.mismatches;
            }
            error.max_absolute = std::max(error.max_absolute, diff);
            if (scaled > error.max_scaled) {
                error.max_scaled = scaled;
                error.row = i;
                error.col = j;
                error.actual = value;
                error.expected = ref;
                error.absolute = diff;
                error.allowed = limit;
            }
        }
    }
    return error;
}

double checksum(const Matrix& matrix) {
    double sum = 0;
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        for (std::size_t j = 0; j < matrix.size(); ++j) {
            sum += matrix(i, j);
        }
    }
    return sum;
}
}
