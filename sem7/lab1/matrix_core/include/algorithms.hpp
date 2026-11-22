#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include "matrix.hpp"
#include "thread_pool.hpp"

#include <array>
#include <string_view>

namespace matmul {

enum class Algorithm {
    naive,
    transposed,
    blocked,
    strassen,
    parallel,
    simd,
    combined,
    combined_no_strassen,
};

inline constexpr std::array algorithms{
    Algorithm::naive,
    Algorithm::transposed,
    Algorithm::blocked,
    Algorithm::strassen,
    Algorithm::parallel,
    Algorithm::simd,
    Algorithm::combined,
    Algorithm::combined_no_strassen,
};

std::string_view name(Algorithm algorithm);
Algorithm parse_algorithm(std::string_view text);
bool is_parallel(Algorithm algorithm);

struct Config {
    std::size_t block = 32;
    std::size_t cutoff = 64;
};

void multiply(Algorithm algorithm,
              const Matrix& a,
              const Matrix& b,
              Matrix& c,
              const Config& config,
              ThreadPool& pool);

}

#endif
