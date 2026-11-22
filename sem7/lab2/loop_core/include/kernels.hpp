#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "data.hpp"

namespace loops {
// A chain in 1v advances by one row and three columns.
struct Chain {
    std::size_t row;
    std::size_t col;
    std::size_t count;
};

std::size_t chain_count(std::size_t rows, std::size_t cols) noexcept;
Chain chain_at(std::size_t index, std::size_t rows, std::size_t cols) noexcept;
void compute_chain(Chain chain, double first, double* output, std::size_t stride) noexcept;

void sequential(Data& data);
void fused(Data& data) noexcept;
void threaded(Data& data, std::size_t workers);
void openmp(Data& data, std::size_t workers);
} // namespace loops

#endif // KERNELS_HPP
