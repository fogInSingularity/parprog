#ifndef SIMD_HPP
#define SIMD_HPP

#include "matrix.hpp"

namespace matmul {
// B is already transposed. Only the specified output tile is written.
void simd_tile(ConstView a,
               ConstView bt,
               MutableView c,
               std::size_t row_begin,
               std::size_t row_end,
               std::size_t col_begin,
               std::size_t col_end,
               std::size_t block);

struct Tile {
    std::size_t row_begin;
    std::size_t row_end;
    std::size_t col_begin;
    std::size_t col_end;
};

void simd_block_tile(ConstView a, ConstView b, MutableView c, Tile tile, std::size_t block);

const char* simd_target();
std::size_t simd_lanes();
} // namespace matmul

#endif // SIMD_HPP
