#include "simd.hpp"

#include <algorithm>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd_block.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace matmul {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

namespace {

template <class D>
HWY_INLINE void microkernel_4x2(D d,
                                ConstView a,
                                ConstView b,
                                MutableView c,
                                std::size_t row,
                                std::size_t col,
                                std::size_t k_begin,
                                std::size_t k_end) {
    const auto lanes = hn::Lanes(d);
    const auto* a0 = a.data + row * a.stride;
    const auto* a1 = a0 + a.stride;
    const auto* a2 = a1 + a.stride;
    const auto* a3 = a2 + a.stride;
    auto* c0 = c.data + row * c.stride + col;
    auto* c1 = c0 + c.stride;
    auto* c2 = c1 + c.stride;
    auto* c3 = c2 + c.stride;

    auto sum00 = hn::Zero(d);
    auto sum01 = hn::Zero(d);
    auto sum10 = hn::Zero(d);
    auto sum11 = hn::Zero(d);
    auto sum20 = hn::Zero(d);
    auto sum21 = hn::Zero(d);
    auto sum30 = hn::Zero(d);
    auto sum31 = hn::Zero(d);
    if (k_begin != 0) {
        sum00 = hn::LoadU(d, c0);
        sum01 = hn::LoadU(d, c0 + lanes);
        sum10 = hn::LoadU(d, c1);
        sum11 = hn::LoadU(d, c1 + lanes);
        sum20 = hn::LoadU(d, c2);
        sum21 = hn::LoadU(d, c2 + lanes);
        sum30 = hn::LoadU(d, c3);
        sum31 = hn::LoadU(d, c3 + lanes);
    }

    for (auto k = k_begin; k < k_end; ++k) {
        const auto* b_row = b.data + k * b.stride + col;
        const auto b0 = hn::LoadU(d, b_row);
        const auto b1 = hn::LoadU(d, b_row + lanes);
        const auto x0 = hn::Set(d, a0[k]);
        const auto x1 = hn::Set(d, a1[k]);
        const auto x2 = hn::Set(d, a2[k]);
        const auto x3 = hn::Set(d, a3[k]);
        sum00 = hn::MulAdd(x0, b0, sum00);
        sum01 = hn::MulAdd(x0, b1, sum01);
        sum10 = hn::MulAdd(x1, b0, sum10);
        sum11 = hn::MulAdd(x1, b1, sum11);
        sum20 = hn::MulAdd(x2, b0, sum20);
        sum21 = hn::MulAdd(x2, b1, sum21);
        sum30 = hn::MulAdd(x3, b0, sum30);
        sum31 = hn::MulAdd(x3, b1, sum31);
    }

    hn::StoreU(sum00, d, c0);
    hn::StoreU(sum01, d, c0 + lanes);
    hn::StoreU(sum10, d, c1);
    hn::StoreU(sum11, d, c1 + lanes);
    hn::StoreU(sum20, d, c2);
    hn::StoreU(sum21, d, c2 + lanes);
    hn::StoreU(sum30, d, c3);
    hn::StoreU(sum31, d, c3 + lanes);
}

template <class D>
HWY_INLINE void microkernel_row(D d,
                                ConstView a,
                                ConstView b,
                                MutableView c,
                                std::size_t row,
                                std::size_t col_begin,
                                std::size_t col_end,
                                std::size_t k_begin,
                                std::size_t k_end) {
    const auto lanes = hn::Lanes(d);
    const auto* a_row = a.data + row * a.stride;
    auto* c_row = c.data + row * c.stride;
    for (auto col = col_begin; col < col_end; col += lanes) {
        const auto count = std::min(lanes, col_end - col);
        auto sum = k_begin == 0 ? hn::Zero(d) : hn::LoadN(d, c_row + col, count);
        for (auto k = k_begin; k < k_end; ++k) {
            const auto* b_values = b.data + k * b.stride + col;
            sum = hn::MulAdd(hn::Set(d, a_row[k]), hn::LoadN(d, b_values, count), sum);
        }
        hn::StoreN(sum, d, c_row + col, count);
    }
}

}

void SimdBlockTile(ConstView a, ConstView b, MutableView c, Tile tile, std::size_t block) {
    const hn::ScalableTag<float> d;
    constexpr std::size_t rows_per_microkernel = 4;
    const auto cols_per_microkernel = 2 * hn::Lanes(d);
    for (std::size_t k = 0; k < a.n; k += block) {
        const auto k_end = k + std::min(block, a.n - k);
        auto row = tile.row_begin;
        for (; tile.row_end - row >= rows_per_microkernel; row += rows_per_microkernel) {
            auto col = tile.col_begin;
            for (; tile.col_end - col >= cols_per_microkernel; col += cols_per_microkernel) {
                microkernel_4x2(d, a, b, c, row, col, k, k_end);
            }
            if (col < tile.col_end) {
                for (auto tail = row; tail < row + rows_per_microkernel; ++tail) {
                    microkernel_row(d, a, b, c, tail, col, tile.col_end, k, k_end);
                }
            }
        }
        for (; row < tile.row_end; ++row) {
            microkernel_row(d, a, b, c, row, tile.col_begin, tile.col_end, k, k_end);
        }
    }
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace matmul {
HWY_EXPORT(SimdBlockTile);

void simd_block_tile(ConstView a, ConstView b, MutableView c, Tile tile, std::size_t block) {
    HWY_DYNAMIC_DISPATCH(SimdBlockTile)(a, b, c, tile, block);
}

}
#endif
