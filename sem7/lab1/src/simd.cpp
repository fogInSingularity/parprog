#include "simd.hpp"

#include <algorithm>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace matmul {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void SimdTile(ConstView a, ConstView bt, MutableView c,
              std::size_t row_begin, std::size_t row_end,
              std::size_t col_begin, std::size_t col_end, std::size_t block) {
    const hn::ScalableTag<float> d;
    const auto lanes = hn::Lanes(d);
    for (std::size_t i = row_begin; i < row_end; ++i) {
        for (std::size_t j = col_begin; j < col_end; ++j) c(i, j) = 0;
    }
    for (std::size_t kk = 0; kk < a.n; kk += block) {
        const auto end = std::min(kk + block, a.n);
        for (std::size_t i = row_begin; i < row_end; ++i) {
            for (std::size_t j = col_begin; j < col_end; ++j) {
                auto sum0 = hn::Zero(d);
                auto sum1 = hn::Zero(d);
                std::size_t k = kk;
                for (; end - k >= 2 * lanes; k += 2 * lanes) {
                    sum0 = hn::MulAdd(hn::LoadU(d, a.data + i * a.stride + k),
                                     hn::LoadU(d, bt.data + j * bt.stride + k), sum0);
                    sum1 = hn::MulAdd(hn::LoadU(d, a.data + i * a.stride + k + lanes),
                                     hn::LoadU(d, bt.data + j * bt.stride + k + lanes), sum1);
                }
                for (; end - k >= lanes; k += lanes)
                    sum0 = hn::MulAdd(hn::LoadU(d, a.data + i * a.stride + k),
                                     hn::LoadU(d, bt.data + j * bt.stride + k), sum0);
                float sum = hn::GetLane(hn::SumOfLanes(d, hn::Add(sum0, sum1)));
                for (; k < end; ++k) sum += a(i, k) * bt(j, k);
                c(i, j) += sum;
            }
        }
    }
}

const char* SimdTarget() { return hwy::TargetName(HWY_TARGET); }
std::size_t SimdLanes() { return hn::Lanes(hn::ScalableTag<float>()); }

}  // namespace HWY_NAMESPACE
}  // namespace matmul
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace matmul {
HWY_EXPORT(SimdTile);
HWY_EXPORT(SimdTarget);
HWY_EXPORT(SimdLanes);

void simd_tile(ConstView a, ConstView bt, MutableView c,
               std::size_t row_begin, std::size_t row_end,
               std::size_t col_begin, std::size_t col_end, std::size_t block) {
    HWY_DYNAMIC_DISPATCH(SimdTile)(a, bt, c, row_begin, row_end, col_begin, col_end, block);
}
const char* simd_target() { return HWY_DYNAMIC_DISPATCH(SimdTarget)(); }
std::size_t simd_lanes() { return HWY_DYNAMIC_DISPATCH(SimdLanes)(); }
}  // namespace matmul
#endif
