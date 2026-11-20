#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "data.hpp"

namespace loops {
// Literal sequential version from the handout, including both passes of 3v.
void sequential(Data& data);
// Source column j in A owns destination column j+2 in B.
void fused_rows(Data& data, Slice rows) noexcept;
void fused(Data& data) noexcept;
void threaded(Data& data, std::size_t workers);
} // namespace loops

#endif // KERNELS_HPP
