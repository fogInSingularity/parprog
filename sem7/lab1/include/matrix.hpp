#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <hwy/aligned_allocator.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

namespace matmul {

template <class T>
struct View {
    T* data;
    std::size_t n;
    std::size_t stride;

    View(T* ptr, std::size_t size, std::size_t pitch) : data(ptr), n(size), stride(pitch) {}
    template <class U> requires std::is_convertible_v<U*, T*>
    View(View<U> other) : data(other.data), n(other.n), stride(other.stride) {}

    T& operator()(std::size_t i, std::size_t j) const { return data[i * stride + j]; }
    View sub(std::size_t row, std::size_t col, std::size_t size) const {
        return {data + row * stride + col, size, stride};
    }
};
using ConstView = View<const float>;
using MutableView = View<float>;

enum class InputRange { positive, signed_unit };
std::string_view name(InputRange range);

class Matrix {
public:
    explicit Matrix(std::size_t n);
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    std::size_t size() const { return n_; }
    MutableView view() { return {data_.get(), n_, n_}; }
    ConstView view() const { return {data_.get(), n_, n_}; }
    float& operator()(std::size_t i, std::size_t j) { return view()(i, j); }
    float operator()(std::size_t i, std::size_t j) const { return view()(i, j); }
    void fill(float value);
    void randomize(std::uint32_t seed, InputRange range = InputRange::positive);

private:
    std::size_t n_;
    hwy::AlignedFreeUniquePtr<float[]> data_;
};

void transpose(ConstView in, MutableView out);
std::vector<double> reference_product(const Matrix& a, const Matrix& b);

struct Error {
    bool passed = true;
    std::size_t mismatches = 0;
    double max_absolute = 0;
    double max_scaled = 0;
    // Element with the largest error relative to its allowed tolerance.
    std::size_t row = 0;
    std::size_t col = 0;
    double actual = 0;
    double expected = 0;
    double absolute = 0;
    double allowed = 0;
};
// Absolute tolerance handles cancellation near zero; relative tolerance handles scale.
Error compare(const Matrix& actual, const std::vector<double>& expected,
              double absolute_tolerance = 2e-4, double relative_tolerance = 2e-4);
double checksum(const Matrix& matrix);

}  // namespace matmul

#endif  // MATRIX_HPP
