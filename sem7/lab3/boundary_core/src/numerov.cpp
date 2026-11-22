#include "numerov.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace boundary {
namespace {

double reaction(double y) noexcept {
    return y * (y - 1.0) * (y + 1.0);
}

double derivative(double y) noexcept {
    return 3.0 * y * y - 1.0;
}

double max_abs(double current, double value) noexcept {
    return std::isfinite(value) ? std::max(current, std::abs(value))
                                : std::numeric_limits<double>::infinity();
}

}

double Grid::step() const noexcept {
    return (full ? 20.0 : 10.0) / static_cast<double>(intervals);
}

std::size_t Grid::unknowns() const noexcept {
    return full ? intervals - 1 : intervals;
}

double Grid::coordinate(std::size_t index) const noexcept {
    return full ? -10.0 + static_cast<double>(index + 1) * step()
                : static_cast<double>(index) * step();
}

Slice partition(std::size_t size, std::size_t workers, std::size_t rank) noexcept {
    const auto remainder = size % workers;
    const auto base = size / workers;
    return {rank * base + std::min(rank, remainder), base + (rank < remainder ? 1 : 0)};
}

double boundary_value() noexcept {
    return std::sqrt(2.0);
}

NewtonState::NewtonState(Grid grid, Slice slice)
    : grid_(grid), slice_(slice), values_(slice.count + 2), rows_(slice.count),
      correction_(slice.count) {
}

void NewtonState::reset() {
    const auto decay = std::sqrt(2.0 * grid_.a);
    const auto difference = (boundary_value() - 1.0) / (1.0 + std::exp(-20.0 * decay));
    for (std::size_t i = 0; i < slice_.count; ++i) {
        const auto x = grid_.coordinate(slice_.begin + i);
        values_[i + 1] =
            1.0 + difference * (std::exp(-decay * (x + 10.0)) + std::exp(-decay * (10.0 - x)));
    }
    values_.front() = boundary_value();
    values_.back() = boundary_value();
}

double NewtonState::assemble() {
    const auto h = grid_.step();
    const auto c = grid_.a * h * h / 12.0;
    double maximum = 0;
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        const auto global = slice_.begin + i;
        const auto y = values_[i + 1];
        const auto right = values_[i + 2];
        const bool symmetry = !grid_.full && global == 0;
        const auto left = symmetry ? right : values_[i];
        const auto residual =
            (y - left) + (y - right) + c * (reaction(left) + 10.0 * reaction(y) + reaction(right));
        rows_[i] = {
            global == 0 ? 0.0 : -1.0 + c * derivative(left),
            2.0 + 10.0 * c * derivative(y),
            global + 1 == grid_.unknowns()
                ? 0.0
                : (symmetry ? 2.0 : 1.0) * (-1.0 + c * derivative(right)),
            -residual,
        };
        maximum = max_abs(maximum, residual);
    }
    return maximum;
}

double NewtonState::update() {
    double maximum = 0;
    for (std::size_t i = 0; i < correction_.size(); ++i) {
        values_[i + 1] += correction_[i];
        maximum = max_abs(maximum, correction_[i]);
    }
    return maximum;
}

std::span<double> NewtonState::with_halos() {
    return values_;
}

std::span<const double> NewtonState::values() const {
    return std::span(values_).subspan(1, slice_.count);
}

std::span<Row> NewtonState::system() {
    return rows_;
}

std::span<double> NewtonState::correction() {
    return correction_;
}

}
