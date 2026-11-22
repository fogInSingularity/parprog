#ifndef NUMEROV_HPP
#define NUMEROV_HPP

#include "tridiagonal.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace boundary {

struct Grid {
    double a;
    std::size_t intervals;
    bool full;

    double step() const noexcept;
    std::size_t unknowns() const noexcept;
    double coordinate(std::size_t index) const noexcept;
};

struct Slice {
    std::size_t begin;
    std::size_t count;
};

Slice partition(std::size_t size, std::size_t workers, std::size_t rank) noexcept;
double boundary_value() noexcept;

class NewtonState {
public:
    NewtonState(Grid grid, Slice slice);
    void reset();
    double assemble();
    double update();

    std::span<double> with_halos();
    std::span<const double> values() const;
    std::span<Row> system();
    std::span<double> correction();

private:
    Grid grid_;
    Slice slice_;
    std::vector<double> values_;
    std::vector<Row> rows_;
    std::vector<double> correction_;
};

struct Convergence {
    int iterations;
    double residual;
    double correction;
};

Convergence
newton(NewtonState& state, const auto& exchange, const auto& linear_solve, const auto& global_max) {
    constexpr double tolerance = 1e-12;
    constexpr int max_iterations = 30;
    double correction = 1;
    for (int iteration = 0; iteration <= max_iterations; ++iteration) {
        exchange(state.with_halos());
        const auto [residual, max_correction] =
            global_max(std::array{state.assemble(), correction});
        if (residual <= tolerance && max_correction <= tolerance) {
            return {iteration, residual, max_correction};
        }
        if (iteration == max_iterations) {
            break;
        }
        linear_solve(state.system(), state.correction());
        correction = state.update();
    }

    throw std::runtime_error("Newton did not converge");
}

} // namespace boundary

#endif // NUMEROV_HPP
