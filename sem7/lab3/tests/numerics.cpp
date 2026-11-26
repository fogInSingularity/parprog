#include "numerov.hpp"
#include "tridiagonal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace boundary;

void require(bool passed, const char* reason) {
    if (!passed) {
        throw std::runtime_error(reason);
    }
}

void check_tridiagonal() {
    CyclicReduction reduction;
    for (std::size_t n = 1; n <= 130; ++n) {
        std::vector<double> expected(n), solution(n);
        for (std::size_t i = 0; i < n; ++i) {
            expected[i] = std::sin(static_cast<double>(i) / 3.0);
        }
        std::vector<Row> rows(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double lower = i == 0 ? 0.0 : -0.75;
            const double upper = i + 1 == n ? 0.0 : -1.5;
            double rhs = 4 * expected[i];
            if (i != 0) {
                rhs += lower * expected[i - 1];
            }
            if (i + 1 < n) {
                rhs += upper * expected[i + 1];
            }
            rows[i] = {lower, 4.0, upper, rhs};
        }
        reduction.solve(rows, solution);
        for (std::size_t i = 0; i < n; ++i) {
            require(std::abs(solution[i] - expected[i]) < 1e-12, "Wrong cyclic reduction result");
        }
    }
}

void check_zero_reaction() {
    constexpr double left = 1.25;
    constexpr double right = 2.5;
    CyclicReduction reduction;
    for (bool full : {false, true}) {
        const Grid grid{0.0, 31, full};
        NewtonState state(grid, {0, grid.unknowns()});
        auto values = state.with_halos();
        values.front() = left;
        values.back() = right;
        for (std::size_t i = 0; i < grid.unknowns(); ++i) {
            values[i + 1] = 1.0 + 0.1 * static_cast<double>(i % 5);
        }
        newton(
            state,
            [](std::span<double>) {},
            [&reduction](std::span<Row> rows, std::span<double> correction) {
                reduction.solve(rows, correction);
            },
            [](std::array<double, 2> norms) { return norms; });
        for (std::size_t i = 0; i < grid.unknowns(); ++i) {
            const auto x = grid.coordinate(i);
            const auto exact = full ? left + (right - left) * (x + 10.0) / 20.0 : right;
            require(std::abs(state.values()[i] - exact) < 1e-12, "Wrong solution for y'' = 0");
        }
    }
}

std::vector<double> solve(std::size_t intervals) {
    Grid grid{100.0, intervals, false};
    NewtonState state(grid, {0, grid.unknowns()});
    state.reset();
    const auto result = newton(
        state,
        [](std::span<double>) {},
        thomas,
        [](std::array<double, 2> values) { return values; });
    require(result.residual < 1e-12, "Numerov residual is too large");
    return {state.values().begin(), state.values().end()};
}

double refinement_delta(std::span<const double> coarse, std::span<const double> fine) {
    double maximum = 0;
    for (std::size_t i = 0; i < coarse.size(); ++i) {
        maximum = std::max(maximum, std::abs(coarse[i] - fine[2 * i]));
    }
    return maximum;
}

void check_accuracy() {
    const auto coarse = solve(500);
    const auto medium = solve(1000);
    const auto fine = solve(2000);
    const auto ratio = refinement_delta(coarse, medium) / refinement_delta(medium, fine);
    require(ratio > 14 && ratio < 18, "Grid refinement does not exhibit fourth-order convergence");
}

}

int main() {
    try {
        check_tridiagonal();
        check_zero_reaction();
        check_accuracy();
        std::cout
            << "Tridiagonal, exact boundary-value, and fourth-order convergence tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
