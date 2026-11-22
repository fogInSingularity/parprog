#include "tridiagonal.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace boundary {
namespace {

double quotient(double numerator, double pivot) {
    if (pivot == 0.0 || !std::isfinite(pivot)) {
        throw std::runtime_error("Invalid pivot in tridiagonal system");
    }
    return numerator / pivot;
}

void remove_left(Row& row, const Row& left) {
    const double factor = quotient(row.lower, left.diagonal);
    row.lower = -factor * left.lower;
    row.diagonal -= factor * left.upper;
    row.rhs -= factor * left.rhs;
}

void remove_right(Row& row, const Row& right) {
    const double factor = quotient(row.upper, right.diagonal);
    row.upper = -factor * right.upper;
    row.diagonal -= factor * right.lower;
    row.rhs -= factor * right.rhs;
}

} // namespace

void thomas(std::span<Row> rows, std::span<double> solution) {
    if (rows.empty()) {
        return;
    }
    for (std::size_t i = 1; i < rows.size(); ++i) {
        remove_left(rows[i], rows[i - 1]);
    }
    solution.back() = quotient(rows.back().rhs, rows.back().diagonal);
    for (std::size_t i = rows.size() - 1; i-- > 0;) {
        solution[i] = quotient(rows[i].rhs - rows[i].upper * solution[i + 1], rows[i].diagonal);
    }
}

void CyclicReduction::reduce(std::span<Row> rows) {
    eliminated_.clear();
    eliminated_.reserve(rows.size());
    active_.resize(rows.size());
    std::iota(active_.begin(), active_.end(), std::size_t{0});
    next_.reserve(rows.size());

    while (active_.size() > 2) {
        const auto last = active_.size() - 1;
        next_.clear();
        for (std::size_t j = 0; j < active_.size(); ++j) {
            const auto i = active_[j];
            if (j % 2 != 0 && j != last) {
                eliminated_.push_back({i, active_[j - 1], active_[j + 1]});
                continue;
            }
            // Eliminated equations are preserved for the reverse sweep.
            if (j != 0 && (j - 1) % 2 != 0) {
                remove_left(rows[i], rows[active_[j - 1]]);
            }
            if (j + 1 < last && (j + 1) % 2 != 0) {
                remove_right(rows[i], rows[active_[j + 1]]);
            }
            next_.push_back(i);
        }
        active_.swap(next_);
    }
}

void CyclicReduction::restore(std::span<const Row> rows, std::span<double> solution) const {
    for (auto entry = eliminated_.rbegin(); entry != eliminated_.rend(); ++entry) {
        const auto& row = rows[entry->index];
        solution[entry->index] = quotient(row.rhs - row.lower * solution[entry->left] -
                                              row.upper * solution[entry->right],
                                          row.diagonal);
    }
}

void CyclicReduction::solve(std::span<Row> rows, std::span<double> solution) {
    if (rows.empty()) {
        return;
    }
    reduce(rows);
    if (rows.size() == 1) {
        solution.front() = quotient(rows.front().rhs, rows.front().diagonal);
        return;
    }

    auto last = rows.back();
    remove_left(last, rows.front());
    solution.back() = quotient(last.rhs, last.diagonal);
    solution.front() =
        quotient(rows.front().rhs - rows.front().upper * solution.back(), rows.front().diagonal);
    restore(rows, solution);
}

} // namespace boundary
