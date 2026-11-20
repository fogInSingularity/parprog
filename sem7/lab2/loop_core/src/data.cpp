#include "data.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>

namespace loops {

std::string_view name(Task task) {
    switch (task) {
    case Task::reference:
        return "reference";
    case Task::variant3v:
        return "3v";
    }
    throw std::invalid_argument("Unknown task");
}

Task parse_task(std::string_view text) {
    if (text == "reference") {
        return Task::reference;
    }
    if (text == "3v" || text == "3в") {
        return Task::variant3v;
    }
    throw std::invalid_argument("Task must be reference, 3v, or all");
}

std::size_t checked_elements(std::size_t rows, std::size_t cols) {
    if (cols == 0 || rows > std::numeric_limits<std::size_t>::max() / sizeof(double) / cols) {
        throw std::invalid_argument(
            "Array dimensions must fit in address space; columns must be positive");
    }
    return rows * cols;
}

Slice partition(std::size_t rows, std::size_t workers, std::size_t rank) {
    if (workers == 0 || rank >= workers) {
        throw std::invalid_argument("Invalid row partition");
    }
    const auto base = rows / workers;
    const auto remainder = rows % workers;
    return {base * rank + std::min(rank, remainder), base + (rank < remainder ? 1 : 0)};
}

Data::Data(Task selected, std::size_t height, std::size_t width, std::size_t offset)
    : task(selected), rows(height), cols(width), row_offset(offset),
      a(checked_elements(height, width)), b(selected == Task::variant3v ? a.size() : 0) {
    if (offset > std::numeric_limits<std::size_t>::max() - height) {
        throw std::invalid_argument("Global row offset overflows size_t");
    }
    reset();
}

void Data::reset() {
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            a[i * cols + j] = 10.0 * static_cast<double>(row_offset + i) + static_cast<double>(j);
        }
    }
    std::fill(b.begin(), b.end(), 0.0);
}

std::span<const double> Data::output() const {
    return task == Task::reference ? std::span<const double>(a) : std::span<const double>(b);
}

Validation compare(std::span<const double> actual, std::span<const double> expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Comparison size mismatch");
    }
    Validation result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
            std::bit_cast<std::uint64_t>(actual[index]) !=
                std::bit_cast<std::uint64_t>(expected[index])) {
            if (result.mismatches == 0) {
                result.index = index;
                result.actual = actual[index];
                result.expected = expected[index];
            }
            ++result.mismatches;
            const auto difference = std::abs(actual[index] - expected[index]);
            result.max_absolute = std::max(
                result.max_absolute,
                std::isfinite(difference) ? difference : std::numeric_limits<double>::infinity());
        }
    }
    return result;
}

double checksum(std::span<const double> values) {
    double sum = 0;
    for (double value : values) {
        sum += value;
    }
    return sum;
}

void write_matrix(const std::filesystem::path& path,
                  std::span<const double> values,
                  std::size_t rows,
                  std::size_t cols) {
    if (values.size() != checked_elements(rows, cols)) {
        throw std::invalid_argument("Output size mismatch");
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open matrix file: " + path.string());
    }
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            output << values[i * cols + j] << ' ';
        }
        output << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Failed to write matrix file: " + path.string());
    }
}
} // namespace loops
