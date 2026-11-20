#include "kernels.hpp"

#include <cmath>
#include <functional>
#include <thread>
#include <vector>

namespace loops {
void sequential(Data& data) {
    const double coefficient = data.task == Task::reference ? 2.0 : 0.01;
    for (std::size_t i = 0; i < data.rows; ++i) {
        for (std::size_t j = 0; j < data.cols; ++j) {
            data.a[i * data.cols + j] = std::sin(coefficient * data.a[i * data.cols + j]);
        }
    }
    if (data.task == Task::variant3v) {
        for (std::size_t i = 0; i < data.rows; ++i) {
            for (std::size_t j = 2; j < data.cols; ++j) {
                data.b[i * data.cols + j] = data.a[i * data.cols + j - 2] * 2.5;
            }
        }
    }
}

void fused_rows(Data& data, Slice rows) noexcept {
    for (std::size_t i = rows.begin; i < rows.begin + rows.count; ++i) {
        const auto row = i * data.cols;
        if (data.task == Task::reference) {
            for (std::size_t j = 0; j < data.cols; ++j) {
                data.a[row + j] = std::sin(2.0 * data.a[row + j]);
            }
        } else {
            for (std::size_t j = 0; j < data.cols; ++j) {
                const auto value = std::sin(0.01 * data.a[row + j]);
                data.a[row + j] = value;
                if (data.cols - j > 2) {
                    data.b[row + j + 2] = value * 2.5;
                }
            }
        }
    }
}

void fused(Data& data) noexcept {
    fused_rows(data, {0, data.rows});
}

void threaded(Data& data, std::size_t workers) {
    const auto first = partition(data.rows, workers, 0);
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    for (std::size_t rank = 1; rank < workers; ++rank) {
        const auto rows = partition(data.rows, workers, rank);
        threads.emplace_back(fused_rows, std::ref(data), rows);
    }
    fused_rows(data, first);
    // The jthreads join when the vector goes out of scope.
}
} // namespace loops
