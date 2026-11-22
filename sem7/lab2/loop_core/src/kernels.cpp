#include "kernels.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace loops {

std::size_t chain_count(std::size_t rows, std::size_t cols) noexcept {
    return rows == 0 ? 0 : cols + (rows - 1) * std::min<std::size_t>(3, cols);
}

Chain chain_at(std::size_t index, std::size_t rows, std::size_t cols) noexcept {
    std::size_t row = 0;
    std::size_t col = index;
    if (index >= cols) {
        const auto left = std::min<std::size_t>(3, cols);
        row = 1 + (index - cols) / left;
        col = (index - cols) % left;
    }
    return {row, col, std::min(rows - row, 1 + (cols - 1 - col) / 3)};
}

void compute_chain(Chain chain, double first, double* output, std::size_t stride) noexcept {
    double value = first;
    output[0] = value;
    for (std::size_t index = 1; index < chain.count; ++index) {
        value = std::sin(2.0 * value);
        output[index * stride] = value;
    }
}

void sequential(Data& data) {
    const auto rows = data.rows;
    const auto cols = data.cols;
    auto& a = data.a;
    switch (data.task) {
    case Task::reference:
        for (auto& value : a) {
            value = std::sin(2.0 * value);
        }
        break;
    case Task::variant1v:
        for (std::size_t i = 1; i < rows; ++i) {
            for (std::size_t j = 3; j < cols; ++j) {
                a[i * cols + j] = std::sin(2.0 * a[(i - 1) * cols + j - 3]);
            }
        }
        break;
    case Task::variant2g:
        for (std::size_t i = 0; i + 3 < rows; ++i) {
            for (std::size_t j = 2; j < cols; ++j) {
                a[i * cols + j] = std::sin(0.1 * a[(i + 3) * cols + j - 2]);
            }
        }
        break;
    case Task::variant3d:
        for (auto& value : a) {
            value = std::sin(0.001 * value);
        }
        for (std::size_t i = 0; i + 3 < rows; ++i) {
            for (std::size_t j = 5; j < cols; ++j) {
                data.b[i * cols + j] = a[(i + 3) * cols + j - 5] * 3.0;
            }
        }
        break;
    }
}

namespace {

void snapshot_row(Data& data, std::size_t row) noexcept {
    const auto cols = data.cols;
    const auto* source = data.b.data() + (row + 3) * cols;
    auto* destination = data.a.data() + row * cols;
    for (std::size_t col = 2; col < cols; ++col) {
        destination[col] = std::sin(0.1 * source[col - 2]);
    }
}

void compute_rows(Data& data, Slice rows) noexcept {
    const auto cols = data.cols;
    const auto end = rows.begin + rows.count;
    if (data.task == Task::reference) {
        for (std::size_t index = rows.begin * cols; index < end * cols; ++index) {
            data.a[index] = std::sin(2.0 * data.a[index]);
        }
    } else if (data.task == Task::variant2g) {
        for (std::size_t row = rows.begin; row < end && row + 3 < data.rows; ++row) {
            snapshot_row(data, row);
        }
    } else if (data.task == Task::variant3d) {
        // MPI slices have up to three extra input rows for this forward shift.
        const auto input_rows = data.a.size() / cols;
        for (std::size_t row = rows.begin; row < end && row + 3 < input_rows; ++row) {
            const auto* source = data.a.data() + (row + 3) * cols;
            auto* destination = data.b.data() + row * cols;
            for (std::size_t col = 5; col < cols; ++col) {
                destination[col] = std::sin(0.001 * source[col - 5]) * 3.0;
            }
        }
    }
}

void compute_worker(Data& data, std::size_t workers, std::size_t rank) {
    if (data.task == Task::variant1v) {
        const auto count = chain_count(data.rows, data.cols);
        for (std::size_t index = rank; index < count; index += workers) {
            const auto chain = chain_at(index, data.rows, data.cols);
            auto* head = data.a.data() + chain.row * data.cols + chain.col;
            compute_chain(chain, *head, head, data.cols + 3);
        }
    } else {
        compute_rows(data, partition(data.rows, workers, rank));
    }
}

} // namespace

void fused(Data& data) noexcept {
    compute_worker(data, 1, 0);
}

void threaded(Data& data, std::size_t workers) {
    if (workers == 0) {
        throw std::invalid_argument("Thread count must be positive");
    }
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    for (std::size_t rank = 1; rank < workers; ++rank) {
        threads.emplace_back(compute_worker, std::ref(data), workers, rank);
    }
    compute_worker(data, workers, 0);
}

void openmp(Data& data, std::size_t workers) {
    if (data.task != Task::variant2g || workers == 0 ||
        workers > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("OpenMP requires task 2g and a positive int thread count");
    }
    const auto end = data.rows - std::min<std::size_t>(3, data.rows);
    const auto count = static_cast<int>(workers);
    omp_set_dynamic(0);
#pragma omp parallel for schedule(static) num_threads(count)
    for (std::size_t row = 0; row < end; ++row) {
        snapshot_row(data, row);
    }
}

} // namespace loops
