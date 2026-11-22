#include "benchmark.hpp"
#include "data.hpp"
#include "kernels.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace loops;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class F> void invalid(F action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid input accepted");
}

void partitions() {
    for (auto rows : {0U, 1U, 2U, 7U, 33U}) {
        for (auto workers : {1U, 2U, 8U, 32U}) {
            std::size_t next = 0;
            for (std::size_t rank = 0; rank < workers; ++rank) {
                const auto part = partition(rows, workers, rank);
                require(part.begin == next, "Partition has gaps or overlaps");
                next += part.count;
            }
            require(next == rows, "Partition misses rows");
        }
    }
    invalid([] { partition(5, 0, 0); });
    invalid([] { partition(5, 2, 2); });
}

void products() {
    for (auto workers : {1U, 2U, 8U, 32U}) {
        for (auto task : all_tasks) {
            for (auto rows : {0U, 1U, 2U, 3U, 4U, 7U, 17U}) {
                for (auto cols : {1U, 2U, 3U, 5U, 6U, 11U, 19U}) {
                    Data expected(task, rows, cols, 13);
                    Data actual(task, rows, cols, 13);
                    sequential(expected);
                    fused(actual);
                    require(compare(actual.output(), expected.output()).passed(),
                            "Fused result differs");
                    actual.reset();
                    if (task == Task::variant2g) {
                        openmp(actual, workers);
                    } else {
                        threaded(actual, workers);
                    }
                    require(compare(actual.output(), expected.output()).passed(),
                            "Parallel result differs");
                }
            }
        }
    }
    Data recurrence(Task::variant1v, 3, 9);
    sequential(recurrence);
    require(recurrence.a[6] == 6 && recurrence.a[2 * 9 + 2] == 22, "Wrong 1v boundaries");
    require(recurrence.a[2 * 9 + 7] == std::sin(2.0 * std::sin(2.0)), "Wrong 1v recurrence");

    Data snapshot(Task::variant2g, 8, 8);
    sequential(snapshot);
    require(snapshot.a[4] == std::sin(0.1 * 32.0), "2g must use the original source value");
    require(snapshot.a[1] == 1 && snapshot.a[7 * 8 + 7] == 77, "Wrong 2g boundaries");

    Data shifted(Task::variant3d, 7, 7);
    sequential(shifted);
    require(shifted.b[4] == 0 && shifted.b[4 * 7 + 6] == 0, "Wrong 3d boundaries");
    require(shifted.b[2 * 7 + 6] == std::sin(0.001 * 51.0) * 3.0, "Wrong 3d shift");

    invalid([] { Data data(Task::reference, 1, 0); });
    invalid([] { checked_elements(std::numeric_limits<std::size_t>::max(), 2); });
    invalid([&] { threaded(recurrence, 0); });
}

void validation() {
    const std::vector<double> reference{1, 2, 3, 4};
    auto values = reference;
    values[2] = 30;
    auto result = compare(values, reference);
    require(result.mismatches == 1 && result.index == 2 && result.expected == 3 &&
                result.actual == 30,
            "Incorrect mismatch diagnostics");
    values = reference;
    values[1] = std::numeric_limits<double>::quiet_NaN();
    require(compare(values, reference).mismatches == 1, "NaN accepted");
    const std::vector<double> zero{0.0}, negative_zero{-0.0};
    require(!compare(zero, negative_zero).passed(), "Bitwise validation loses signed zero");
}

void failure_is_reported() {
    Options options;
    options.command = "bench";
    options.rows = 1;
    options.cols = 2;
    options.warmups = 0;
    options.repeats = 3;
    int invokes = 0;
    const auto run = [&] {
        Validation result;
        if (++invokes == 3) {
            result.mismatches = 1; // Fail after one successful measured repeat.
        }
        return Sample{0.01, {}, result};
    };
    std::ostringstream csv;
    require(!run_configuration(options, Task::variant3d, "test", 1, run, &csv, false),
            "Failure ignored");
    require(csv.str().empty(), "Partial invalid measurement group was saved");
    require(run_configuration(options, Task::variant3d, "test", 1, run, &csv, false),
            "Later group did not run");
    require(!csv.str().empty(), "Valid measurements lost");
}
} // namespace

int main() {
    try {
        partitions();
        products();
        validation();
        failure_is_reported();
        std::cout << "Kernels, row partitions, repeated thread dispatch, and failure continuation "
                     "passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
