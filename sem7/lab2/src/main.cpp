#include "benchmark.hpp"
#include "kernels.hpp"
#include "options.hpp"

#include <array>
#include <chrono>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>

namespace loops {
namespace {

Sample measure_local(Backend backend, std::size_t workers, Data& data, const Data& expected) {
    data.reset();
    const auto start = std::chrono::steady_clock::now();
    switch (backend) {
    case Backend::seq:
        sequential(data);
        break;
    case Backend::fused:
        fused(data);
        break;
    case Backend::threads:
        threaded(data, workers);
        break;
    case Backend::all:
        throw std::logic_error("Select one backend before running a measurement");
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    const auto output = data.output();
    return {elapsed.count(), output, compare(output, expected.output())};
}

int execute(const Options& options) {
    constexpr std::array<std::size_t, 1> serial_counts{1};
    auto csv = open_csv(options, true);
    auto* csv_output = csv.is_open() ? &csv : nullptr;
    std::size_t passed = 0;
    std::size_t failed = 0;

    for (auto task : options.tasks) {
        Data expected(task, options.rows, options.cols);
        sequential(expected);
        Data data(task, options.rows, options.cols);

        for (auto backend : {Backend::seq, Backend::fused, Backend::threads}) {
            if (options.backend != Backend::all && options.backend != backend) {
                continue;
            }
            std::span<const std::size_t> counts = serial_counts;
            if (backend == Backend::threads) {
                counts = options.workers;
            }
            for (auto workers : counts) {
                const auto measure = [backend, workers, &data, &expected] {
                    return measure_local(backend, workers, data, expected);
                };
                if (run_configuration(options, task, name(backend), workers, measure, csv_output)) {
                    ++passed;
                } else {
                    ++failed;
                }
            }
        }
    }

    std::cout << "Completed: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}

} // namespace
} // namespace loops

int main(int argc, char** argv) {
    try {
        const auto options = loops::parse_options(argc, argv, false);
        if (options.command == "help") {
            loops::help(std::cout, false);
            return 0;
        }
        return loops::execute(options);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
