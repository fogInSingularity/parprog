#include "benchmark.hpp"

#include "simd.hpp"

#include <hwy/base.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace matmul {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

void print_info(std::ostream& out) {
    out << "C++20; float32; standard C++ threads\n"
        << "compiler=" << __VERSION__ << '\n'
        << "highway=" << HWY_MAJOR << '.' << HWY_MINOR << '.' << HWY_PATCH << '\n'
        << "simd_target=" << simd_target() << "; float_lanes=" << simd_lanes() << '\n'
        << "hardware_threads=" << std::thread::hardware_concurrency() << '\n';
#ifdef NDEBUG
    out << "assertions=disabled (use a Release build for performance)\n";
#else
    out << "assertions=enabled (do not use a Debug build for performance conclusions)\n";
#endif
    std::ifstream cpu("/proc/cpuinfo");
    for (std::string line; std::getline(cpu, line);) {
        if (line.starts_with("model name")) {
            out << line << '\n';
            break;
        }
    }
}

namespace {

void metadata(const Options& options) {
    std::ofstream out(options.output.string() + ".meta.txt");
    if (!out) {
        throw std::runtime_error("Cannot write benchmark metadata");
    }
    print_info(out);
    out << "block=" << options.config.block << "; cutoff=" << options.config.cutoff
        << "; repeats=" << options.repeats << "; warmups=" << options.warmups
        << "; seed=" << options.seed << "; values=" << name(options.values) << '\n'
        << "clock=steady_clock; statistic=arithmetic_mean\n"
        << "timing=multiply call, including temporary allocation, padding, transpose, zeroing\n"
        << "excluded=input generation, reference, validation, output allocation, persistent pool "
           "construction\n"
        << "validation=full independent double reference; abs_tol=2e-4; rel_tol=2e-4\n";
    out << "sizes=";
    for (auto n : options.sizes) {
        out << n << ' ';
    }
    out << '\n';
    out << "threads=";
    for (auto p : options.threads) {
        out << p << ' ';
    }
    out << '\n';
    out << "algorithms=";
    for (auto a : options.selected) {
        out << name(a) << ' ';
    }
    out << '\n';
    if (!out) {
        throw std::runtime_error("Failed to write benchmark metadata");
    }
}

struct FailedConfiguration {
    Algorithm algorithm;
    std::size_t n;
    std::size_t threads;
};

void report_failure(Algorithm algorithm,
                    std::size_t n,
                    std::size_t threads,
                    const std::string& stage,
                    const Error& error) {
    std::ostringstream message;
    message << std::setprecision(12) << "FAIL " << name(algorithm) << " n=" << n
            << " threads=" << threads << " stage=" << stage << '\n'
            << "  mismatched elements: " << error.mismatches << '/' << n * n << '\n'
            << "  worst element C[" << error.row << ',' << error.col
            << "]: expected=" << error.expected << ", actual=" << error.actual << '\n'
            << "  abs_error=" << error.absolute << ", allowed=" << error.allowed
            << " (2e-4 + 2e-4 * abs(expected))\n"
            << "  max_abs_error=" << error.max_absolute << ", max_scaled_error=" << error.max_scaled
            << '\n'
            << "  Skipping this configuration; continuing with the next one.\n";
    std::cout << message.str() << std::flush;
}

struct MatrixProblem {
    Matrix a;
    Matrix b;
    Matrix result;
    std::vector<double> expected;

    MatrixProblem(std::size_t n, const Options& options) : a(n), b(n), result(n) {
        a.randomize(options.seed, options.values);
        b.randomize(options.seed ^ 0x9e3779b9U, options.values);
        expected = reference_product(a, b);
    }
};

Error validate(const MatrixProblem& problem,
               Algorithm algorithm,
               std::size_t threads,
               const std::string& stage) {
    const auto error = compare(problem.result, problem.expected);
    if (!error.passed) {
        report_failure(algorithm, problem.result.size(), threads, stage, error);
    }
    return error;
}

bool run_configuration(const Options& options,
                       MatrixProblem& problem,
                       ThreadPool& pool,
                       Algorithm algorithm,
                       bool timed,
                       std::ostream* csv) {
    const auto n = problem.result.size();
    const auto threads = pool.size();
    auto& result = problem.result;
    std::cout << (timed ? "Running " : "Checking ") << name(algorithm) << " n=" << n
              << " threads=" << threads << " values=" << name(options.values) << std::endl;

    multiply(algorithm, problem.a, problem.b, result, options.config, pool);
    if (!validate(problem, algorithm, threads, "initial validation").passed) {
        return false;
    }
    if (!timed) {
        std::cout << "PASS " << name(algorithm) << " n=" << n << " threads=" << threads << '\n';
        return true;
    }
    for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
        multiply(algorithm, problem.a, problem.b, result, options.config, pool);
        if (!validate(problem, algorithm, threads, "warmup " + std::to_string(warmup + 1)).passed) {
            return false;
        }
    }

    double total = 0;
    std::ostringstream measurements;
    measurements << std::setprecision(12);
    for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        multiply(algorithm, problem.a, problem.b, result, options.config, pool);
        const auto end = std::chrono::steady_clock::now();
        const auto error =
            validate(problem, algorithm, threads, "repeat " + std::to_string(repeat + 1));
        if (!error.passed) {
            return false;
        }
        const double seconds = std::chrono::duration<double>(end - start).count();
        total += seconds;
        if (csv) {
            measurements << name(algorithm) << ',' << n << ',' << threads << ','
                         << options.config.block << ',' << options.config.cutoff << ','
                         << options.seed << ',' << name(options.values) << ',' << repeat << ','
                         << seconds << ',' << error.max_absolute << ',' << error.max_scaled << ','
                         << checksum(result) << '\n';
        }
    }
    if (csv) {
        *csv << measurements.str() << std::flush;
        if (!*csv) {
            throw std::runtime_error("Failed to write benchmark results");
        }
    }
    std::cout << name(algorithm) << " n=" << n << " threads=" << threads
              << " values=" << name(options.values)
              << " mean=" << total / static_cast<double>(options.repeats) << " s" << std::endl;
    return true;
}

std::ofstream open_results(const Options& options, bool timed) {
    if (!timed || options.output.empty()) {
        return {};
    }
    if (options.output.has_parent_path()) {
        std::filesystem::create_directories(options.output.parent_path());
    }
    std::ofstream csv(options.output);
    if (!csv) {
        throw std::runtime_error("Cannot open output CSV");
    }
    csv << "algorithm,n,threads,block,cutoff,seed,values,repeat,seconds,"
           "max_abs_error,max_scaled_error,checksum\n";
    metadata(options);
    return csv;
}

}

bool benchmark(const Options& options, bool timed) {
    auto csv = open_results(options, timed);
    auto* csv_output = csv.is_open() ? &csv : nullptr;
    std::vector<FailedConfiguration> failures;
    std::size_t passed = 0;
    for (auto n : options.sizes) {
        MatrixProblem problem(n, options);
        for (auto threads : options.threads) {
            ThreadPool pool(threads);
            for (auto algorithm : options.selected) {
                if (!is_parallel(algorithm) && threads != 1) {
                    continue;
                }
                if (run_configuration(options, problem, pool, algorithm, timed, csv_output)) {
                    ++passed;
                } else {
                    failures.push_back({algorithm, n, threads});
                }
            }
        }
    }
    if (csv_output) {
        std::cout << "Saved " << options.output << " and metadata\n";
    }
    std::cout << "Completed: " << passed << " configurations passed, " << failures.size()
              << " failed.\n";
    for (const auto& failure : failures) {
        std::cout << "  FAIL " << name(failure.algorithm) << " n=" << failure.n
                  << " threads=" << failure.threads << '\n';
    }
    return failures.empty();
}

}
