#include "algorithms.hpp"
#include "simd.hpp"

#include <hwy/base.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace matmul;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

struct Options {
    std::string command = "help";
    std::size_t size = 512;
    std::vector<std::size_t> sizes{8, 17, 32, 63};
    std::vector<std::size_t> threads{1, 2, 8, 32};
    std::vector<Algorithm> selected{algorithms.begin(), algorithms.end()};
    Config config;
    std::size_t repeats = 5;
    std::size_t warmups = 1;
    std::uint32_t seed = 42;
    InputRange values = InputRange::positive;
    std::filesystem::path output;
};

std::size_t number(std::string_view text, bool allow_zero = false) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || (!allow_zero && value == 0))
        throw std::invalid_argument("Invalid integer: " + std::string(text));
    return value;
}

std::vector<std::string_view> split(std::string_view text) {
    std::vector<std::string_view> result;
    for (;;) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        if (item.empty()) throw std::invalid_argument("Empty list item");
        result.push_back(item);
        if (comma == std::string_view::npos) return result;
        text.remove_prefix(comma + 1);
    }
}

std::vector<std::size_t> numbers(std::string_view text) {
    std::set<std::size_t> result;
    for (auto item : split(text)) result.insert(number(item));
    return {result.begin(), result.end()};
}

Options parse(int argc, char** argv) {
    Options options;
    if (argc > 1) options.command = argv[1];
    if (options.command == "--help" || options.command == "-h") options.command = "help";
    for (int i = 2; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key == "--help") { options.command = "help"; return options; }
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + std::string(key));
        const std::string_view value = argv[++i];
        if (key == "--size") options.size = number(value);
        else if (key == "--sizes") options.sizes = numbers(value);
        else if (key == "--threads") options.threads = numbers(value);
        else if (key == "--block") options.config.block = number(value);
        else if (key == "--cutoff") options.config.cutoff = number(value);
        else if (key == "--repeats") options.repeats = number(value);
        else if (key == "--warmups") options.warmups = number(value, true);
        else if (key == "--seed") {
            const auto seed = number(value, true);
            if (seed > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("Seed must fit in uint32_t");
            options.seed = static_cast<std::uint32_t>(seed);
        }
        else if (key == "--output") options.output = value;
        else if (key == "--values") {
            if (value == "positive") options.values = InputRange::positive;
            else if (value == "signed") options.values = InputRange::signed_unit;
            else throw std::invalid_argument("--values must be positive or signed");
        }
        else if (key == "--algorithms") {
            options.selected.clear();
            for (auto item : split(value)) {
                const auto algorithm = parse_algorithm(item);
                if (std::find(options.selected.begin(), options.selected.end(), algorithm) == options.selected.end())
                    options.selected.push_back(algorithm);
            }
        } else throw std::invalid_argument("Unknown option: " + std::string(key));
    }
    // Every parallel series needs its own one-thread baseline for S_p and E_p.
    if (std::find(options.threads.begin(), options.threads.end(), 1) == options.threads.end())
        options.threads.insert(options.threads.begin(), 1);
    return options;
}

void info(std::ostream& out) {
    out << "C++20; float32; standard C++ threads\n"
        << "compiler=" << __VERSION__ << '\n'
        << "highway=" << HWY_MAJOR << '.' << HWY_MINOR << '.' << HWY_PATCH << '\n'
        << "simd_target=" << simd_target() << "; float_lanes=" << simd_lanes() << '\n'
        << "hardware_threads=" << std::thread::hardware_concurrency() << '\n'
        << "scalar_auto_vectorization=disabled; fast_math=disabled\n";
#ifdef NDEBUG
    out << "assertions=disabled (use a Release build for performance)\n";
#else
    out << "assertions=enabled (do not use a Debug build for performance conclusions)\n";
#endif
    std::ifstream cpu("/proc/cpuinfo");
    for (std::string line; std::getline(cpu, line); ) {
        if (line.starts_with("model name")) { out << line << '\n'; break; }
    }
}

void metadata(const Options& options) {
    std::ofstream out(options.output.string() + ".meta.txt");
    if (!out) throw std::runtime_error("Cannot write benchmark metadata");
    info(out);
    out << "block=" << options.config.block << "; cutoff=" << options.config.cutoff
        << "; repeats=" << options.repeats << "; warmups=" << options.warmups
        << "; seed=" << options.seed << "; values=" << name(options.values) << '\n'
        << "clock=steady_clock; statistic=arithmetic_mean\n"
        << "timing=multiply call, including temporary allocation, padding, transpose, zeroing\n"
        << "excluded=input generation, reference, validation, output allocation, persistent pool construction\n"
        << "validation=full independent double reference; abs_tol=2e-4; rel_tol=2e-4\n";
    out << "sizes="; for (auto n : options.sizes) out << n << ' '; out << '\n';
    out << "threads="; for (auto p : options.threads) out << p << ' '; out << '\n';
    out << "algorithms="; for (auto a : options.selected) out << name(a) << ' '; out << '\n';
    if (!out) throw std::runtime_error("Failed to write benchmark metadata");
}

void admission(std::size_t n) {
    Matrix a(n), b(n), c(n);
    a.fill(1); b.fill(1);
    ThreadPool pool(1);
    multiply(Algorithm::naive, a, b, c, {}, pool);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (c(i, j) != static_cast<float>(n))
                throw std::runtime_error("Admission check failed");
    std::cout << "PASS: naive, " << n << 'x' << n << ", A=B=1, every C[i,j]=" << n << '\n';
}

struct FailedConfiguration {
    Algorithm algorithm;
    std::size_t n;
    std::size_t threads;
};

void report_failure(Algorithm algorithm, std::size_t n, std::size_t threads,
                    const std::string& stage, const Error& error) {
    std::ostringstream message;
    message << std::setprecision(12)
        << "FAIL " << name(algorithm) << " n=" << n << " threads=" << threads
        << " stage=" << stage << '\n'
        << "  mismatched elements: " << error.mismatches << '/' << n * n << '\n'
        << "  worst element C[" << error.row << ',' << error.col << "]: expected="
        << error.expected << ", actual=" << error.actual << '\n'
        << "  abs_error=" << error.absolute << ", allowed=" << error.allowed
        << " (2e-4 + 2e-4 * abs(expected))\n"
        << "  max_abs_error=" << error.max_absolute << ", max_scaled_error=" << error.max_scaled << '\n'
        << "  Skipping this configuration; continuing with the next one.\n";
    std::cout << message.str() << std::flush;
}

bool benchmark(const Options& options, bool timed) {
    const bool export_csv = timed && !options.output.empty();
    if (export_csv && options.output.has_parent_path())
        std::filesystem::create_directories(options.output.parent_path());
    std::ofstream csv;
    if (export_csv) {
        csv.open(options.output);
        if (!csv) throw std::runtime_error("Cannot open output CSV");
        csv << "algorithm,n,threads,block,cutoff,seed,values,repeat,seconds,max_abs_error,max_scaled_error,checksum\n"
            << std::setprecision(12);
        metadata(options);
    }
    std::vector<FailedConfiguration> failures;
    std::size_t passed = 0;
    for (auto n : options.sizes) {
        Matrix a(n), b(n), c(n);
        a.randomize(options.seed, options.values);
        b.randomize(options.seed ^ 0x9e3779b9U, options.values);
        const auto expected = reference_product(a, b);
        for (auto p : options.threads) {
            ThreadPool pool(p);
            for (auto algorithm : options.selected) {
                if (!is_parallel(algorithm) && p != 1) continue;
                const auto run = [&] {
                    multiply(algorithm, a, b, c, options.config, pool);
                };
                const auto validate = [&](const std::string& stage) {
                    const auto error = compare(c, expected);
                    if (!error.passed) {
                        report_failure(algorithm, n, p, stage, error);
                        failures.push_back({algorithm, n, p});
                    }
                    return error;
                };
                // Always validate once before any timed repeats, even with --warmups 0.
                std::cout << (timed ? "Running " : "Checking ") << name(algorithm)
                          << " n=" << n << " threads=" << p << " values=" << name(options.values) << std::endl;
                run();
                if (!validate("initial validation").passed) continue;
                if (!timed) {
                    std::cout << "PASS " << name(algorithm) << " n=" << n << " threads=" << p << '\n';
                    ++passed;
                    continue;
                }
                bool valid = true;
                for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
                    run();
                    if (!validate("warmup " + std::to_string(warmup + 1)).passed) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) continue;
                double total = 0;
                // Commit CSV records only when every repeat in this group is valid.
                std::ostringstream measurements;
                measurements << std::setprecision(12);
                for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    run();
                    const auto end = std::chrono::steady_clock::now();
                    const auto error = validate("repeat " + std::to_string(repeat + 1));
                    if (!error.passed) {
                        valid = false;
                        break;
                    }
                    const double seconds = std::chrono::duration<double>(end - start).count();
                    total += seconds;
                    if (export_csv) {
                        measurements << name(algorithm) << ',' << n << ',' << p << ',' << options.config.block
                            << ',' << options.config.cutoff << ',' << options.seed << ',' << name(options.values) << ',' << repeat
                            << ',' << seconds << ',' << error.max_absolute << ',' << error.max_scaled
                            << ',' << checksum(c) << '\n';
                    }
                }
                if (!valid) continue;
                if (export_csv) {
                    csv << measurements.str();
                    csv.flush();
                    if (!csv) throw std::runtime_error("Failed to write benchmark results");
                }
                std::cout << name(algorithm) << " n=" << n << " threads=" << p
                          << " values=" << name(options.values)
                          << " mean=" << total / static_cast<double>(options.repeats) << " s" << std::endl;
                ++passed;
            }
        }
    }
    if (export_csv) std::cout << "Saved " << options.output << " and metadata\n";
    std::cout << "Completed: " << passed << " configurations passed, " << failures.size() << " failed.\n";
    for (const auto& failure : failures)
        std::cout << "  FAIL " << name(failure.algorithm) << " n=" << failure.n
                  << " threads=" << failure.threads << '\n';
    return failures.empty();
}

void help() {
    std::cout << "Matrix multiplication lab (C++20, float32, std::jthread, Highway)\n"
        << "  matrix_lab info\n"
        << "  matrix_lab admission [--size 512]\n"
        << "  matrix_lab check [options]\n"
        << "  matrix_lab bench [options]\n\n"
        << "Options:\n"
        << "  --sizes 8,17,32,63       matrix sizes (small by default)\n"
        << "  --algorithms naive,transposed,blocked,strassen,parallel,simd,combined\n"
        << "  --threads 1,2,8,32       total participants, including caller; 1 added if missing\n"
        << "  --block 32               cache tile size\n"
        << "  --cutoff 64              Strassen leaf size\n"
        << "  --repeats 5 --warmups 1  timing repetitions / extra warmup multiplications\n"
        << "  --seed 42                reproducible input seed\n"
        << "  --values positive|signed [1,2] by default; signed selects [-1,1]\n"
        << "  --output FILE           optional CSV plus .meta.txt; default: terminal only\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.command == "help") help();
        else if (options.command == "info") info(std::cout);
        else if (options.command == "admission") admission(options.size);
        else if (options.command == "check") return benchmark(options, false) ? 0 : 1;
        else if (options.command == "bench") return benchmark(options, true) ? 0 : 1;
        else throw std::invalid_argument("Unknown command: " + options.command);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
