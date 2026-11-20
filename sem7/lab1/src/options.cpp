#include "options.hpp"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace matmul {
namespace {

std::size_t number(std::string_view text, bool allow_zero = false) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || (!allow_zero && value == 0)) {
        throw std::invalid_argument("Invalid integer: " + std::string(text));
    }
    return value;
}

std::vector<std::string_view> split(std::string_view text) {
    std::vector<std::string_view> result;
    for (;;) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        if (item.empty()) {
            throw std::invalid_argument("Empty list item");
        }
        result.push_back(item);
        if (comma == std::string_view::npos) {
            return result;
        }
        text.remove_prefix(comma + 1);
    }
}

std::vector<std::size_t> numbers(std::string_view text) {
    std::set<std::size_t> result;
    for (auto item : split(text)) {
        result.insert(number(item));
    }
    return {result.begin(), result.end()};
}

} // namespace

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc > 1) {
        options.command = argv[1];
    }
    if (options.command == "--help" || options.command == "-h") {
        options.command = "help";
    }
    for (int i = 2; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key == "--help") {
            options.command = "help";
            return options;
        }
        if (i + 1 == argc) {
            throw std::invalid_argument("Missing value for " + std::string(key));
        }
        const std::string_view value = argv[++i];
        if (key == "--size") {
            options.size = number(value);
        } else if (key == "--sizes") {
            options.sizes = numbers(value);
        } else if (key == "--threads") {
            options.threads = numbers(value);
        } else if (key == "--block") {
            options.config.block = number(value);
        } else if (key == "--cutoff") {
            options.config.cutoff = number(value);
        } else if (key == "--repeats") {
            options.repeats = number(value);
        } else if (key == "--warmups") {
            options.warmups = number(value, true);
        } else if (key == "--seed") {
            const auto seed = number(value, true);
            if (seed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("Seed must fit in uint32_t");
            }
            options.seed = static_cast<std::uint32_t>(seed);
        } else if (key == "--output") {
            options.output = value;
        } else if (key == "--values") {
            if (value == "positive") {
                options.values = InputRange::positive;
            } else if (value == "signed") {
                options.values = InputRange::signed_unit;
            } else {
                throw std::invalid_argument("--values must be positive or signed");
            }
        } else if (key == "--algorithms") {
            options.selected.clear();
            for (auto item : split(value)) {
                const auto algorithm = parse_algorithm(item);
                if (std::find(options.selected.begin(), options.selected.end(), algorithm) ==
                    options.selected.end()) {
                    options.selected.push_back(algorithm);
                }
            }
        } else {
            throw std::invalid_argument("Unknown option: " + std::string(key));
        }
    }
    // Every parallel series needs its own one-thread baseline for S_p and E_p.
    if (std::find(options.threads.begin(), options.threads.end(), 1) == options.threads.end()) {
        options.threads.insert(options.threads.begin(), 1);
    }
    return options;
}

void print_help() {
    std::cout
        << "Matrix multiplication lab (C++20, float32, std::jthread, Highway)\n"
        << "  matrix_lab info\n"
        << "  matrix_lab admission [--size 512]\n"
        << "  matrix_lab check [options]\n"
        << "  matrix_lab bench [options]\n\n"
        << "Options:\n"
        << "  --sizes 8,17,32,63       matrix sizes (small by default)\n"
        << "  --algorithms naive,transposed,blocked,strassen,parallel,simd,combined,"
           "combined-no-strassen\n"
        << "  --threads 1,2,8,32       total participants, including caller; 1 added if missing\n"
        << "  --block 32               cache tile size\n"
        << "  --cutoff 64              Strassen leaf size\n"
        << "  --repeats 5 --warmups 1  timing repetitions / extra warmup multiplications\n"
        << "  --seed 42                reproducible input seed\n"
        << "  --values positive|signed [1,2] by default; signed selects [-1,1]\n"
        << "  --output FILE           optional CSV plus .meta.txt; default: terminal only\n";
}

} // namespace matmul
