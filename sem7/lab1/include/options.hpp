#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include "algorithms.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace matmul {

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

Options parse_options(int argc, char** argv);
void print_help();

} // namespace matmul

#endif // OPTIONS_HPP
