#include "benchmark.hpp"

#include <iostream>
#include <stdexcept>

namespace {

using matmul::Algorithm;
using matmul::Matrix;
using matmul::ThreadPool;

void admission(std::size_t n) {
    Matrix a(n);
    Matrix b(n);
    Matrix c(n);
    a.fill(1);
    b.fill(1);
    ThreadPool pool(1);
    multiply(Algorithm::naive, a, b, c, {}, pool);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (c(i, j) != static_cast<float>(n)) {
                throw std::runtime_error("Admission check failed");
            }
        }
    }
    std::cout << "PASS: naive, " << n << 'x' << n << ", A=B=1, every C[i,j]=" << n << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = matmul::parse_options(argc, argv);
        if (options.command == "help") {
            matmul::print_help();
        } else if (options.command == "info") {
            matmul::print_info(std::cout);
        } else if (options.command == "admission") {
            admission(options.size);
        } else if (options.command == "check") {
            return matmul::benchmark(options, false) ? 0 : 1;
        } else if (options.command == "bench") {
            return matmul::benchmark(options, true) ? 0 : 1;
        } else {
            throw std::invalid_argument("Unknown command: " + options.command);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
