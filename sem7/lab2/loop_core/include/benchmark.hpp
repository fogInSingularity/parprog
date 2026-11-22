#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include "options.hpp"

#include <fstream>
#include <functional>
#include <iosfwd>

namespace loops {
struct Sample {
    double seconds;
    std::span<const double> output;
    Validation validation;
};

std::ofstream open_csv(const Options& options, bool report);
bool run_configuration(const Options& options,
                       Task task,
                       std::string_view backend,
                       std::size_t workers,
                       const std::function<Sample()>& run,
                       std::ostream* csv,
                       bool report = true);
}

#endif
