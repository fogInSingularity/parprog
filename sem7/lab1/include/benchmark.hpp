#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include "options.hpp"

#include <iosfwd>

namespace matmul {

void print_info(std::ostream& out);
bool benchmark(const Options& options, bool timed);

} // namespace matmul

#endif // BENCHMARK_HPP
