#include "benchmark.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace loops {
namespace {

bool validate(const Validation& result,
              std::string_view label,
              std::string_view stage,
              std::size_t columns,
              bool report) {
    if (result.passed()) {
        return true;
    }
    if (report) {
        std::ostringstream message;
        message << std::setprecision(17) << "FAIL " << label << " stage=" << stage << '\n'
                << "  mismatches=" << result.mismatches << ", first element ["
                << result.index / columns << ',' << result.index % columns
                << "]: expected=" << result.expected << ", actual=" << result.actual << '\n'
                << "  max_abs_error=" << result.max_absolute << "; bitwise comparison\n"
                << "  Continuing with the next configuration.\n";
        std::cout << message.str() << std::flush;
    }
    return false;
}

void dump_output(const Options& options,
                 Task task,
                 std::string_view backend,
                 std::size_t workers,
                 std::span<const double> output,
                 bool report) {
    if (report && !options.dump.empty()) {
        write_matrix(
            matrix_path(options, task, backend, workers), output, options.rows, options.cols);
    }
}

} // namespace

std::ofstream open_csv(const Options& options, bool report) {
    std::ofstream output;
    if (report && options.command == "bench" && !options.output.empty()) {
        if (options.output.has_parent_path()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        output.open(options.output);
        if (!output) {
            throw std::runtime_error("Cannot open CSV: " + options.output.string());
        }
        output << "task,backend,rows,cols,workers,repeat,seconds,checksum\n";
    }
    return output;
}

bool run_configuration(const Options& options,
                       Task task,
                       std::string_view backend,
                       std::size_t workers,
                       const std::function<Sample()>& run,
                       std::ostream* csv,
                       bool report) {
    const std::string label = std::string(name(task)) + " backend=" + std::string(backend) +
                              " size=" + std::to_string(options.rows) + 'x' +
                              std::to_string(options.cols) + " workers=" + std::to_string(workers);
    if (report) {
        std::cout << (options.command == "check" ? "Checking " : "Running ") << label << std::endl;
    }
    auto sample = run();
    if (!validate(sample.validation, label, "initial check", options.cols, report)) {
        return false;
    }
    if (options.command == "check") {
        dump_output(options, task, backend, workers, sample.output, report);
        if (report) {
            std::cout << "PASS " << label << std::endl;
        }
        return true;
    }
    for (std::size_t i = 0; i < options.warmups; ++i) {
        sample = run();
        if (!validate(sample.validation,
                      label,
                      "warmup " + std::to_string(i + 1),
                      options.cols,
                      report)) {
            return false;
        }
    }
    double total = 0;
    std::ostringstream measurements;
    measurements << std::setprecision(17);
    for (std::size_t i = 0; i < options.repeats; ++i) {
        sample = run();
        if (!validate(sample.validation,
                      label,
                      "repeat " + std::to_string(i + 1),
                      options.cols,
                      report)) {
            return false;
        }
        total += sample.seconds;
        if (csv) {
            measurements << name(task) << ',' << backend << ',' << options.rows << ','
                         << options.cols << ',' << workers << ',' << i << ',' << sample.seconds
                         << ',' << checksum(sample.output) << '\n';
        }
    }
    dump_output(options, task, backend, workers, sample.output, report);
    if (csv) {
        *csv << measurements.str() << std::flush;
        if (!*csv) {
            throw std::runtime_error("Failed writing CSV");
        }
    }
    if (report) {
        std::cout << label << " mean=" << total / static_cast<double>(options.repeats) << " s"
                  << std::endl;
    }
    return true;
}
} // namespace loops
