#include "options.hpp"

#include <charconv>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string_view>

namespace loops {
namespace {
std::size_t number(std::string_view text, bool allow_zero = false) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || (!allow_zero && value == 0)) {
        throw std::invalid_argument("Invalid integer: " + std::string(text));
    }
    return value;
}

std::vector<std::size_t> workers(std::string_view text) {
    std::set<std::size_t> result{1};
    for (;;) {
        const auto comma = text.find(',');
        result.insert(number(text.substr(0, comma)));
        if (comma == std::string_view::npos) {
            break;
        }
        text.remove_prefix(comma + 1);
    }
    return {result.begin(), result.end()};
}

Backend parse_backend(std::string_view text) {
    for (auto backend : {Backend::all, Backend::seq, Backend::fused, Backend::threads}) {
        if (name(backend) == text) {
            return backend;
        }
    }
    throw std::invalid_argument("Backend must be all, seq, fused, or threads");
}
} // namespace

std::string_view name(Backend backend) {
    switch (backend) {
    case Backend::all:
        return "all";
    case Backend::seq:
        return "seq";
    case Backend::fused:
        return "fused";
    case Backend::threads:
        return "threads";
    }
    throw std::invalid_argument("Unknown backend");
}

Options parse_options(int argc, char** argv, bool mpi) {
    Options options;
    if (argc > 1) {
        options.command = argv[1];
    }
    if (options.command == "--help" || options.command == "-h") {
        options.command = "help";
    }
    if (options.command != "help" && options.command != "check" && options.command != "bench") {
        throw std::invalid_argument("Command must be check or bench");
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
        if (key == "--rows") {
            options.rows = number(value);
        } else if (key == "--cols") {
            options.cols = number(value);
        } else if (key == "--task") {
            if (value == "all") {
                options.tasks = {Task::reference, Task::variant3v};
            } else {
                options.tasks = {parse_task(value)};
            }
        } else if (key == "--workers" && !mpi) {
            options.workers = workers(value);
        } else if (key == "--backend" && !mpi) {
            options.backend = parse_backend(value);
        } else if (key == "--repeats") {
            options.repeats = number(value);
        } else if (key == "--warmups") {
            options.warmups = number(value, true);
        } else if (key == "--output") {
            options.output = value;
        } else if (key == "--dump") {
            options.dump = value;
        } else if (mpi && (key == "--workers" || key == "--backend")) {
            throw std::invalid_argument("MPI worker count is set by mpirun -np; backend is mpi");
        } else {
            throw std::invalid_argument("Unknown option: " + std::string(key));
        }
    }
    checked_elements(options.rows, options.cols);
    return options;
}

void help(std::ostream& out, bool mpi) {
    out << (mpi ? "loop_mpi" : "loop_lab") << " check|bench [options]\n"
        << "  --task reference|3v|all  default: all (3v is task 3в)\n"
        << "  --rows 1000 --cols 1000  positive independent dimensions; supports 5000x5000\n";
    if (!mpi) {
        out << "  --backend all|seq|fused|threads\n"
            << "  --workers 1,2,8,32       total C++ threads, including the caller\n";
    } else {
        out << "  MPI process count is set with mpirun -np N\n";
    }
    out << "  --repeats 3 --warmups 1  reset inputs before every invocation\n"
        << "  --output FILE           optional CSV; text goes to terminal by default\n"
        << "  --dump DIR              write full result matrices for file comparison\n"
        << "Timing excludes initialization, reference, verification, and file output.\n";
}

std::filesystem::path
matrix_path(const Options& options, Task task, std::string_view backend, std::size_t workers) {
    return options.dump / (std::string(name(task)) + '-' + std::to_string(options.rows) + 'x' +
                           std::to_string(options.cols) + '-' + std::string(backend) + "-p" +
                           std::to_string(workers) + ".txt");
}
} // namespace loops
