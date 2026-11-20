#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include "data.hpp"

#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace loops {
enum class Backend { all, seq, fused, threads };
std::string_view name(Backend backend);

struct Options {
    std::string command = "help";
    std::vector<Task> tasks{Task::reference, Task::variant3v};
    Backend backend = Backend::all;
    std::size_t rows = 1000;
    std::size_t cols = 1000;
    std::vector<std::size_t> workers{1, 2, 8, 32};
    std::size_t repeats = 3;
    std::size_t warmups = 1;
    std::filesystem::path output;
    std::filesystem::path dump;
};

Options parse_options(int argc, char** argv, bool mpi);
void help(std::ostream& out, bool mpi);
std::filesystem::path
matrix_path(const Options& options, Task task, std::string_view backend, std::size_t workers);
} // namespace loops

#endif // OPTIONS_HPP
