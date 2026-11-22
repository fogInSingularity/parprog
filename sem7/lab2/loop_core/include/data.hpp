#ifndef DATA_HPP
#define DATA_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace loops {

enum class Task { reference, variant1v, variant2g, variant3d };
inline constexpr std::array all_tasks{
    Task::reference,
    Task::variant1v,
    Task::variant2g,
    Task::variant3d,
};

std::string_view name(Task task);
Task parse_task(std::string_view text);
std::size_t checked_elements(std::size_t rows, std::size_t cols);
double initial_value(std::size_t row, std::size_t col) noexcept;

struct Slice {
    std::size_t begin;
    std::size_t count;
};

Slice partition(std::size_t rows, std::size_t workers, std::size_t rank);

// Zero local rows are valid: an MPI rank can own an empty slice.
struct Data {
    Task task;
    std::size_t rows;
    std::size_t cols;
    std::size_t row_offset;
    std::vector<double> a;
    // Input snapshot for 2g, output matrix for 3d.
    std::vector<double> b;

    Data(Task task,
         std::size_t rows,
         std::size_t cols,
         std::size_t row_offset = 0,
         std::size_t trailing_rows = 0);
    void reset();
    std::span<const double> output() const;
};

struct Validation {
    std::uint64_t mismatches = 0;
    std::size_t index = 0;
    double expected = 0;
    double actual = 0;
    double max_absolute = 0;

    bool passed() const { return mismatches == 0; }
};

Validation compare(std::span<const double> actual, std::span<const double> expected);
double checksum(std::span<const double> values);
void write_matrix(const std::filesystem::path& path,
                  std::span<const double> values,
                  std::size_t rows,
                  std::size_t cols,
                  Task task);

} // namespace loops

#endif // DATA_HPP
