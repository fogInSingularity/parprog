#include "benchmark.hpp"
#include "kernels.hpp"
#include "options.hpp"

#include <mpi.h>

#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace loops {
namespace {

constexpr int root_rank = 0;

struct GatherLayout {
    std::vector<int> counts;
    std::vector<int> offsets;
};

GatherLayout gather_layout(const Options& options, int processes) {
    GatherLayout layout;
    layout.counts.reserve(processes);
    layout.offsets.reserve(processes);
    for (int rank = 0; rank < processes; ++rank) {
        const auto part = partition(options.rows, processes, rank);
        layout.counts.push_back(static_cast<int>(part.count * options.cols));
        layout.offsets.push_back(static_cast<int>(part.begin * options.cols));
    }
    return layout;
}

double measure_kernel(Data& local) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    fused(local);
    // Include the time until every rank completes its work.
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;

    double maximum = 0;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, root_rank, MPI_COMM_WORLD);
    return maximum;
}

Validation gather_and_validate(std::span<const double> local,
                               std::span<double> collected,
                               std::span<const double> expected,
                               const GatherLayout& layout,
                               bool root) {
    MPI_Gatherv(local.data(),
                static_cast<int>(local.size()),
                MPI_DOUBLE,
                collected.data(),
                layout.counts.data(),
                layout.offsets.data(),
                MPI_DOUBLE,
                root_rank,
                MPI_COMM_WORLD);

    Validation validation;
    if (root) {
        validation = compare(collected, expected);
    }
    // All ranks must skip a failed configuration together.
    MPI_Bcast(&validation.mismatches, 1, MPI_UINT64_T, root_rank, MPI_COMM_WORLD);
    return validation;
}

Sample measure_mpi(Data& local,
                   std::span<double> collected,
                   std::span<const double> expected,
                   const GatherLayout& layout,
                   bool root) {
    local.reset();
    const double elapsed = measure_kernel(local);
    const auto validation = gather_and_validate(local.output(), collected, expected, layout, root);
    return {elapsed, collected, validation};
}

int execute(const Options& options, int rank, int processes) {
    const auto elements = checked_elements(options.rows, options.cols);
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("MPI_Gatherv count exceeds INT_MAX");
    }
    const auto slice = partition(options.rows, processes, rank);
    const auto layout = gather_layout(options, processes);
    const bool root = rank == root_rank;
    auto csv = open_csv(options, root);
    auto* csv_output = csv.is_open() ? &csv : nullptr;
    std::size_t passed = 0;
    std::size_t failed = 0;

    for (auto task : options.tasks) {
        Data local(task, slice.count, options.cols, slice.begin);
        Data expected(task, root ? options.rows : 0, options.cols);
        std::vector<double> collected(root ? elements : 0);
        if (root) {
            sequential(expected);
        }
        const auto measure = [&local, &collected, &expected, &layout, root] {
            return measure_mpi(local, collected, expected.output(), layout, root);
        };
        if (run_configuration(options, task, "mpi", processes, measure, csv_output, root)) {
            ++passed;
        } else {
            ++failed;
        }
    }
    if (root) {
        std::cout << "Completed: " << passed << " passed, " << failed << " failed.\n";
    }
    return failed == 0 ? 0 : 1;
}

} // namespace
} // namespace loops

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 2;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        const auto options = loops::parse_options(argc, argv, true);
        int status = 0;
        if (options.command == "help") {
            if (rank == loops::root_rank) {
                loops::help(std::cout, true);
            }
        } else {
            int processes = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &processes);
            status = loops::execute(options, rank, processes);
        }
        MPI_Finalize();
        return status;
    } catch (const std::exception& error) {
        std::cerr << "Error on MPI rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
