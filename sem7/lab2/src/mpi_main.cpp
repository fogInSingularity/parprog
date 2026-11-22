#include "benchmark.hpp"
#include "kernels.hpp"
#include "options.hpp"

#include <mpi.h>

#include <algorithm>
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

GatherLayout gather_layout(Task task, const Options& options, int processes) {
    GatherLayout layout;
    layout.counts.reserve(processes);
    layout.offsets.reserve(processes);
    std::size_t offset = 0;
    for (int rank = 0; rank < processes; ++rank) {
        std::size_t count = 0;
        if (task == Task::variant1v) {
            const auto chains = chain_count(options.rows, options.cols);
            for (std::size_t index = rank; index < chains; index += processes) {
                count += chain_at(index, options.rows, options.cols).count;
            }
        } else {
            count = partition(options.rows, processes, rank).count * options.cols;
        }
        layout.counts.push_back(static_cast<int>(count));
        layout.offsets.push_back(static_cast<int>(offset));
        offset += count;
    }
    return layout;
}

Data make_local_data(Task task, const Options& options, Slice slice) {
    if (task == Task::variant1v) {
        return {task, 0, options.cols};
    }
    const auto trailing = task == Task::variant3d && slice.count != 0
                              ? std::min<std::size_t>(3, options.rows - slice.begin - slice.count)
                              : 0;
    return {task, slice.count, options.cols, slice.begin, trailing};
}

void pack_chains(const Options& options, int rank, int processes, std::span<double> packed) {
    std::size_t offset = 0;
    const auto chains = chain_count(options.rows, options.cols);
    for (std::size_t index = rank; index < chains; index += processes) {
        const auto chain = chain_at(index, options.rows, options.cols);
        compute_chain(chain, initial_value(chain.row, chain.col), packed.data() + offset, 1);
        offset += chain.count;
    }
}

void unpack_chains(const Options& options,
                   const GatherLayout& layout,
                   std::span<const double> packed,
                   std::span<double> result) {
    const auto processes = layout.counts.size();
    const auto stride = options.cols + 3;
    const auto chains = chain_count(options.rows, options.cols);
    for (std::size_t rank = 0; rank < processes; ++rank) {
        std::size_t offset = layout.offsets[rank];
        for (std::size_t index = rank; index < chains; index += processes) {
            const auto chain = chain_at(index, options.rows, options.cols);
            const auto head = chain.row * options.cols + chain.col;
            for (std::size_t i = 0; i < chain.count; ++i) {
                result[head + i * stride] = packed[offset++];
            }
        }
    }
}

double measure_kernel(const auto& compute) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    compute();
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;

    double maximum = 0;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, root_rank, MPI_COMM_WORLD);
    return maximum;
}

void gather(std::span<const double> local,
            std::span<double> collected,
            const GatherLayout& layout) {
    MPI_Gatherv(local.data(),
                static_cast<int>(local.size()),
                MPI_DOUBLE,
                collected.data(),
                layout.counts.data(),
                layout.offsets.data(),
                MPI_DOUBLE,
                root_rank,
                MPI_COMM_WORLD);
}

int execute(const Options& options, int rank, int processes) {
    const auto elements = checked_elements(options.rows, options.cols);
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("MPI_Gatherv count exceeds INT_MAX");
    }
    const auto slice = partition(options.rows, processes, rank);
    const bool root = rank == root_rank;
    auto csv = open_csv(options, root);
    auto* csv_output = csv.is_open() ? &csv : nullptr;
    std::size_t passed = 0;
    std::size_t failed = 0;

    for (auto task : options.tasks) {
        const bool chains = task == Task::variant1v;
        const auto layout = gather_layout(task, options, processes);
        auto local = make_local_data(task, options, slice);
        Data expected(task, root ? options.rows : 0, options.cols);
        std::vector<double> packed(chains ? layout.counts[rank] : 0);
        std::vector<double> collected(root ? elements : 0);
        std::vector<double> unpacked(root && chains ? elements : 0);
        if (root) {
            sequential(expected);
        }
        const auto compute = [&] {
            if (chains) {
                pack_chains(options, rank, processes, packed);
            } else {
                fused(local);
            }
        };
        const auto measure = [&] {
            local.reset();
            const auto elapsed = measure_kernel(compute);
            gather(chains ? std::span<const double>(packed) : local.output(), collected, layout);
            const auto result =
                chains ? std::span<const double>(unpacked) : std::span<const double>(collected);
            Validation validation;
            if (root) {
                if (chains) {
                    unpack_chains(options, layout, collected, unpacked);
                }
                validation = compare(result, expected.output());
            }
            MPI_Bcast(&validation.mismatches, 1, MPI_UINT64_T, root_rank, MPI_COMM_WORLD);
            return Sample{elapsed, result, validation};
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

}
}

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
