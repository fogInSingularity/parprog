#include "distributed.hpp"

#include <mpi.h>

#include <array>

namespace boundary {
namespace {

constexpr int coefficients_per_row = 4;

void pack(Row row, double* output) {
    output[0] = row.lower;
    output[1] = row.diagonal;
    output[2] = row.upper;
    output[3] = row.rhs;
}

}

DistributedReduction::DistributedReduction(std::size_t unknowns, int rank, int ranks)
    : rank_(rank), ranks_(ranks), counts_(ranks), offsets_(ranks) {
    int offset = 0;
    for (int p = 0; p < ranks; ++p) {
        const auto slice = partition(unknowns, ranks, p);
        counts_[p] = coefficients_per_row * (slice.count == 1 ? 1 : 2);
        offsets_[p] = offset;
        offset += counts_[p];
    }
    packed_.resize(offset);
    interface_.resize(offset / coefficients_per_row);
    endpoints_.resize(interface_.size());
}

void DistributedReduction::exchange(std::span<double> values) const {
    const auto left = rank_ == 0 ? MPI_PROC_NULL : rank_ - 1;
    const auto right = rank_ + 1 == ranks_ ? MPI_PROC_NULL : rank_ + 1;
    MPI_Sendrecv(&values[1],
                 1,
                 MPI_DOUBLE,
                 left,
                 0,
                 &values.back(),
                 1,
                 MPI_DOUBLE,
                 right,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(&values[values.size() - 2],
                 1,
                 MPI_DOUBLE,
                 right,
                 1,
                 &values.front(),
                 1,
                 MPI_DOUBLE,
                 left,
                 1,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
}

void DistributedReduction::solve(std::span<Row> rows, std::span<double> correction) {
    local_.reduce(rows);

    std::array<double, 2 * coefficients_per_row> border{};
    pack(rows.front(), border.data());
    if (rows.size() > 1) {
        pack(rows.back(), border.data() + coefficients_per_row);
    }
    MPI_Allgatherv(border.data(),
                   counts_[rank_],
                   MPI_DOUBLE,
                   packed_.data(),
                   counts_.data(),
                   offsets_.data(),
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);

    for (std::size_t i = 0; i < interface_.size(); ++i) {
        const auto* row = packed_.data() + coefficients_per_row * i;
        interface_[i] = {row[0], row[1], row[2], row[3]};
    }
    reduced_.solve(interface_, endpoints_);
    const auto offset = offsets_[rank_] / coefficients_per_row;
    correction.front() = endpoints_[offset];
    if (rows.size() > 1) {
        correction.back() = endpoints_[offset + 1];
    }
    local_.restore(rows, correction);
}

std::vector<double>
gather_solution(std::span<const double> local, std::size_t total, int rank, int ranks) {
    std::vector<double> collected;
    std::vector<int> counts;
    std::vector<int> offsets;
    if (rank == 0) {
        collected.resize(total);
        counts.resize(ranks);
        offsets.resize(ranks);
        for (int p = 0; p < ranks; ++p) {
            const auto slice = partition(total, ranks, p);
            counts[p] = static_cast<int>(slice.count);
            offsets[p] = static_cast<int>(slice.begin);
        }
    }
    MPI_Gatherv(local.data(),
                static_cast<int>(local.size()),
                MPI_DOUBLE,
                collected.data(),
                counts.data(),
                offsets.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);
    return collected;
}

}
