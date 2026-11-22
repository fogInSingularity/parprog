#ifndef DISTRIBUTED_HPP
#define DISTRIBUTED_HPP

#include "numerov.hpp"

#include <span>
#include <vector>

namespace boundary {

class DistributedReduction {
public:
    DistributedReduction(std::size_t unknowns, int rank, int ranks);
    void exchange(std::span<double> values) const;
    void solve(std::span<Row> rows, std::span<double> correction);

private:
    int rank_;
    int ranks_;
    std::vector<int> counts_;
    std::vector<int> offsets_;
    std::vector<double> packed_;
    std::vector<Row> interface_;
    std::vector<double> endpoints_;
    CyclicReduction local_;
    CyclicReduction reduced_;
};

std::vector<double>
gather_solution(std::span<const double> local, std::size_t total, int rank, int ranks);

}

#endif
