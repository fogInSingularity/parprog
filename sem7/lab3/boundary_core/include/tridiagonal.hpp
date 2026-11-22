#ifndef TRIDIAGONAL_HPP
#define TRIDIAGONAL_HPP

#include <cstddef>
#include <span>
#include <vector>

namespace boundary {

struct Row {
    double lower;
    double diagonal;
    double upper;
    double rhs;
};

void thomas(std::span<Row> rows, std::span<double> solution);

class CyclicReduction {
public:
    void reduce(std::span<Row> rows);
    void restore(std::span<const Row> rows, std::span<double> solution) const;
    void solve(std::span<Row> rows, std::span<double> solution);

private:
    struct Eliminated {
        std::size_t index;
        std::size_t left;
        std::size_t right;
    };

    std::vector<Eliminated> eliminated_;
    std::vector<std::size_t> active_;
    std::vector<std::size_t> next_;
};

}

#endif
