#include "distributed.hpp"
#include "numerov.hpp"
#include "tridiagonal.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace boundary {
namespace {

struct Options {
    Grid grid{100.0, 0, false};
    std::string_view solver = "reduction";
    int repeats = 1;
    int warmups = 0;
    std::filesystem::path output;
    bool help = false;
};

template <class T> T number(std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("Invalid number: " + std::string(text));
    }
    return value;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key == "--help") {
            options.help = true;
            return options;
        }
        if (key == "--full") {
            options.grid.full = true;
            continue;
        }
        if (++i == argc) {
            throw std::invalid_argument("Missing value for " + std::string(key));
        }
        const std::string_view value = argv[i];
        if (key == "--a") {
            options.grid.a = number<double>(value);
        } else if (key == "--intervals") {
            options.grid.intervals = number<std::size_t>(value);
            if (options.grid.intervals < 2) {
                throw std::invalid_argument("At least two intervals are required");
            }
        } else if (key == "--solver") {
            options.solver = value;
        } else if (key == "--repeats") {
            options.repeats = number<int>(value);
        } else if (key == "--warmups") {
            options.warmups = number<int>(value);
        } else if (key == "--output") {
            options.output = value;
        } else {
            throw std::invalid_argument("Unknown option: " + std::string(key));
        }
    }
    if (options.solver != "thomas" && options.solver != "reduction") {
        throw std::invalid_argument("Solver must be thomas or reduction");
    }
    if (!std::isfinite(options.grid.a) || options.grid.a <= 0 || options.repeats < 1 ||
        options.warmups < 0) {
        throw std::invalid_argument(
            "Expected finite positive a, positive repeats and nonnegative warmups");
    }
    constexpr auto maximum = std::numeric_limits<int>::max();
    if (options.grid.intervals == 0) {
        const auto length = options.grid.full ? 20.0 : 10.0;
        const auto intervals = std::ceil(length * std::sqrt(options.grid.a) / 0.1);
        if (intervals > maximum) {
            throw std::invalid_argument("Grid exceeds MPI count range");
        }
        options.grid.intervals = std::max<std::size_t>(2, static_cast<std::size_t>(intervals));
    }
    if (options.grid.intervals > static_cast<std::size_t>(maximum)) {
        throw std::invalid_argument("Grid exceeds MPI count range");
    }
    return options;
}

void write_solution(const Options& options, std::span<const double> solution) {
    if (options.output.has_parent_path()) {
        std::filesystem::create_directories(options.output.parent_path());
    }
    std::ofstream stream(options.output);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "x,y\n" << std::setprecision(17);
    if (options.grid.full) {
        stream << -10 << ',' << boundary_value() << '\n';
    } else {
        stream << -10 << ',' << boundary_value() << '\n';
        for (std::size_t i = solution.size(); --i > 0;) {
            stream << -options.grid.coordinate(i) << ',' << solution[i] << '\n';
        }
    }
    for (std::size_t i = 0; i < solution.size(); ++i) {
        stream << options.grid.coordinate(i) << ',' << solution[i] << '\n';
    }
    stream << 10 << ',' << boundary_value() << '\n';
    stream.close();
}

std::array<double, 2> mpi_max(std::array<double, 2> local) {
    std::array<double, 2> global{};
    MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

template <class Solve>
void run(const Options& options, NewtonState& state, int rank, int ranks, const Solve& solve) {
    std::vector<double> samples;
    samples.reserve(options.repeats);
    Convergence convergence{};
    for (int i = -options.warmups; i < options.repeats; ++i) {
        state.reset();
        MPI_Barrier(MPI_COMM_WORLD);
        const auto start = MPI_Wtime();
        convergence = solve();
        const auto elapsed = MPI_Wtime() - start;
        double maximum = 0;
        MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && i >= 0) {
            samples.push_back(maximum);
        }
    }
    if (rank == 0) {
        std::cout << std::setprecision(12) << "solver=" << options.solver << " ranks=" << ranks
                  << " a=" << options.grid.a << " intervals=" << options.grid.intervals
                  << " h=" << options.grid.step() << " iterations=" << convergence.iterations
                  << " residual=" << convergence.residual
                  << " correction=" << convergence.correction << " mean_seconds="
                  << std::accumulate(samples.begin(), samples.end(), 0.0) /
                         static_cast<double>(samples.size())
                  << "\nsamples";
        for (const auto value : samples) {
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    }
    if (!options.output.empty()) {
        const auto solution = gather_solution(state.values(), options.grid.unknowns(), rank, ranks);
        if (rank == 0) {
            write_solution(options, solution);
        }
    }
}

void execute(const Options& options, int rank, int ranks) {
    if (options.help) {
        if (rank == 0) {
            std::cout << "boundary_lab [--solver thomas|reduction] [--a VALUE] [--intervals N]\n"
                         "             [--full] [--repeats N] [--warmups N] [--output FILE]\n";
        }
        return;
    }
    const auto unknowns = options.grid.unknowns();
    if (static_cast<std::size_t>(ranks) > unknowns || (options.solver == "thomas" && ranks != 1)) {
        throw std::invalid_argument(
            "Thomas uses one rank; reduction requires at least one unknown per rank");
    }
    const auto slice = partition(unknowns, ranks, rank);
    NewtonState state(options.grid, slice);
    if (options.solver == "thomas") {
        run(options, state, rank, ranks, [&] {
            return newton(
                state,
                [](std::span<double>) {},
                thomas,
                [](std::array<double, 2> values) { return values; });
        });
    } else {
        DistributedReduction reduction(unknowns, rank, ranks);
        run(options, state, rank, ranks, [&] {
            const auto exchange = [&](auto values) { reduction.exchange(values); };
            const auto solve = [&](auto rows, auto correction) {
                reduction.solve(rows, correction);
            };
            return newton(state, exchange, solve, mpi_max);
        });
    }
}

} // namespace
} // namespace boundary

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        boundary::execute(boundary::parse(argc, argv), rank, ranks);
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
}
