#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <iostream>

#include <mpi.h>
#include <mpi_proto.h>

// map reduce pi calculation by random monte carlo method

constexpr int kRoot = 0;

constexpr size_t kNItersPerInst = 100'000'000;

size_t CountPi(uint64_t seed) {
    std::mt19937_64 gen{seed};
    std::uniform_real_distribution<double> rnd{0.0, 1.0};

    size_t count = 0;

    for (size_t i = 0; i < kNItersPerInst; i++) {
        double x = rnd(gen);
        double y = rnd(gen);

        if (x * x + y * y <= 1) {
            count++;
        }
    }

    return count;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<uint64_t> seeds;
    if (rank == kRoot) { 
        seeds.reserve(size);

        std::random_device rdev{};
        for (int i = 0; i < size; i++) {
            seeds.push_back(rdev());
        }
    }

    uint64_t seed = 0;
    MPI_Scatter(
        seeds.data(), 
        1, 
        MPI_UINT64_T, 
        &seed, 
        1, 
        MPI_UINT64_T, 
        kRoot, 
        MPI_COMM_WORLD
    );

    size_t count = CountPi(seed);
    size_t result = 0; // valid for root;
    MPI_Reduce(&count, &result, 1, MPI_UINT64_T, MPI_SUM, kRoot, MPI_COMM_WORLD);

    if (rank == kRoot) {
        std::cout << 4 * static_cast<double>(result) / (size * kNItersPerInst) << "\n";
    }

    MPI_Finalize();

    return 0;
}
