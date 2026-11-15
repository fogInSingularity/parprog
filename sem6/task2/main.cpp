#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ratio>
#include <vector>
#include <iostream>
#include <chrono>

#include <mpi.h>
#include <mpi_proto.h>

constexpr int kRoot = 0;

static int Rank() {
    static bool is_init = false;
    static int rank = 0;
    if (!is_init) {
        is_init = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    return rank;
}

static int NodesSize() {
    static bool is_init = false;
    static int size = 0;
    if (!is_init) {
        is_init = true;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    return size;
}

using MeasureTime = std::chrono::duration<float, std::nano>;

MeasureTime MeasureLatency(int node1, int node2) {
    int rank = Rank();

    constexpr size_t kNIters = 100000;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < kNIters; i++) {
        char dummy_data = 0;
        if (rank == node1) {
            MPI_Send(&dummy_data, 1, MPI_CHAR, node2, 0, MPI_COMM_WORLD);
            MPI_Recv(&dummy_data, 1, MPI_CHAR, node2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == node2) { 
            MPI_Recv(&dummy_data, 1, MPI_CHAR, node1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&dummy_data, 1, MPI_CHAR, node1, 0, MPI_COMM_WORLD);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return (end - start) / (2 * kNIters);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (NodesSize() < 2) {
        std::cerr << "size should be at least 2";
        return EXIT_FAILURE;
    }

    auto lat = MeasureLatency(0, 1);
    if (Rank() == 0) {
        std::cout << "Latency: " << lat << "\n";
    }

    MPI_Finalize();

    return 0;
}
