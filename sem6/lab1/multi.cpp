#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "config.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double tau = T / K;
    const double h   = X / M;
    const double cfl = A * tau / h;
    if (cfl > 1) { MPI_Finalize(); return -1; }

    const int total_pts = (int)M + 1;
    const int local_m   = total_pts / size + (rank < total_pts % size ? 1 : 0);
    const int m_start   = rank * (total_pts / size) + std::min(rank, (int)total_pts % size);

    std::vector<double> cur(local_m + 1, 0.0);
    std::vector<double> nxt(local_m + 1, 0.0);

    std::vector<int> counts(size), displs(size);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            counts[r] = total_pts / size + (r < total_pts % size ? 1 : 0);
            displs[r] = r * (total_pts / size) + std::min(r, (int)total_pts % size);
        }
    }

    std::vector<double> full_row(rank == 0 ? total_pts : 0);

    std::ofstream csv;
    if (rank == 0) {
        csv.open("result.csv");
        csv << "t,x,u\n";
    }

    for (int i = 1; i <= local_m; ++i) {
        cur[i] = phi((m_start + i - 1) * h);
    }

    MPI_Gatherv(
        &cur[1], local_m, MPI_DOUBLE,
        full_row.data(), counts.data(), displs.data(),
        MPI_DOUBLE, 0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        for (int m = 0; m <= (int)M; ++m) {
            csv << 0.0 << ',' << m * h << ',' << full_row[m] << '\n';
        }
    }

    for (size_t k = 0; k < K; ++k) {
        const double tk     = k * tau;
        const double tk1    = (k + 1) * tau;

        MPI_Request req_recv = MPI_REQUEST_NULL;
        MPI_Request req_send = MPI_REQUEST_NULL;

        if (rank < size - 1) {
            MPI_Isend(&cur[local_m], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req_send);
        }
        if (rank > 0) {
            MPI_Irecv(&cur[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req_recv);
        }

        for (int i = 2; i <= local_m; ++i) {
            nxt[i] = cur[i]
                   - cfl * (cur[i] - cur[i - 1])
                   + tau * f(tk, (m_start + i - 1) * h);
        }

        if (rank > 0) { MPI_Wait(&req_recv, MPI_STATUS_IGNORE); }
        if (rank == 0) {
            nxt[1] = psi(tk1);
        } else {
            nxt[1] = cur[1]
                   - cfl * (cur[1] - cur[0])
                   + tau * f(tk, m_start * h);
        }

        if (rank < size - 1) { MPI_Wait(&req_send, MPI_STATUS_IGNORE); }

        std::swap(cur, nxt);

        MPI_Gatherv(
            &cur[1], local_m, MPI_DOUBLE,
            full_row.data(), counts.data(), displs.data(),
            MPI_DOUBLE, 0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            for (int m = 0; m <= (int)M; ++m) {
                csv << tk1 << ',' << m * h << ',' << full_row[m] << '\n';
            }
        }
    }

    if (rank == 0) { csv.close(); }
    MPI_Finalize();
    return 0;
}
