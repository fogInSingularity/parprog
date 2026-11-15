#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include "config.hpp"

int main() {
    const double tau = T / K;
    const double h = X / M;
    const double cfl = A * tau / h;
    if (cfl > 1) { return -1; }

    std::vector<std::vector<double>> grid(K + 1, std::vector<double>(M + 1, 0.0));

    // u(0, x_m) = phi(x_m)
    for (size_t m = 0; m <= M; m++) {
        grid[0][m] = phi(m * h);
    }

    for (size_t k = 0; k < K; k++) {
        const double tk = k * tau;

        // граничное u(t_{k+1}, 0) = psi(t_{k+1})
        grid[k + 1][0] = psi((k + 1) * tau);

        for (size_t m = 1; m <= M; ++m) {
            grid[k + 1][m] = 
                grid[k][m]
                - cfl * (grid[k][m] - grid[k][m - 1])
                + tau * f(tk, m * h);
        }
    }

    std::ofstream csv("result.csv");
    csv << "t,x,u\n";
    for (size_t k = 0; k <= K; ++k) {
        for (size_t m = 0; m <= M; ++m) {
            csv << k * tau << ',' << m * h << ',' << grid[k][m] << '\n';
        }
    }
}
