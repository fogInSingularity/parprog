#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <stdint.h>

#include "config.h"

static void child_loop(int pipe_in, int pipe_out)
{
    uint64_t ts;
    while (read(pipe_in, &ts, sizeof ts) == sizeof ts) {
        if (write(pipe_out, &ts, sizeof ts) != sizeof ts)
            break;
    }
    _exit(0);
}

int main(int argc, char *argv[])
{
    int iters = (argc > 1) ? atoi(argv[1]) : DEFAULT_ITER;
    if (iters <= 0) iters = DEFAULT_ITER;

    int p2c[2], c2p[2];
    if (pipe(p2c) || pipe(c2p)) { perror("pipe"); return 1; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        close(p2c[1]); close(c2p[0]);
        child_loop(p2c[0], c2p[1]);
    }

    close(p2c[0]); close(c2p[1]);

    uint64_t sum_time = 0;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = now_ns();
        if (write(p2c[1], &t0, sizeof t0) != sizeof t0) break;

        uint64_t echo;
        if (read(c2p[0], &echo, sizeof echo) != sizeof echo) break;

        uint64_t t1 = now_ns();
        sum_time += (t1 - t0) / 2;
    }

    close(p2c[1]); close(c2p[0]);
    waitpid(pid, NULL, 0);

    printf("=== pipe IPC latency (%d iterations) ===\n", iters);
    printf("  mean   : %7.2f µs\n", (double)sum_time / iters / 1e3);

    return 0;
}
