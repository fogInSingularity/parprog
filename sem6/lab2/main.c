#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdatomic.h>
#include <threads.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "thread_pool.h"

#define N_THREADS    8
#define N_CHUNKS     (1ull << 23)
#define H_BASE       1e-6
#define TOLERANCE    1e-9

typedef struct {
    double a;
    double b;
    double result; // worker fills
} Chunk;

static void integrate_chunk(void* arg) {
    Chunk* chunk = (Chunk*)arg;

    double sum = 0.0;
    double x   = chunk->a;

    while (x < chunk->b) {
        double h = H_BASE * x * x;
        if (h < TOLERANCE) h = TOLERANCE;

        double x_next = x + h;
        if (x_next > chunk->b) x_next = chunk->b;

        double fx      = sin(1.0 / x);
        double fx_next = sin(1.0 / x_next);
        sum += 0.5 * (fx + fx_next) * (x_next - x);

        x = x_next;
    }

    chunk->result = sum;
}

static void make_balanced_chunks(double a, double b, int n, Chunk* chunks) {
    double inv_a = 1.0 / a;
    double inv_b = 1.0 / b;

    for (int k = 0; k <= n; k++) {
        double inv_x = inv_a - (double)k / (double)n * (inv_a - inv_b);
        double x = 1.0 / inv_x;

        if (k > 0) chunks[k - 1].b = x;
        if (k < n) chunks[k].a     = x;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <a> <b>   (0 < a < b)\n", argv[0]);
        return 1;
    }

    double a = atof(argv[1]);
    double b = atof(argv[2]);

    if (a <= 0.0 || b <= a) {
        fprintf(stderr, "Error: need 0 < a < b\n");
        return 1;
    }

    printf("Integrating sin(1/x) from %g to %g\n", a, b);
    printf("Threads: %d,  Chunks: %llu,  H_BASE: %g\n\n",
           N_THREADS, N_CHUNKS, H_BASE);

    Chunk* chunks = calloc(N_CHUNKS, sizeof(Chunk));
    if (!chunks) { perror("calloc"); return 1; }

    make_balanced_chunks(a, b, N_CHUNKS, chunks);

    ThreadPool* pool = ThreadPoolCreate(N_THREADS, N_CHUNKS);

    for (size_t i = 0; i < N_CHUNKS; i++) {
        ThreadPoolError err = ThreadPoolQueueTask(pool, integrate_chunk, &chunks[i]);
        while (err == ThreadPoolError_kTaskQueueFull) {
            struct timespec time_to_sleep = {.tv_sec = 0, .tv_nsec = 1000000};
            thrd_sleep(&time_to_sleep, NULL);
            err = ThreadPoolQueueTask(pool, integrate_chunk, &chunks[i]);
        }
    }

    ThreadPoolDestroy(pool); // == barrier

    double total = 0.0;
    for (size_t i = 0; i < N_CHUNKS; i++) {
        total += chunks[i].result;
    }

    printf("Result:  %.15f\n", total);

    free(chunks);
    return 0;
}
