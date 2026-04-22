#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#include "config.h"

typedef struct {
    pthread_mutex_t mtx;
    pthread_cond_t  cond_a;
    pthread_cond_t  cond_b;

    int turn;
    int done;

    uint64_t t_ping;
    uint64_t t_pong;
} shared_t;

static void *thread_b(void *arg)
{
    shared_t *s = arg;
    pthread_mutex_lock(&s->mtx);
    while (!s->done) {
        while (s->turn != 1 && !s->done)
            pthread_cond_wait(&s->cond_b, &s->mtx);
        if (s->done) break;

        s->t_pong = now_ns();
        s->turn = 0;
        pthread_cond_signal(&s->cond_a);
    }
    pthread_mutex_unlock(&s->mtx);
    return NULL;
}

int main(int argc, char *argv[])
{
    int iters = (argc > 1) 
        ? atoi(argv[1]) 
        : DEFAULT_ITER;

    if (iters <= 0) 
        iters = DEFAULT_ITER;

    shared_t s = {
        .mtx = PTHREAD_MUTEX_INITIALIZER,
        .cond_a = PTHREAD_COND_INITIALIZER,
        .cond_b = PTHREAD_COND_INITIALIZER,
        .turn = 0,
        .done = 0,
    };

    pthread_t tid = {};
    pthread_create(&tid, NULL, thread_b, &s);

    uint64_t sum_time = 0;

    pthread_mutex_lock(&s.mtx);
    for (int i = 0; i < iters; i++) {
        s.t_ping = now_ns();
        s.turn = 1;
        pthread_cond_signal(&s.cond_b);

        while (s.turn != 0) // pong
            pthread_cond_wait(&s.cond_a, &s.mtx);

        sum_time += (s.t_pong - s.t_ping) / 2;
    }

    s.done = 1;
    pthread_cond_signal(&s.cond_b);
    pthread_mutex_unlock(&s.mtx);

    pthread_join(tid, NULL);

    printf("=== condvar thread latency (%d iterations) ===\n", iters);
    printf("  mean   : %7.2f µs\n", (double)sum_time / iters / 1e3);

    return 0;
}
