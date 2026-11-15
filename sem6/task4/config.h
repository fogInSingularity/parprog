#ifndef CONFIG_H
#define CONFIG_H

#define DEFAULT_ITER 10000

static inline uint64_t now_ns(void)
{
    struct timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 10'0000'0000ULL + ts.tv_nsec;
}

#endif // CONFIG_H
