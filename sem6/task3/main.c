#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>

#define THRESHOLD 2048
#define MAX_THREADS 16

typedef int (*cmp_fn_t)(const void*, const void*);

typedef struct {
    void* arr;
    void* tmp;
    size_t left;
    size_t right;
    size_t size;
    cmp_fn_t cmp;
    int depth;
} sort_args_t;

static inline void* elem(void* base, size_t idx, size_t size) {
    assert(base != NULL);

    return (char*)base + idx * size;
}

static void merge(
    void* arr, 
    void* tmp,
    size_t left, 
    size_t mid, 
    size_t right,
    size_t size, 
    cmp_fn_t cmp
) {
    size_t i = left;
    size_t j = mid;
    size_t k = left;

    while (i < mid && j < right) {
        if (cmp(elem(arr, i, size), elem(arr, j, size)) <= 0) {
            memcpy(elem(tmp, k, size), elem(arr, i, size), size);
            i++;
        } else {
            memcpy(elem(tmp, k, size), elem(arr, j, size), size);
            j++;
        }
        k++;
    }
    while (i < mid) {
        memcpy(elem(tmp, k++, size), elem(arr, i++, size), size);
    }
    while (j < right) {
        memcpy(elem(tmp, k++, size), elem(arr, j++, size), size);
    }

    memcpy(elem(arr, left, size), elem(tmp, left, size), (right - left) * size);
}

static void merge_sort(
    void* arr, 
    void* tmp,
    size_t left, 
    size_t right,
    size_t size, 
    cmp_fn_t cmp, 
    int depth
);

static void* thread_sort(void* arg) {
    assert(arg != NULL);

    sort_args_t *a = (sort_args_t *)arg;
    merge_sort(a->arr, a->tmp, a->left, a->right, a->size, a->cmp, a->depth);
    return NULL;
}

static void merge_sort(
    void* arr, 
    void* tmp,
    size_t left, 
    size_t right,
    size_t size, 
    cmp_fn_t cmp, 
    int depth
) {
    assert(arr != NULL);
    assert(tmp != NULL);

    size_t n = right - left;

    if (n < 2)
        return;

    if (n <= THRESHOLD) {
        qsort(elem(arr, left, size), n, size, cmp);
        return;
    }

    size_t mid = left + n / 2;

    if (depth > 0) {
        pthread_t tid;
        sort_args_t left_arg = {
            .arr   = arr,
            .tmp   = tmp,
            .left  = left,
            .right = mid,
            .size  = size,
            .cmp   = cmp,
            .depth = depth - 1,
        };
        pthread_create(&tid, NULL, thread_sort, &left_arg);
        merge_sort(arr, tmp, mid, right, size, cmp, depth - 1);
        pthread_join(tid, NULL);
    } else {
        merge_sort(arr, tmp, left, mid,  size, cmp, 0);
        merge_sort(arr, tmp, mid,  right, size, cmp, 0);
    }

    merge(arr, tmp, left, mid, right, size, cmp);
}

void parallel_mergesort(void* arr, size_t n, size_t size, cmp_fn_t cmp) {
    assert(arr != NULL);

    if (n < 2) return;

    void* tmp = malloc(n * size);
    if (!tmp) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    int depth = 0;
    while ((1 << (depth + 1)) <= MAX_THREADS)
        ++depth;

    merge_sort(arr, tmp, 0, n, size, cmp, depth);
    free(tmp);
}

static int cmp_int(const void* a, const void* b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int* make_random_array(size_t n, unsigned seed)
{
    int* arr = (int*)malloc(n * sizeof(int));
    if (!arr) { perror("malloc"); exit(EXIT_FAILURE); }
    srand(seed);
    for (size_t i = 0; i < n; i++)
        arr[i] = rand();
    return arr;
}

static int is_sorted(const int* arr, size_t n)
{
    for (size_t i = 1; i < n; i++)
        if (arr[i] < arr[i - 1]) return 0;
    return 1;
}

int main(int argc, char* argv[])
{
    size_t N = 1 << 20;

    if (argc == 2) {
        char *end;
        long v = strtol(argv[1], &end, 10);
        if (*end != '\0' || v <= 0) {
            fprintf(stderr, "Usage: %s [array_size]\n", argv[0]);
            return EXIT_FAILURE;
        }
        N = (size_t)v;
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [array_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int *arr_pms = make_random_array(N, 42);
    double t0 = now_sec();
    parallel_mergesort(arr_pms, N, sizeof(int), cmp_int);
    double t_pms = now_sec() - t0;
    free(arr_pms);

    int *arr_qs = make_random_array(N, 42);
    t0 = now_sec();
    qsort(arr_qs, N, sizeof(int), cmp_int);
    double t_qs = now_sec() - t0;
    free(arr_qs);

    printf("%.6f %.6f\n", t_pms, t_qs);
    return 0;
}
