include(FetchContent)

FetchContent_Declare(
    thread_pool
        GIT_REPOSITORY https://github.com/fogInSingularity/thread_pool.git
        GIT_TAG main
        GIT_SHALLOW true
)
FetchContent_MakeAvailable(thread_pool)
