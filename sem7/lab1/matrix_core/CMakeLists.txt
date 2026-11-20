find_package(Threads REQUIRED)

find_package(hwy 1.3 CONFIG REQUIRED)

add_library(matrix_core)

target_sources(matrix_core
    PRIVATE
        src/algorithms.cpp
        src/matrix.cpp
        src/simd.cpp
        src/simd_block.cpp
        src/thread_pool.cpp
)

target_include_directories(matrix_core
    PUBLIC
        include/
    PRIVATE
        src/
)

target_link_libraries(matrix_core
    PUBLIC
        par-common-flags
        hwy::hwy
        Threads::Threads
)
