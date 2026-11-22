find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(loop_core)

target_sources(loop_core
    PRIVATE
        src/benchmark.cpp
        src/data.cpp
        src/kernels.cpp
        src/options.cpp
)

target_include_directories(loop_core
    PUBLIC
        include/
)

target_link_libraries(loop_core
    PUBLIC
        par-common-flags
        Threads::Threads
    PRIVATE
        OpenMP::OpenMP_CXX
)
