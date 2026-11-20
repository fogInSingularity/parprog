cmake_minimum_required(VERSION 3.21)

project(loop_lab LANGUAGES CXX)

set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_subdirectory(../../flags flags)

add_subdirectory(loop_core)

add_executable(loop_lab)

target_sources(loop_lab
    PRIVATE
        src/main.cpp
)

target_link_libraries(loop_lab
    PRIVATE
        loop_core
)

add_executable(loop_mpi)

target_sources(loop_mpi
    PRIVATE
        src/mpi_main.cpp
)

target_link_libraries(loop_mpi
    PRIVATE
        loop_core
        MPI::MPI_CXX
)

file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mpi-launcher.txt"
    CONTENT "${MPIEXEC_EXECUTABLE}\n"
)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
