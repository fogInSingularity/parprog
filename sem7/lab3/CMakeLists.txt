cmake_minimum_required(VERSION 3.21)

project(boundary_lab LANGUAGES CXX)

set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_subdirectory(../../flags flags)

add_subdirectory(boundary_core)

add_executable(boundary_lab)

target_sources(boundary_lab
    PRIVATE
        src/main.cpp
        src/distributed.cpp
)

target_include_directories(boundary_lab
    PRIVATE
        include/
)

target_link_libraries(boundary_lab
    PRIVATE
        boundary_core
        MPI::MPI_CXX
)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
