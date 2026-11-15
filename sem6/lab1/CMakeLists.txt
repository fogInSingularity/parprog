cmake_minimum_required(VERSION 3.21)

project(lab1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # to generate compile_commands.json

find_package(MPI REQUIRED)

add_subdirectory(flags)

add_executable(lab1s.x)

target_sources(lab1s.x
    PRIVATE
        single.cpp
)

target_include_directories(lab1s.x
    PRIVATE
        include/
)

target_link_libraries(lab1s.x
    PRIVATE 
        par-flags
        MPI::MPI_C
)

add_executable(lab1m.x)

target_sources(lab1m.x
    PRIVATE
        multi.cpp
)

target_include_directories(lab1m.x
    PRIVATE
        include/
)

target_link_libraries(lab1m.x
    PRIVATE 
        par-flags
        MPI::MPI_C
)
