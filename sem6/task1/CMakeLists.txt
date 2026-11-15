cmake_minimum_required(VERSION 3.21)

project(task1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # to generate compile_commands.json

find_package(MPI REQUIRED)

add_subdirectory(flags)

add_executable(task1.x)

target_sources(task1.x
    PRIVATE
        main.cpp
)

target_include_directories(task1.x
    PRIVATE
        include/
)

target_link_libraries(task1.x
    PRIVATE 
        par-flags
        MPI::MPI_C
)
