cmake_minimum_required(VERSION 3.21)

project(task3)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # to generate compile_commands.json

find_package(MPI REQUIRED)

add_subdirectory(flags)

add_executable(task3.x)

target_sources(task3.x
    PRIVATE
        main.c
)

target_include_directories(task3.x
    PRIVATE
        include/
)

target_link_libraries(task3.x
    PRIVATE 
        par-flags
        MPI::MPI_C
)
