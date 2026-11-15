cmake_minimum_required(VERSION 3.21)

project(lab2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # to generate compile_commands.json

find_package(MPI REQUIRED)

add_subdirectory(flags)
add_subdirectory(tp)

add_executable(lab2.x)

target_sources(lab2.x
    PRIVATE
        main.c
)

target_include_directories(lab2.x
    PRIVATE
        include/
)

target_link_libraries(lab2.x
    PRIVATE 
        par-flags
        thread_pool
        m
)
