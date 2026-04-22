cmake_minimum_required(VERSION 3.21)

project(task4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # to generate compile_commands.json

find_package(MPI REQUIRED)

add_subdirectory(flags)

add_executable(pipe.x)

target_sources(pipe.x
    PRIVATE
        pipe.c
)

target_include_directories(pipe.x
    PRIVATE
        include/
)

target_link_libraries(pipe.x
    PRIVATE 
        par-flags
)

add_executable(thread.x)

target_sources(thread.x
    PRIVATE
        thread.c
)

target_include_directories(thread.x
    PRIVATE
        include/
)

target_link_libraries(thread.x
    PRIVATE 
        par-flags
)
