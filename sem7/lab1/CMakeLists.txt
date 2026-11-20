cmake_minimum_required(VERSION 3.21)

project(matrix_lab LANGUAGES CXX)

add_subdirectory(../../flags flags)

add_subdirectory(matrix_core)

add_executable(matrix_lab)

target_sources(matrix_lab
    PRIVATE
        src/main.cpp
        src/benchmark.cpp
        src/options.cpp
)

target_include_directories(matrix_lab
    PRIVATE
        include/
)

target_link_libraries(matrix_lab
    PRIVATE
        matrix_core
)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
