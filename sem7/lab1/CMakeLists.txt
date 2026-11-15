cmake_minimum_required(VERSION 3.21)
project(matrix_lab VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)
find_package(hwy 1.3 CONFIG REQUIRED)

option(MATRIX_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(MATRIX_THREAD_SANITIZER "Enable thread sanitizer (separate build)" OFF)
if(MATRIX_SANITIZERS AND MATRIX_THREAD_SANITIZER)
    message(FATAL_ERROR "Use separate builds for ASan and TSan")
endif()

add_library(matrix_core src/matrix.cpp src/algorithms.cpp src/simd.cpp src/thread_pool.cpp)
target_include_directories(matrix_core PUBLIC include PRIVATE src)
target_link_libraries(matrix_core PUBLIC hwy::hwy Threads::Threads)
target_compile_options(matrix_core PRIVATE -Wall -Wextra -Wpedantic)

# Keep the scalar control versions scalar; explicit Highway vectors still work.
# Do not inherit sem6's -march=native: Highway handles CPU dispatch itself.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(matrix_core PRIVATE -fno-tree-vectorize -ffp-contract=off)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(matrix_core PRIVATE -fno-vectorize -fno-slp-vectorize -ffp-contract=off)
endif()

if(MATRIX_SANITIZERS)
    target_compile_options(matrix_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(matrix_core PUBLIC -fsanitize=address,undefined)
elseif(MATRIX_THREAD_SANITIZER)
    target_compile_options(matrix_core PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(matrix_core PUBLIC -fsanitize=thread)
endif()

add_executable(matrix_lab src/main.cpp)
target_link_libraries(matrix_lab PRIVATE matrix_core)
target_compile_options(matrix_lab PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
    add_executable(matrix_tests tests/matrix_tests.cpp)
    target_link_libraries(matrix_tests PRIVATE matrix_core)
    target_compile_options(matrix_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME matrix_correctness COMMAND matrix_tests)
    add_test(NAME admission_small COMMAND matrix_lab admission --size 17)
    add_test(NAME benchmark_small COMMAND matrix_lab bench --sizes 7,17 --threads 1,2
        --block 5 --cutoff 8 --repeats 2 --output smoke.csv)
    add_test(NAME reject_zero_block COMMAND matrix_lab bench --block 0)
    set_tests_properties(reject_zero_block PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(matrix_correctness admission_small benchmark_small PROPERTIES TIMEOUT 90)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        add_test(NAME analysis_formulas COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_plot.py)
    endif()
endif()
