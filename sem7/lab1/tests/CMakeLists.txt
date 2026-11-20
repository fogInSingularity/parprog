add_executable(matrix_tests)

target_sources(matrix_tests
    PRIVATE
        matrix_tests.cpp
)

target_link_libraries(matrix_tests
    PRIVATE
        matrix_core
)

add_test(NAME matrix_correctness COMMAND matrix_tests)

add_test(NAME admission_small COMMAND matrix_lab admission --size 17)

add_test(NAME benchmark_small
    COMMAND matrix_lab bench --sizes 7,17 --threads 1,2 --block 5 --cutoff 8
            --repeats 2 --output smoke.csv
)

add_test(NAME reject_zero_block COMMAND matrix_lab bench --block 0)

set_tests_properties(reject_zero_block PROPERTIES WILL_FAIL TRUE)

set_tests_properties(matrix_correctness admission_small benchmark_small PROPERTIES TIMEOUT 90)

find_package(Python3 QUIET COMPONENTS Interpreter)

if(Python3_FOUND)
    add_test(NAME analysis_formulas
        COMMAND Python3::Interpreter -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}"
                -p test_plot.py
    )
endif()
