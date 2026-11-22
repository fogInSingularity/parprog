add_executable(loop_tests)

target_sources(loop_tests
    PRIVATE
        loop_tests.cpp
)

target_link_libraries(loop_tests
    PRIVATE
        loop_core
)

add_test(NAME kernels COMMAND loop_tests)

add_test(NAME local_check COMMAND loop_lab check --rows 7 --cols 11 --workers 1,2,8,32)

add_test(NAME local_bench
    COMMAND loop_lab bench --rows 7 --cols 11 --workers 1,2
            --repeats 2 --warmups 1 --output smoke.csv
)

add_test(NAME reject_zero_rows COMMAND loop_lab check --rows 0)

set_tests_properties(reject_zero_rows PROPERTIES WILL_FAIL TRUE)

set_tests_properties(kernels local_check local_bench PROPERTIES TIMEOUT 90)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(verify_files
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/verify_files.py"
    --bin-dir "${PROJECT_BINARY_DIR}"
)

add_test(NAME output_files_shared COMMAND ${verify_files})

set_tests_properties(output_files_shared PROPERTIES TIMEOUT 90)

add_test(NAME output_files_mpi_1 COMMAND ${verify_files} --ranks 1 --rows 2 --cols 7)

add_test(NAME output_files_mpi_2 COMMAND ${verify_files} --ranks 2 --rows 2 --cols 7)

add_test(NAME output_files_mpi_4 COMMAND ${verify_files} --ranks 4 --rows 2 --cols 7)

add_test(NAME output_files_mpi_uneven COMMAND ${verify_files} --ranks 4 --rows 11 --cols 19)

add_test(NAME output_files_mpi_boundary COMMAND ${verify_files} --ranks 2 --rows 1 --cols 1)

set_tests_properties(
    output_files_mpi_1 output_files_mpi_2 output_files_mpi_4
    output_files_mpi_uneven output_files_mpi_boundary
    PROPERTIES TIMEOUT 90 RUN_SERIAL TRUE
)

add_test(NAME analysis_formulas
    COMMAND Python3::Interpreter -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}"
            -p test_plot.py
)
