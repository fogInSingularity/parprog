add_executable(boundary_tests)

target_sources(boundary_tests
    PRIVATE
        numerics.cpp
)

target_link_libraries(boundary_tests
    PRIVATE
        boundary_core
)

add_test(NAME lab3_numerics COMMAND boundary_tests)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME lab3_mpi
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/mpi_check.py"
            "${MPIEXEC_EXECUTABLE}" "$<TARGET_FILE:boundary_lab>"
)
set_tests_properties(lab3_mpi PROPERTIES TIMEOUT 90)
