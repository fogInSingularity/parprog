add_library(boundary_core)

target_sources(boundary_core
    PRIVATE
        src/numerov.cpp
        src/tridiagonal.cpp
)

target_include_directories(boundary_core
    PUBLIC
        include/
)

target_link_libraries(boundary_core
    PUBLIC
        par-common-flags
)
