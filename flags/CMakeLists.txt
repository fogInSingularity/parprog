set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)

add_library(par-common-flags INTERFACE)

target_compile_features(par-common-flags
    INTERFACE
        cxx_std_20
)

target_compile_options(par-common-flags
    INTERFACE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(par-flags INTERFACE)

target_link_libraries(par-flags
    INTERFACE
        par-common-flags
)

target_compile_features(par-flags
    INTERFACE
        c_std_11
)

target_compile_options(par-flags
    INTERFACE
        -fdiagnostics-color=always

        # -Wshadow
        # -Wconversion
        # -Wsign-conversion
        -Wunused
        -Wformat=2

        -fstack-protector-strong # inserts checks for buffer overflow
        # -fPIE -pie

        $<$<CONFIG:Debug>:
            -Og
            -g3
            -ggdb
            -fsanitize=address,leak,undefined
            -fno-omit-frame-pointer
        >

        $<$<CONFIG:Release>:
            -g
            -O2
            -march=native
            # -flto
            -DNDEBUG
            -Wa,-g
        >
)

target_link_options(par-flags
    INTERFACE
        -fdiagnostics-color=always

        -Wall
        -Wextra
        # -fPIE -pie
        -rdynamic

        $<$<CONFIG:Debug>:
            -Og
            -g3
            -ggdb
            -fsanitize=address,leak,undefined
            -fno-omit-frame-pointer
        >

        $<$<CONFIG:Release>:
            -g
            -O2
            -march=native
            # -flto
        >
)
