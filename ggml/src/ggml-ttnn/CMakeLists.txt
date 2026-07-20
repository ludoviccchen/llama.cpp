message(STATUS "Using TT-Metalium (TTNN) backend")

# TT-Metalium ships a CMake package config under <build>/lib/cmake/tt-metalium
# (a second, broken copy also sits at <build>/tt-metalium-config.cmake with no
# sibling Metalium.cmake - point at the real one explicitly rather than let
# CMAKE_PREFIX_PATH search order pick the wrong one). TT_METAL_HOME is the
# same env var tt-metal's own docs and this project's environment assume;
# skip this if the caller already supplied their own TT-Metalium_DIR.
if (DEFINED ENV{TT_METAL_HOME})
    if (NOT DEFINED TT-Metalium_DIR)
        set(TT-Metalium_DIR "$ENV{TT_METAL_HOME}/build_Release/lib/cmake/tt-metalium")
    endif()
    # Same broken-duplicate-config issue as TT-Metalium above: a second,
    # broken tt-nn-config.cmake with no sibling TT-NN.cmake sits at the
    # build tree root.
    if (NOT DEFINED tt-nn_DIR)
        set(tt-nn_DIR "$ENV{TT_METAL_HOME}/build_Release/lib/cmake/tt-nn")
    endif()
    # TT-Metalium's own find_dependency() calls (umd, spdlog, tt-logger, fmt, ...)
    # need this too - those packages only exist inside the tt-metal build tree.
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TT_METAL_HOME}/build_Release")
endif()

find_package(TT-Metalium REQUIRED)
find_package(tt-nn REQUIRED)

ggml_add_backend_library(ggml-ttnn
                         ggml-ttnn.cpp
                        )

target_link_libraries(ggml-ttnn PRIVATE TT::Metalium TTNN::TTNN)
