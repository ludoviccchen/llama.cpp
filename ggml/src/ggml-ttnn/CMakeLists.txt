message(STATUS "Using TT-Metalium (TTNN) backend")

ggml_add_backend_library(ggml-ttnn
                         ggml-ttnn.cpp
                        )
