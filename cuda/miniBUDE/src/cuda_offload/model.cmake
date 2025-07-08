register_flag_required(CMAKE_CXX_COMPILER
        "Absolute path to the AMD HIP C++ compiler")

macro(setup)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=native -foffload-via-llvm -O3 -fgpu-rdc" CACHE STRING "Clang++ compiler flags" FORCE)
endmacro()
