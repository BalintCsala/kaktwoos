cmake_minimum_required(VERSION 3.16)
project(Kaktwoos CUDA)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so)
set(CMAKE_CUDA_STANDARD 14)

add_executable(Kaktwoos main.cu)

set_target_properties(
        Kaktwoos
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)