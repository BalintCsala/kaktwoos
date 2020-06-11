cmake_minimum_required(VERSION 3.16)
project(Kaktwoos CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Kaktwoos main.cu)

set_target_properties(
        Kaktwoos
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)