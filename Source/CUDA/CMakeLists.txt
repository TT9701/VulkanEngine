set(CMAKE_CUDA_STANDARD 20)

set (CUDA_SRC 
    simpleCUDA.h
    simpleCUDA.cu)

add_library(CUDA ${CUDA_SRC})