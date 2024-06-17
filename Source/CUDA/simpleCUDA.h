#pragma once

#include <cuda_runtime.h>

#include "device_launch_parameters.h"

template <typename T>
__global__ void CudaAdd(T* a, T* b, T* sum) {
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    sum[i] = a[i] + b[i];
}

float* MatAdd(float* a, float* b, int length);