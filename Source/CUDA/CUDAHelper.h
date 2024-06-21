#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "device_launch_parameters.h"

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                                \
    {                                                                 \
        if (a == NULL) {                                              \
            printf("Host memory failed in %s at line %d\n", __FILE__, \
                   __LINE__);                                         \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    }

template <typename T>
    requires(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16)
struct TexelDataType {
    template <int ByteSize>
    struct DataTypeBinder;

    template <>
    struct DataTypeBinder<4> {
        using Type = float;
    };

    template <>
    struct DataTypeBinder<8> {
        using Type = float2;
    };

    template <>
    struct DataTypeBinder<16> {
        using Type = float4;
    };

    using Type = typename DataTypeBinder<sizeof(T)>::Type;
};

template <typename T>
using CudaTexelTypeBinder_t = typename TexelDataType<T>::Type;