#pragma once

#include <vector>
#include <string>
#include <cstdint>

#define CISDI_3DModel_Subfix_Str ".cisdi"
#define CISDI_3DModel_Subfix_WStr L".cisdi"

// #ifdef USING_NVIDIA_GPU
#define MESHLET_MAX_VERTEX_COUNT 64
#define MESHLET_MAX_TRIANGLE_COUNT 124

// #endif

namespace IntelliDesign_NS::ModelData {

template <class T>
using Type_STLVector = ::std::pmr::vector<T>;
using Type_STLString = ::std::string;

template <class T, uint32_t Dim>
struct Vec {
    T elem[Dim];

    T& operator[](uint32_t idx) { return elem[idx]; }
};

using Float2 = Vec<float, 2>;
using Float3 = Vec<float, 3>;
using Float4 = Vec<float, 4>;

struct Version {
    uint32_t major : 8;
    uint32_t minor : 8;
    uint32_t patch : 16;
};

}