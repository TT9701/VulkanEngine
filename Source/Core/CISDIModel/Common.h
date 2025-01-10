#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

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
struct Vec;

template <class T>
struct Vec<T, 2> {
    Vec(T x = (T)0, T y = (T)0) : x(x), y(y) {}

    T x, y;

    T& operator[](uint32_t idx) { return (&x)[idx]; }
};

template <class T>
struct Vec<T, 3> {
    Vec(T x = (T)0, T y = (T)0, T z = (T)0) : x(x), y(y), z(z) {}

    T x, y, z;

    T& operator[](uint32_t idx) { return (&x)[idx]; }
};

template <class T>
struct Vec<T, 4> {
    Vec(T x = (T)0, T y = (T)0, T z = (T)0, T w = (T)0)
        : x(x), y(y), z(z), w(w) {}

    T x, y, z, w;

    T& operator[](uint32_t idx) { return (&x)[idx]; }
};

using Float32_2 = Vec<float, 2>;
using Float32_3 = Vec<float, 3>;
using Float32_4 = Vec<float, 4>;

using UInt16_2 = Vec<uint16_t, 2>;
using UInt16_3 = Vec<uint16_t, 3>;
using Int16_2 = Vec<int16_t, 2>;
using Int16_3 = Vec<int16_t, 3>;

struct Version {
    uint32_t major : 8;
    uint32_t minor : 8;
    uint32_t patch : 16;
};

struct AABoundingBox {
    AABoundingBox()
        : min(FLT_MAX, FLT_MAX, FLT_MAX), max(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}

    Float32_3 min;
    Float32_3 max;
};

template <class T>
constexpr const T& Clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : ((hi < v) ? hi : v);
}

inline Float32_3 Normalize(Float32_3 v) {
    float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0.0f) {
        float invLength = 1.0f / length;
        v.x *= invLength;
        v.y *= invLength;
        v.z *= invLength;
    }
    return v;
}

// https://jcgt.org/published/0003/02/01/
// https://zhuanlan.zhihu.com/p/33905696
inline Float32_2 UnitVectorToOctahedron(Float32_3 n) {
    float absSum = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    float u = n.x / absSum;
    float v = n.y / absSum;
    if (n.z <= 0) {
        float oldU = u;
        u = (1 - std::abs(v)) * (u >= 0 ? 1 : -1);
        v = (1 - std::abs(oldU)) * (v >= 0 ? 1 : -1);
    }
    return {u, v};
}

inline Float32_3 OctahedronToUnitVector(Float32_2 oct) {
    float nx = oct.x;
    float ny = oct.y;
    float nz = 1 - std::abs(oct.x) - std::abs(oct.y);
    if (nz < 0) {
        float oldNx = nx;
        nx = (1 - std::abs(ny)) * (nx >= 0 ? 1 : -1);
        ny = (1 - std::abs(oldNx)) * (ny >= 0 ? 1 : -1);
    }
    return Normalize({nx, ny, nz});
}

inline uint16_t PackUnorm16(float v) {
    return static_cast<uint16_t>(round(Clamp(v, 0.0f, 1.0f) * 65535.0f));
}

inline int16_t PackSnorm16(float v) {
    int16_t const topack =
        static_cast<int16_t>(round(Clamp(v, -1.0f, 1.0f) * 32767.0f));
    int16_t packed = 0;
    memcpy(&packed, &topack, sizeof(packed));
    return packed;
}

inline float UnpackSnorm16(int16_t p) {
    // p / 32767.0f,
    return Clamp(static_cast<float>(p) * 3.0518509475997192297128208258309e-5f,
                 -1.0f, 1.0f);
}

inline Float32_2 ClampTexCoords(Float32_2 texCoords) {
    texCoords.x = Clamp(texCoords.x, 0.0f, 1.0f);
    texCoords.y = Clamp(texCoords.y, 0.0f, 1.0f);
    return texCoords;
}

inline Float32_2 RepeatTexCoords(Float32_2 texCoords) {
    texCoords.x = texCoords.x - std::floor(texCoords.x);
    texCoords.y = texCoords.y - std::floor(texCoords.y);
    return texCoords;
}

inline float Mirror(float v) {
    v = std::abs(v);
    int intPart = static_cast<int>(v);
    float fracPart = v - static_cast<float>(intPart);
    return (intPart % 2 == 0) ? fracPart : 1.0f - fracPart;
}

inline Float32_2 MirrorTexCoords(Float32_2 texCoords) {
    texCoords.x = Mirror(texCoords.x);
    texCoords.y = Mirror(texCoords.y);
    return texCoords;
}

inline void UpdateAABB(AABoundingBox& aabb, Float32_3 pos) {
    aabb.min.x = std::min(aabb.min.x, pos.x);
    aabb.min.y = std::min(aabb.min.y, pos.y);
    aabb.min.z = std::min(aabb.min.z, pos.z);
    aabb.max.x = std::max(aabb.max.x, pos.x);
    aabb.max.y = std::max(aabb.max.y, pos.y);
    aabb.max.z = std::max(aabb.max.z, pos.z);
}

inline void UpdateAABB(AABoundingBox& aabb, AABoundingBox const& other) {
    aabb.min.x = std::min(aabb.min.x, other.min.x);
    aabb.min.y = std::min(aabb.min.y, other.min.y);
    aabb.min.z = std::min(aabb.min.z, other.min.z);
    aabb.max.x = std::max(aabb.max.x, other.max.x);
    aabb.max.y = std::max(aabb.max.y, other.max.y);
    aabb.max.z = std::max(aabb.max.z, other.max.z);
}

inline Float32_3 GetAABBCenter(AABoundingBox const& aabb) {
    return {(aabb.min.x + aabb.max.x) * 0.5f, (aabb.min.y + aabb.max.y) * 0.5f,
            (aabb.min.z + aabb.max.z) * 0.5f};
}

}  // namespace IntelliDesign_NS::ModelData