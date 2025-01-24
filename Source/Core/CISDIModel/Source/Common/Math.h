#pragma once

#include <cmath>

#include "BaseTypes.h"

namespace IntelliDesign_NS::ModelData {

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

/* Unpack in shader. examples in glsl:
    float UnpackUnorm16(uint16_t value)
    {
	    // value / 65535.0
	    return float(value) * 1.5259021896696421759365224689097e-5;
    }

    float UnpackSnorm16(int16_t value)
    {
	    // value / 32767.0
	    return clamp(
	    float(value) * 3.0518509475997192297128208258309e-5, -1.0, 1.0);
    }
 */

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
    aabb.min.x = aabb.min.x < pos.x ? aabb.min.x : pos.x;
    aabb.min.y = aabb.min.y < pos.y ? aabb.min.y : pos.y;
    aabb.min.z = aabb.min.z < pos.z ? aabb.min.z : pos.z;

    aabb.max.x = aabb.max.x > pos.x ? aabb.max.x : pos.x;
    aabb.max.y = aabb.max.y > pos.y ? aabb.max.y : pos.y;
    aabb.max.z = aabb.max.z > pos.z ? aabb.max.z : pos.z;
}

inline void UpdateAABB(AABoundingBox& aabb, AABoundingBox const& other) {
    aabb.min.x = aabb.min.x < other.min.x ? aabb.min.x : other.min.x;
    aabb.min.y = aabb.min.y < other.min.y ? aabb.min.y : other.min.y;
    aabb.min.z = aabb.min.z < other.min.z ? aabb.min.z : other.min.z;

    aabb.max.x = aabb.max.x > other.max.x ? aabb.max.x : other.max.x;
    aabb.max.y = aabb.max.y > other.max.y ? aabb.max.y : other.max.y;
    aabb.max.z = aabb.max.z > other.max.z ? aabb.max.z : other.max.z;
}

inline Float32_3 GetAABBCenter(AABoundingBox const& aabb) {
    return {(aabb.min.x + aabb.max.x) * 0.5f, (aabb.min.y + aabb.max.y) * 0.5f,
            (aabb.min.z + aabb.max.z) * 0.5f};
}

}  // namespace IntelliDesign_NS::ModelData