/**
 * @file Math.h
 * @author 
 * @brief 通用数学函数
 * @version 0.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <cmath>

#include "BaseTypes.h"

namespace IntelliDesign_NS::ModelData {

/**
 * @brief 钳制函数
 */
template <class T>
constexpr const T& Clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : ((hi < v) ? hi : v);
}

/**
 * @brief 归一化
 */
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

/**
 * @brief 单位向量转八面体坐标
 * @details 参考：https://jcgt.org/published/0003/02/01/
                https://zhuanlan.zhihu.com/p/33905696
 */
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

/**
 * @brief 八面体坐标转单位向量
 */
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

/**
 * @brief 将 float 值转换为 uint16_t 类型的 unorm16
 */
inline uint16_t PackUnorm16(float v) {
    return static_cast<uint16_t>(round(Clamp(v, 0.0f, 1.0f) * 65535.0f));
}

/**
 * @brief 将 float 值转换为 int16_t 类型的 snorm16
 */
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

/**
 * @brief 纹理坐标钳制
 */
inline Float32_2 ClampTexCoords(Float32_2 texCoords) {
    texCoords.x = Clamp(texCoords.x, 0.0f, 1.0f);
    texCoords.y = Clamp(texCoords.y, 0.0f, 1.0f);
    return texCoords;
}

/**
 * @brief 纹理坐标重复
 */
inline Float32_2 RepeatTexCoords(Float32_2 texCoords) {
    texCoords.x = texCoords.x - std::floor(texCoords.x);
    texCoords.y = texCoords.y - std::floor(texCoords.y);
    return texCoords;
}

/**
 * @brief 纹理坐标镜像帮助函数
 */
inline float Mirror(float v) {
    v = std::abs(v);
    int intPart = static_cast<int>(v);
    float fracPart = v - static_cast<float>(intPart);
    return (intPart % 2 == 0) ? fracPart : 1.0f - fracPart;
}

/**
 * @brief 纹理坐标镜像
 */
inline Float32_2 MirrorTexCoords(Float32_2 texCoords) {
    texCoords.x = Mirror(texCoords.x);
    texCoords.y = Mirror(texCoords.y);
    return texCoords;
}

}  // namespace IntelliDesign_NS::ModelData