#ifndef MESH_SHADER_PUSH_CONSTANT
#define MESH_SHADER_PUSH_CONSTANT

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

struct UInt16Pos {
    uint16_t x, y, z, w;
};

struct Norm {
    int16_t x, y;
};

struct Tex {
    uint16_t x, y;
};

// **IMPORTANT**
// use vec3 will round up to vec4
// see https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf setion 7.6.2.2, page 146
// use struct to bypass this. (page 147)
layout(buffer_reference, std430) readonly buffer VertexPosBuffer {
    // vec3 position[];
    UInt16Pos position[];
};

layout(buffer_reference, std430) readonly buffer VertexNormBuffer {
    Norm normal[];
};

layout(buffer_reference, std430) readonly buffer VertexTexBuffer {
    Tex texcoords[];
};

struct Meshlet {
    uint vertexOffset;
    uint triangleOffset;
    uint vertexCount;
    uint triangleCount;
};

layout(buffer_reference, std430) readonly buffer MeshletBuffer {
    Meshlet meshlets[];
};

layout(buffer_reference, std430) readonly buffer MeshletTriangleBuffer {
    uint8_t meshletTriangles[];
};

layout(buffer_reference, std430) readonly buffer MeshOffsetBuffer {
    uint meshOffsets[];
};

layout(buffer_reference, std430) readonly buffer MeshletOffsetBuffer {
    uint meshletOffsets[];
};

layout(buffer_reference, std430) readonly buffer MeshletTriangleOffsetBuffer {
    uint meshletTriangleOffsets[];
};

layout(buffer_reference, std430) readonly buffer MeshletCountBuffer {
    uint meshletCounts[];
};

struct Vec3 {
    float x, y, z;
};

struct BoundingBox {
    Vec3 center;
    Vec3 extent;
};

layout(buffer_reference, std430) readonly buffer BoundingBoxBuffer {
    BoundingBox boundingBoxes[];
};

layout(buffer_reference, std430) readonly buffer MeshMaterialIndexBuffer {
    uint materialIndices[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer {
    Material materials[];
};

struct Statistics {
    uint vertexCount;
    uint meshletCount;
    uint triangleCount;
    uint materialCount;
};

layout(buffer_reference, std430) readonly buffer StatsBuffer {

    Statistics stats;
};

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;

    VertexPosBuffer posBuffer;
    VertexNormBuffer normBuffer;
    VertexTexBuffer texBuffer;

    MeshletBuffer meshletBuffer;
    MeshletTriangleBuffer meshletTriangleBuffer;
    MeshOffsetBuffer meshOffsetBuffer;
    MeshletOffsetBuffer meshletOffsetBuffer;
    MeshletTriangleOffsetBuffer meshletTriangleOffsetBuffer;
    MeshletCountBuffer meshletCountBuffer;
    BoundingBoxBuffer boundingBoxBuffer;
    BoundingBoxBuffer meshletBoundingBoxBuffer;

    MeshMaterialIndexBuffer meshMaterialIndexBuffer;
    MaterialBuffer materialBuffer;

    StatsBuffer statsBuffer;
}

constants;

#endif