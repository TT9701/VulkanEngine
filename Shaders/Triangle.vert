#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_shading_language_include : require

layout (location = 0) out vec3 outVertPosition;
layout (location = 1) out vec3 outVertNormal;
layout (location = 2) out vec2 outUV;

struct Vertex {
	vec4 position;
	vec4 normal;
	vec2 texcoords;
	vec2 padding;
	vec4 tangent;
	vec4 bitangent;
}; 

#include "Include/Structures.glsl"

layout (set = 0, binding = 0) uniform SceneDataUBO{
	SceneData data;
} ubo;

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

layout( push_constant ) uniform constants
{	
	mat4 modelMatrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main() 
{
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	vec4 worldSpacePosition = PushConstants.modelMatrix * vec4(v.position.xyz, 1.0f);
	outVertPosition = worldSpacePosition.xyz;
	outVertNormal = mat3(transpose(inverse(PushConstants.modelMatrix))) * v.normal.xyz;

	gl_Position = ubo.data.viewproj * worldSpacePosition;
    outUV = v.texcoords.xy;
}