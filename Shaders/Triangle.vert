#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_shading_language_include : require
#extension GL_EXT_nonuniform_qualifier : require
// #extension ARB_shader_draw_parameters : require

layout (location = 0) out vec3 outVertPosition;
layout (location = 1) out vec3 outVertNormal;
layout (location = 2) out vec2 outUV;

#include "Include/Structures.glsl"

layout (set = 0, binding = 0) uniform SceneDataUBO{
	SceneData data;
} ubo;

layout(buffer_reference, std430) readonly buffer VertexPosBuffer{ 
	vec4 position[];
};

layout(buffer_reference, std430) readonly buffer VertexNormBuffer{ 
	vec2 normal[];
};

layout(buffer_reference, std430) readonly buffer VertexTexBuffer{ 
	vec2 texcoords[];
};

layout(buffer_reference, std430) readonly buffer IndexBuffer{ 
	uint idx[];
};

layout(buffer_reference, std430) readonly buffer VertOffsetBuffer{ 
	uint offset[];
};

layout( push_constant ) uniform constants
{	
	mat4 modelMatrix;
	VertexPosBuffer posBuffer;
	VertexNormBuffer normBuffer;
	VertexTexBuffer texBuffer;

	IndexBuffer idxBuffer;
	VertOffsetBuffer offsetBuffer;
} PushConstants;

void main() 
{

	uint idx = PushConstants.offsetBuffer.offset[gl_DrawID] + PushConstants.idxBuffer.idx[gl_VertexIndex];

	vec4 pos = PushConstants.posBuffer.position[idx];
	vec2 encodedNorm = PushConstants.normBuffer.normal[idx];

	vec4 worldSpacePosition = PushConstants.modelMatrix * vec4(pos.xyz, 1.0f);
	outVertPosition = worldSpacePosition.xyz;

	vec3 normal = DecodeNormal(encodedNorm);
	outVertNormal = mat3(transpose(inverse(PushConstants.modelMatrix))) * normal;

	gl_Position = ubo.data.viewproj * worldSpacePosition;

    outUV = PushConstants.texBuffer.texcoords[idx];
}