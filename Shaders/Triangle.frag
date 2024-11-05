#version 460
#extension GL_ARB_shading_language_include : require

layout (location = 0) in vec3 inVertPosition;
layout (location = 1) in vec3 inVertNormal;
layout (location = 2) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

#include "Include/Structures.glsl"
layout (set = 0, binding = 0) uniform SceneDataUBO{
	SceneData data;
} ubo;

layout (set = 1, binding = 0) uniform sampler2D tex0;


void main() 
{
	// vec3 objectColor = texture(tex0, inUV).xyz;
	vec3 objectColor = vec3(0.8, 0.8, 0.8);

	vec3 result = BlinnPhong(ubo.data.sunLightPos.xyz, ubo.data.cameraPos.xyz, 
		inVertPosition, inVertNormal, ubo.data.sunLightColor.xyz, objectColor);

	outFragColor = vec4(result, 1.0);
}