#version 460
 
#extension GL_ARB_shading_language_include : require

layout (location = 0) in vec4 InColor;
layout (location = 1) in vec3 InNormal;
layout (location = 2) in vec3 InPos;
layout (location = 3) in vec2 InUV;

layout(location = 0) out vec4 outFragColor;
 
#include "Include/Structures.glsl"

layout (set = 0, binding = 0) uniform UBO 
{
    SceneData data;
} ubo;

void main()
{
    vec3 result = BlinnPhong(ubo.data.sunLightPos.xyz, ubo.data.cameraPos.xyz, 
		InPos.xyz, InNormal.xyz, ubo.data.sunLightColor.xyz, InColor.xyz);

	  outFragColor = vec4(result, 1.0);
}