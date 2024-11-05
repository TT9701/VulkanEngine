#version 460
 
#extension GL_ARB_shading_language_include : require

layout (location = 0) in VertexInput {
    vec4 color;
    vec3 normal;
    vec3 pos;
} vertexInput;

layout(location = 0) out vec4 outFragColor;
 
#include "Include/Structures.glsl"

layout (set = 0, binding = 0) uniform UBO 
{
    SceneData data;
} ubo;

void main()
{
    vec3 result = BlinnPhong(ubo.data.sunLightPos.xyz, ubo.data.cameraPos.xyz, 
		vertexInput.pos.xyz, vertexInput.normal.xyz, ubo.data.sunLightColor.xyz, vertexInput.color.xyz);

	  outFragColor = vec4(result, 1.0);
}