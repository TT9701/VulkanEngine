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

const vec3 objectColor = vec3(0.8, 0.8, 0.8);

void main() 
{
	float ambientStrenth = 0.1;
	vec3 lightPos = ubo.data.sunLightPos.xyz;
	vec3 lightColor = ubo.data.sunLightColor.xyz;
	vec3 cameraPos = ubo.data.cameraPos.xyz;
	vec3 normal = normalize(inVertNormal);
	vec3 lightDir = normalize(lightPos - inVertPosition);
	vec3 viewDir = normalize(cameraPos - inVertPosition);

	// ambient
	vec3 ambient = ambientStrenth * lightColor;

	// diffuse
	vec3 diffuse = max(dot(normal, lightDir), 0.0) * lightColor;

	// specular
	float specularStrength = 0.5;
	vec3 halfVec = normalize(lightDir + viewDir);
	vec3 specular = specularStrength * pow(max(dot(halfVec, normal), 0.0), 32) * lightColor;

	vec3 result = (ambient + diffuse + specular) * objectColor;

	outFragColor = vec4(result, 1.0);
	// outFragColor = vec4(normal, 1.0);
}