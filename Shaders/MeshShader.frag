#version 460

#extension GL_ARB_shading_language_include : require
#extension GL_EXT_nonuniform_qualifier : require

layout (location = 0) in vec4 InColor;
layout (location = 1) in vec3 InNormal;
layout (location = 2) in vec3 InPos;
layout (location = 3) in vec2 InUV;
layout (location = 4) in MeshIndex {
    flat uint index;
} InMeshIdx;

layout(location = 0) out vec4 outFragColor;

#include "Include/Structures.glsl"
#include "MeshShaderPushConstant.h"

layout (set = 0, binding = 0) uniform UBO
{
    SceneData data;
} ubo;

layout (set = 1, binding = 0) uniform sampler2D sceneTexs[];

void main()
{
    uint InMaterialIdx = constants.meshMaterialIndexBuffer.materialIndices[InMeshIdx.index];
    Material material = constants.materialBuffer.materials[InMaterialIdx];

    // vec3 texColor = texture(sceneTexs[ubo.data.texIndex], InUV).xyz;

    // vec3 albedo = texColor;
    // vec3 albedo = InColor.xyz;
    // vec3 albedo = material.diffuse.xyz;

    vec3 N = normalize(InNormal.xyz);
    vec3 V = normalize(ubo.data.cameraPos.xyz - InPos.xyz);

    if (dot(N, V) < 0.0){
        N = -N;
    }

    // float metallic = ubo.data.metallicRoughness.x;
    // float roughness = ubo.data.metallicRoughness.y;
    // float ao = 1.0;

    // vec3 F0 = vec3(0.04);
    // F0 = mix(F0, albedo, metallic);

    // vec3 Lo = vec3(0.0);

    // vec3 L = normalize(ubo.data.sunLightPos.xyz);
    // vec3 H = normalize(V + L);
    // float attenuation = 1.0;
    // vec3 radiance = ubo.data.sunLightColor.xyz * attenuation;

    // float NDF = DistributionGGX(N, H, roughness);
    // float G   = GeometrySmith(N, V, L, roughness);
    // vec3 F    = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);

    // vec3 numerator    = NDF * G * F;
    // float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    // vec3 specular = numerator / denominator;

    // vec3 kS = F;
    // vec3 kD = vec3(1.0) - kS;
    // kD *= 1.0 - metallic;

    // float NdotL = max(dot(N, L), 0.0);

    // Lo += (kD * albedo / PI + specular) * radiance * NdotL;

    // vec3 ambient = vec3(0.03) * albedo * ao;
    // vec3 color = ambient + Lo;

    // color = color / (color + vec3(1.0));
    // color = pow(color, vec3(1.0/2.2));

    vec4 color = CalculateLight(material,
    normalize(ubo.data.sunLightPos.xyz),
    ubo.data.cameraPos.xyz,
    InPos.xyz, N,
    ubo.data.sunLightColor.xyz);

    outFragColor = color;
}