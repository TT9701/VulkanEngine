#ifndef STRUCTURES_H
#define STRUCTURES_H

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

struct SceneData
{
    vec4 sunLightPos;
    vec4 sunLightColor;
    vec4 cameraPos;
    mat4 view;
    mat4 proj;
    mat4 viewproj;

    vec4 objColor;
    vec4 metallicRoughness;
    int texIndex;
    uint selectedObjectID;
};

const uint ShadingModel_Lambert = 0;
const uint ShadingModel_Phong = 1;

struct Material {
    uint shadingModel;
    float shininess;
    vec2 padding;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec4 emissive;
    vec4 reflection;
    vec4 transparent;
};

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

// https://jcgt.org/published/0003/02/01/
// https://zhuanlan.zhihu.com/p/33905696
vec2 UnitVectorToOctahedron(vec3 v)
{
    vec2 oct = v.xy / (abs(v.x) + abs(v.y) + abs(v.z));
    return v.z > 0.0 ? oct : (1.0 - abs(oct.yx)) * sign(v.xy);
}

vec3 OctahedronToUnitVector(vec2 oct)
{
    vec3 n = vec3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    n.xy = n.z < 0.0 ? (1.0 - abs(n.yx)) * sign(n.xy) : n.xy;
    return normalize(n);
}

vec3 BlinnPhong(vec3 lightDir, vec3 cameraPos, vec3 fragPos, vec3 normal, vec3 lightColor, vec3 objectColor) {
    float diffuseStrength = 0.6;
    float specularStrength = 0.2;
    float ambientStrength = 1.0 - diffuseStrength - specularStrength;

    lightDir = normalize(lightDir);
    normal = normalize(normal);
    vec3 viewDir = normalize(cameraPos - fragPos);
    vec3 halfVec = normalize(lightDir + viewDir);

    float diffuse = diffuseStrength * max(dot(normal, lightDir), 0.0);
    float specular = specularStrength * pow(max(dot(halfVec, normal), 0.0), 32);

    return (vec3(ambientStrength) + (diffuse + specular) * lightColor) * objectColor;
}

vec4 CalculateLight(Material mat, vec3 lightDir, vec3 cameraPos, vec3 fragPos, vec3 normal, vec3 lightColor) {
    lightDir = normalize(lightDir);
    normal = normalize(normal);

    vec3 ambient = mat.ambient.rgb * mat.ambient.w;
    vec3 diffuse = mat.diffuse.rgb * max(dot(normal, lightDir), 0.0) * mat.diffuse.w;
    vec3 specular =
    mat.shadingModel == ShadingModel_Lambert ? vec3(0.0)
    : int(mat.shininess) == 0 ? vec3(0.0)
    : mat.specular.rgb * pow(max(dot(normalize(lightDir + normalize(cameraPos - fragPos)), normal), 0.0), int(mat.shininess)) * mat.specular.w;

    return vec4((ambient + diffuse + specular) * lightColor + mat.emissive.rgb * mat.emissive.w, mat.transparent.a);
}

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------

#endif