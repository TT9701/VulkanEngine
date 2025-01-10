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