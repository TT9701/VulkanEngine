
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

// normal
// pre calculated on convertion of 3d model to cisdi model -> "spheremap transform"
// wikipedia: https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
vec3 DecodeNormal(vec2 enc) {
	float f = dot(enc, enc);
	float g = sqrt(1 - f / 4);
	vec3 n;
	n.yz = enc * g;
	n.x = -1 + f / 2;
	return n;
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