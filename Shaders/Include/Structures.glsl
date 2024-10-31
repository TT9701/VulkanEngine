
struct SceneData
{
	vec4 sunLightPos;
    vec4 sunLightColor;
	vec4 cameraPos;
	mat4 view;
	mat4 proj;
	mat4 viewproj;
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