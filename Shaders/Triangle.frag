#version 460

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout (set = 1, binding = 0) uniform sampler2D tex0;

void main() 
{
	outFragColor = texture(tex0, inUV);
}