#version 460
 
layout (location = 0) in VertexInput {
  vec4 color;
  vec3 normal;
} vertexInput;

layout(location = 0) out vec4 outFragColor;
 

void main()
{
  // vec3 positiveNorm = vertexInput.normal * 0.5 + 0.5;
	outFragColor = vec4(vertexInput.color.xyz /** positiveNorm*/, 1.0);
}