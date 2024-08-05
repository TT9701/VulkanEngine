#version 460
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

struct Vertex {
	vec4 position;
	vec4 normal;
	vec4 color;
	vec2 texcoords;
	vec2 padding;
	vec4 tangent;
	vec4 bitangent;
}; 

layout (set = 0, binding = 0) uniform SceneDataUBO 
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
} ubo;

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

layout( push_constant ) uniform constants
{	
	mat4 modelMatrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main() 
{
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	gl_Position = ubo.viewproj * PushConstants.modelMatrix * vec4(v.position.xyz, 1.0f);
    outUV = v.texcoords.xy;
}