#version 460
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

struct Vertex {
	vec3 position;
	float uvX;
	vec3 normal;
	float uvY;
	vec4 color;
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

	gl_Position = ubo.viewproj * PushConstants.modelMatrix * vec4(v.position, 1.0f);
	outColor = v.color.xyz;
    outUV.x = v.uvX;
    outUV.y = v.uvY;
}