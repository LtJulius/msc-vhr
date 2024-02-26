#version 460 core

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec4 in_position;
layout (location = 1) in vec3 in_color;

layout (location = 0) out vec3 out_color;
layout (location = 1) flat out uint out_vertex_index;

void main() {
	out_color = in_color;
	gl_Position = vec4(in_position.xy * 2.0 - 1.0, in_position.z, in_position.w);
	out_vertex_index = gl_VertexIndex;
}