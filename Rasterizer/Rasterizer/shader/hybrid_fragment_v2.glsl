#version 460 core

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_control_flow_attributes : enable

/**/

struct IntermediateMetaPayload {
    uint32_t native_primitive_offset;
    uint32_t native_primitive_count;
    uint32_t virtual_primitive_offset;
    uint32_t virtual_primitive_count;
    /**/
    f32vec2 resolution;
};

/**/

layout (location = 0) in vec3 in_color;
layout (location = 1) flat in uint in_vertex_index;

layout (origin_upper_left) in vec4 gl_FragCoord;

/**/

layout(std430, set = 0, binding = 0) readonly uniform IntermediateMetaData {
    IntermediateMetaPayload data;
} intermediate;

layout(set = 0, binding = 1) restrict volatile coherent buffer DepthBuffer {
	uint64_t data[];
} depth_buffer;

layout(set = 0, binding = 2, r32ui) uniform restrict uimage2D target_image;

/**/

uint32_t packF32(float32_t val_) {
    return uint32_t(round(clamp(val_, 0.0f, 1.0f) * 2147483647.0f));
}

float32_t unpackF32(uint32_t val_) {
    return float32_t(val_) / 2147483647.0f;
}

/**/

void main() {

	const float32_t depth = gl_FragCoord.z;
    const uint32_t location = uint32_t(gl_FragCoord.x) + uint32_t(gl_FragCoord.y) * uint32_t(intermediate.data.resolution.x);

    uint64_t packed = (uint64_t(packF32(depth)) << 33) | uint64_t((in_vertex_index / 3) << 1);
    uint64_t maybe = 0;

    bool loop = true;
    [[loop]] do {
        maybe = atomicOr(depth_buffer.data[location], 0x1, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
        [[branch]] if ((maybe & uint64_t(0x1u)) == uint64_t(0x0u)) {
    
            if (maybe > packed) {
                imageAtomicExchange(target_image, ivec2(gl_FragCoord.xy), pack32(u8vec4(u8vec3(in_color.bgr * 255.0f), 255u)));
                atomicMin(depth_buffer.data[location], packed);
            } else {
                atomicXor(depth_buffer.data[location], 0x1, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
            }

            loop = false;

        } else {
            loop = maybe > packed && (maybe & 0x1) == 0x1;
        }
    } while (loop);
}