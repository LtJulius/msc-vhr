#version 460 core

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float2 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_memory_scope_semantics : enable

#define local_group_size 64
layout(local_size_x = local_group_size, local_size_y = 1, local_size_z = 1) in;

struct Vertex {
    float32_t px,py,pz,pw;
    uint32_t c;
};

struct IntermediateMetaPayload {
    uint32_t native_primitive_offset;
    uint32_t native_primitive_count;
    uint32_t virtual_primitive_offset;
    uint32_t virtual_primitive_count;
    /**/
    f32vec2 resolution;
};

layout(std430, set = 0, binding = 0) readonly uniform IntermediateMetaData {
    IntermediateMetaPayload data;
} intermediate;

layout(scalar, set = 0, binding = 1) readonly buffer IndexData {
    uint32_t data[];
} indices;

layout(scalar, set = 0, binding = 2) readonly buffer VertexData {
    Vertex data[];
} vertices;

layout(set = 0, binding = 3) restrict buffer DepthBuffer {
	uint64_t data[];
} depth_buffer;

layout(set = 0, binding = 4, r32ui) uniform restrict uimage2D target_image;

/**/

f32vec3 interScreenBarycentric(f32vec2 v0, f32vec2 v1, f32vec2 v2, f32vec2 p) {

    float32_t idenom = 1.0 / ((v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y));

    float32_t u = (p.x - v2.x) * (v1.y - v2.y) * idenom + (p.y - v2.y) * (v2.x - v1.x) * idenom;
    float32_t v = (p.x - v2.x) * (v2.y - v0.y) * idenom + (p.y - v2.y) * (v0.x - v2.x) * idenom;
    return f32vec3(u, v, 1.0 - u - v);
}

uint32_t packF32(float32_t val_) {
    return uint32_t(round(clamp(val_, 0.0, 1.0) * 2147483647.0));
}

float32_t unpackF32(uint32_t val_) {
    return float32_t(val_) / 2147483647.0;
}

/**/

const uint primitive_vertex_count = 3u;
const uint group_num_vertices = uint(local_group_size) * primitive_vertex_count;

shared uint shPrimitiveCount;
shared uint shVertexCount;
shared uint shIndexBase;

shared vec4 shVertexPosition[group_num_vertices];

/**/

void rasterize() {

    vec3 v0 = shVertexPosition[(gl_LocalInvocationIndex * primitive_vertex_count + 0u)].xyz;
    vec3 v1 = shVertexPosition[(gl_LocalInvocationIndex * primitive_vertex_count + 1u)].xyz;
    vec3 v2 = shVertexPosition[(gl_LocalInvocationIndex * primitive_vertex_count + 2u)].xyz;
    
    if(max(v0.z, max(v1.z, v2.z)) < 0.0) {
        return;
    }

    // Determine Edge Equations

    vec3 edge10 = v1.xyz - v0.xyz;
    vec3 edge20 = v2.xyz - v0.xyz;
    vec3 ed = cross(edge10, edge20);

    // Counter-Clockwise Backface-Culling

    [[branch]] if (ed.z < 0.0) {
        return;
    }

    // Determine pixel AABB

    vec2 aabbMin = min(v0.xy, min(v1.xy, v2.xy));
    vec2 aabbMax = max(v0.xy, max(v1.xy, v2.xy));

    [[branch]] if (aabbMax.x < 0.0f || aabbMax.y < 0.0f || aabbMin.x >= intermediate.data.resolution.x || aabbMin.y >= intermediate.data.resolution.y) {
        return;
    }

    // Clip candiate AABB
    uvec2 uiAabbMin = uvec2(aabbMin) + uvec2((fract(aabbMin.x) <= 0.5f ? 0u : 1u), (fract(aabbMin.y) <= 0.5f ? 0u : 1u));
    uvec2 uiAabbMax = uvec2(aabbMax) - uvec2((fract(aabbMax.x) <= 0.5f ? 1u : 0u), (fract(aabbMax.y) <= 0.5f ? 1u : 0u));

    aabbMin = clamp(uiAabbMin, uvec2(0), intermediate.data.resolution - 1.0f) + 0.5f;
    aabbMax = clamp(uiAabbMax, uvec2(0), intermediate.data.resolution - 1.0f) + 0.5f;

    // Cull elements between pixel-probes
    if ((fract(aabbMin.x) > 0.5f) && (fract(aabbMax.x) <= 0.5f) && ((aabbMax.x - aabbMin.x) <= 0.0f)) {
        return;
    }
    
    if ((fract(aabbMin.y) > 0.5f) && (fract(aabbMax.y) <= 0.5f) && ((aabbMax.y - aabbMin.y) <= 0.0f)) {
        return;
    }
    
    #define epsilon 0.00005f
    aabbMin -= vec2(epsilon);
    aabbMax += vec2(epsilon);

    //

    const u32vec3 vaddress = u32vec3(
        indices.data[shIndexBase - (gl_LocalInvocationIndex * 3u + 1u)],
        indices.data[shIndexBase - (gl_LocalInvocationIndex * 3u + 2u)],
        indices.data[shIndexBase - (gl_LocalInvocationIndex * 3u + 3u)]
    );
    const u32vec3 pc = u32vec3(vertices.data[vaddress.x].c, vertices.data[vaddress.y].c, vertices.data[vaddress.z].c);
    const vec3 pw = vec3(
        shVertexPosition[gl_LocalInvocationIndex * 3u + 0u].w,
        shVertexPosition[gl_LocalInvocationIndex * 3u + 1u].w,
        shVertexPosition[gl_LocalInvocationIndex * 3u + 2u].w
    );

    for(float32_t y = aabbMin.y; y <= aabbMax.y; ++y) {
        for(float32_t x = aabbMin.x; x <= aabbMax.x; ++x) {
            
            vec3 ssInterBase = interScreenBarycentric(v0.xy, v1.xy, v2.xy, vec2(x, y));

            // Only use values inside the triangle
            if (ssInterBase.x < -epsilon || ssInterBase.y < -epsilon || ssInterBase.z < 0.0f) {
                continue;
            }
            if (max(ssInterBase.x, max(ssInterBase.y, ssInterBase.z)) > 1.0f) {
                continue;
            }

            // vec3 interBase = vec3(ssInterBase.x * v0.w, ssInterBase.y * v1.w, ssInterBase.z * v2.w);
            vec3 interBase = ssInterBase * pw;
            interBase = interBase / (interBase.x + interBase.y + interBase.z);

            float32_t depth = interBase.x * v0.z + interBase.y * v1.z + interBase.z * v2.z;

            // Drop pixel behind camera
            if(depth < 0.0 || depth >= 1.0) {
                continue;
            }
            
            /**/

            u16vec2 tuv = u16vec2(x, y);
            const uint32_t location = tuv.x + uint32_t(tuv.y * intermediate.data.resolution.x);

            uint64_t packed = (uint64_t(packF32(depth)) << 33) | uint64_t((vaddress.x / 3) << 1);
            uint64_t maybe = 0;

            bool loop = true;
            [[loop]] do {
                maybe = atomicOr(depth_buffer.data[location], 0x1, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
                [[branch]] if ((maybe & uint64_t(0x1u)) == uint64_t(0x0u)) {
    
                    if (maybe > packed) {
                        const f32vec3 interColor = (
                            interBase.x * unpackUnorm4x8(pc.x).rgb /* / pp0.w */ +
                            interBase.y * unpackUnorm4x8(pc.y).rgb /* / pp1.w */ +
                            interBase.z * unpackUnorm4x8(pc.z).rgb /* / pp2.w */
                        );
                        imageAtomicExchange(target_image, tuv, pack32(u8vec4(u8vec3(interColor.bgr * 255.0f), 255u)));

                        atomicMin(depth_buffer.data[location], packed);
                    } else {
                        atomicXor(depth_buffer.data[location], 0x1, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
                    }

                    loop = false;

                } else {
                    loop = maybe > packed && (maybe & 0x1) == 0x1;
                }
            } while (loop);

            /**/
        }
    }
}

/**/

void main() {

    const uint groupOffset = gl_WorkGroupID.y * gl_NumWorkGroups.x;
    const uint groupId = gl_WorkGroupID.x + groupOffset;
    const uint localId = gl_LocalInvocationIndex;

    [[branch]] if ( localId == 0u ) {
        shPrimitiveCount = min(intermediate.data.virtual_primitive_count - (groupId * local_group_size), local_group_size /* Max : 1 Primitive ~ 1 Thread */);
        shVertexCount = shPrimitiveCount * primitive_vertex_count/* Triangle : 3 Vertices ~ 3 Indices */;
        shIndexBase = (intermediate.data.native_primitive_count * primitive_vertex_count) - (groupId * local_group_size * primitive_vertex_count);
    }

    memoryBarrierShared();
    barrier();

    [[loop]] for ( uint i = 0; i < group_num_vertices; i += local_group_size ) {
        /*
         * 0 : localId + i * local_grp_size ~ localId
         * 1 : localId + i * local_grp_size ~ localId + 128
         * 2 : localId + i * local_grp_size ~ localId + 256
         */
        uint index = localId + i;
        [[branch]] if ( index < shVertexCount ) {
            
            uint address = indices.data[shIndexBase - (index + 1u)];
            vec4 vp = vec4(vertices.data[address].px, vertices.data[address].py, vertices.data[address].pz, 1.0);

            const mat4x4 mvp = mat4x4(
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            );

            vp = mvp * vp;
            shVertexPosition[index] = vec4(
                (vp.xy / vp.w) * (intermediate.data.resolution),
                vp.z,
                1.0f / vp.w
            );
        }

    }

    memoryBarrierShared();
    barrier();

    if ( localId < shPrimitiveCount ) {
        
        uint index = localId * primitive_vertex_count;
        rasterize();
    }
}