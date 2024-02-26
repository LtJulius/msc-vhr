#version 460 core

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float2 : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_KHR_memory_scope_semantics : enable

#define local_group_size 64
layout(local_size_x = local_group_size, local_size_y = 1, local_size_z = 1) in;

struct Vertex {
    float32_t px,py,pz,pw;
    uint32_t c;
};

struct DrawIndirectCommand {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
};

struct IntermediateMetaPayload {
    uint32_t native_primitive_offset;
    uint32_t native_primitive_count;
    /**/
    f32vec2 resolution;
};

layout(scalar, set = 0, binding = 0) readonly buffer VertexData {
    Vertex data[];
} vertices;

layout(std430, set = 0, binding = 1) restrict buffer IntermediateMetaData {
    DrawIndirectCommand inter_draw;
    IntermediateMetaPayload meta;
    uint32_t clusterId;
} intermediate;

layout(scalar, set = 0, binding = 2) writeonly volatile coherent buffer InterVertexData {
    Vertex data[];
} inter_vertices;

/**/

layout(set = 0, binding = 3) restrict buffer DepthBuffer {
	uint64_t data[];
} depth_buffer;

layout(set = 0, binding = 4, r32ui) uniform restrict uimage2D target_image;

/**/

layout(push_constant) uniform readonly Constants {
    uint32_t primitives;
    uint32_t padding;
    f32vec2 resolution;
} consts;

/**/

f32vec3 interScreenBarycentric(f32vec2 v0, f32vec2 v1, f32vec2 v2, f32vec2 p) {

    float32_t idenom = 1.0 / ((v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y));

    float32_t u = (p.x - v2.x) * (v1.y - v2.y) * idenom + (p.y - v2.y) * (v2.x - v1.x) * idenom;
    float32_t v = (p.x - v2.x) * (v2.y - v0.y) * idenom + (p.y - v2.y) * (v0.x - v2.x) * idenom;
    return f32vec3(u, v, 1.0 - u - v);
}

/**/

uint32_t packF32(float32_t val_) {
    return uint32_t(round(clamp(val_, 0.0f, 1.0f) * 2147483647.0f));
}

float32_t unpackF32(uint32_t val_) {
    return float32_t(val_) / 2147483647.0f;
}

/**/

const uint primitive_vertex_count = 3u;
const uint group_num_vertices = uint(local_group_size) * primitive_vertex_count;

/**/

shared uint indirect_draw_vertices;
shared uint sh_vertex_base;
shared uint sh_primitive_count;

/**/

void process_cluster() {

    const uint fwd_base_index = gl_LocalInvocationIndex * 3u + sh_vertex_base;
    Vertex v0 = vertices.data[fwd_base_index + 0];
    Vertex v1 = vertices.data[fwd_base_index + 1];
    Vertex v2 = vertices.data[fwd_base_index + 2];
    
    // Load Transformation
    const mat4x4 mvp = mat4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
    
    // Apply Transformation
    vec4 pp0 = mvp * vec4(v0.px, v0.py, v0.pz, 1.0f);
    vec4 pp1 = mvp * vec4(v1.px, v1.py, v1.pz, 1.0f);
    vec4 pp2 = mvp * vec4(v2.px, v2.py, v2.pz, 1.0f);
    
    // Drop triangles behind camera
    if(max(pp0.z, max(pp1.z, pp2.z)) < 0.0) {
        return;
    }
    
    // Transform positions to NDC
    vec4 vp0 = f32vec4(pp0.xy / pp0.w, pp0.z, 1.0f / pp0.w);
    vec4 vp1 = f32vec4(pp1.xy / pp1.w, pp1.z, 1.0f / pp1.w);
    vec4 vp2 = f32vec4(pp2.xy / pp2.w, pp2.z, 1.0f / pp2.w);
    
    // Transform clip space to screen space
    
    vp0.xy *= consts.resolution;
    vp1.xy *= consts.resolution;
    vp2.xy *= consts.resolution;
    
    // Determine Edge Equations
    
    vec3 edge10 = vp1.xyz - vp0.xyz;
    vec3 edge20 = vp2.xyz - vp0.xyz;
    vec3 ed = cross(edge10, edge20);
    
    // Counter-Clockwise Backface-Culling
    if(ed.z < 0.0) {
        return;
    }
    
    // Determine pixel AABB
    vec2 aabbMin = min(vp0.xy, min(vp1.xy, vp2.xy));
    vec2 aabbMax = max(vp0.xy, max(vp1.xy, vp2.xy));
    
    if (aabbMax.x < 0.0f || aabbMax.y < 0.0f || aabbMin.x >= consts.resolution.x || aabbMin.y >= consts.resolution.y) {
        return;
    }
    
    // Clip candiate AABB
    u32vec2 uiAabbMin = u32vec2(aabbMin) + u32vec2((fract(aabbMin.x) <= 0.5f ? 0u : 1u), (fract(aabbMin.y) <= 0.5f ? 0u : 1u));
    u32vec2 uiAabbMax = u32vec2(aabbMax) - u32vec2((fract(aabbMax.x) <= 0.5f ? 1u : 0u), (fract(aabbMax.y) <= 0.5f ? 1u : 0u));
    
    aabbMin = f32vec2(min(max(uiAabbMin, u32vec2(0, 0)), consts.resolution - 1.0f)) + 0.5f;
    aabbMax = f32vec2(min(max(uiAabbMax, u32vec2(0, 0)), consts.resolution - 1.0f)) + 0.5f;
    
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
    
    /**/
    
    const vec2 stride = (aabbMax - aabbMin);
    const float area = stride.x * stride.y;

    if (area <= 0) {
        return;
    }

    /**/

    const uint base_index = gl_LocalInvocationIndex * 3u + sh_vertex_base;
    u32vec3 pc = u32vec3(vertices.data[base_index].c, vertices.data[base_index + 1].c, vertices.data[base_index + 2].c);

    /**/

    [[branch]] if (area >= 4.0) {

        const uint32_t inter_base = atomicAdd(intermediate.inter_draw.vertexCount, 3);

        inter_vertices.data[inter_base + 0].c = pc.x;
        inter_vertices.data[inter_base + 1].c = pc.y;
        inter_vertices.data[inter_base + 2].c = pc.z;
        
        inter_vertices.data[inter_base + 0].px = pp0.x;
        inter_vertices.data[inter_base + 0].py = pp0.y;
        inter_vertices.data[inter_base + 0].pz = pp0.z;
        inter_vertices.data[inter_base + 0].pw = pp0.w;
        
        inter_vertices.data[inter_base + 1].px = pp1.x;
        inter_vertices.data[inter_base + 1].py = pp1.y;
        inter_vertices.data[inter_base + 1].pz = pp1.z;
        inter_vertices.data[inter_base + 1].pw = pp1.w;
        
        inter_vertices.data[inter_base + 2].px = pp2.x;
        inter_vertices.data[inter_base + 2].py = pp2.y;
        inter_vertices.data[inter_base + 2].pz = pp2.z;
        inter_vertices.data[inter_base + 2].pw = pp2.w;

        return;
    }

    /**/

    for(float32_t y = aabbMin.y; y <= aabbMax.y; ++y) {
        for(float32_t x = aabbMin.x; x <= aabbMax.x; ++x) {
    
            vec3 ssInterBase = interScreenBarycentric(vp0.xy, vp1.xy, vp2.xy, vec2(x, y));
    
            // Only use values inside the triangle
            if (ssInterBase.x < -epsilon || ssInterBase.y < -epsilon || ssInterBase.z < 0.0f) {
                continue;
            }
            if (max(ssInterBase.x, max(ssInterBase.y, ssInterBase.z)) > 1.0f) {
                continue;
            }
    
            vec3 interBase = vec3(ssInterBase.x * vp0.w, ssInterBase.y * vp1.w, ssInterBase.z * vp2.w);
            interBase = interBase / (interBase.x + interBase.y + interBase.z);
    
            float32_t depth = interBase.x * vp0.z + interBase.y * vp1.z + interBase.z * vp2.z;
    
            // Drop pixel behind camera
            if(depth < 0.0f || depth >= 1.0f) {
                continue;
            }

            /**/

            u16vec2 tuv = u16vec2(x, y);
            const uint32_t location = tuv.x + uint32_t(tuv.y * consts.resolution.x);

            uint64_t packed = (uint64_t(packF32(depth)) << 33) | uint64_t((base_index / 3) << 1);
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

void main() {

    [[branch]] if ( gl_LocalInvocationIndex == 0u ) {
        indirect_draw_vertices = 0u;
        intermediate.meta.resolution = consts.resolution;

        const uint32_t prev_cluster = atomicAdd(intermediate.clusterId, 1u);
        sh_primitive_count = (consts.primitives > prev_cluster * local_group_size) ? min(consts.primitives - (prev_cluster * local_group_size), local_group_size) : 0u;
        sh_vertex_base = prev_cluster * local_group_size * primitive_vertex_count;
    }

    /**/

    memoryBarrierShared();
    barrier();

    /**/

    while (sh_primitive_count > 0u) {

        if (gl_LocalInvocationIndex < sh_primitive_count) {

            /**/

            process_cluster();

            /**/

        }

        /**/

        memoryBarrierShared();
        barrier();

        /**/

        [[branch]] if (gl_LocalInvocationIndex == 0u) {
            const uint32_t prev_cluster = atomicAdd(intermediate.clusterId, 1u);
            sh_primitive_count = (consts.primitives > prev_cluster * local_group_size) ? min(consts.primitives - (prev_cluster * local_group_size), local_group_size) : 0u;
            sh_vertex_base = prev_cluster * local_group_size * primitive_vertex_count;
        }

        /**/

        memoryBarrierShared();
        barrier();
    }

    /**/

    if (gl_LocalInvocationIndex == 0u) {
        atomicMax(intermediate.inter_draw.instanceCount, intermediate.inter_draw.vertexCount > 0 ? 1u : 0u);
    }
}