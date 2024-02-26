#version 460 core

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float2 : enable
#extension GL_EXT_control_flow_attributes : enable

#define local_group_size 64
layout(local_size_x = local_group_size, local_size_y = 1, local_size_z = 1) in;

struct Vertex {
    float32_t px,py,pz,pw;
    uint32_t c;
};

struct DispatchIndirectCommand {
    uint32_t x,y,z;
};

struct DrawIndexedIndirectCommand {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance; 
};

struct IntermediateMetaPayload {
    uint32_t native_primitive_offset;
    uint32_t native_primitive_count;
    uint32_t virtual_primitive_offset;
    uint32_t virtual_primitive_count;
    /**/
    f32vec2 resolution;
};

layout(scalar, set = 0, binding = 0) readonly buffer VertexData {
    Vertex data[];
} vertices;

layout(std430, set = 0, binding = 1) restrict buffer IntermediateMetaData {
    DispatchIndirectCommand inter_dispatch;
    DrawIndexedIndirectCommand inter_draw;
    IntermediateMetaPayload meta;
} intermediate;

layout(scalar, set = 0, binding = 2) writeonly buffer InterIndexData {
    uint32_t data[];
} inter_indices;

layout(push_constant) uniform readonly Constants {
    uint32_t primitives;
    uint32_t padding;
    f32vec2 resolution;
} consts;

/**/

shared bool indirect_dispatch_primitives;
shared bool indirect_draw_vertices;

/**/

void main() {

    if (gl_LocalInvocationIndex == 0u) {
        indirect_dispatch_primitives = false;
        indirect_draw_vertices = false;
        intermediate.meta.resolution = consts.resolution;
    }

    memoryBarrierShared();
    barrier();

    //uint threadId = gl_WorkGroupID.x * local_group_size + gl_LocalInvocationIndex;
    uint threadId = gl_LocalInvocationIndex + gl_WorkGroupID.x * local_group_size + gl_WorkGroupID.y * gl_NumWorkGroups.x * local_group_size;

    if ( threadId < consts.primitives ) {

        // Load Vertices
        u32vec3 indices = u32vec3(threadId * 3 + 0, threadId * 3 + 1, threadId * 3 + 2);
        Vertex v0 = vertices.data[indices.x];
        Vertex v1 = vertices.data[indices.y];
        Vertex v2 = vertices.data[indices.z];

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

        if (area < 4) {

            const uint inter_base_index = atomicAdd(intermediate.meta.virtual_primitive_count, 1);
            const uint store_index = (consts.primitives * 3u) - (inter_base_index * 3u + 1u);
            indirect_dispatch_primitives = true;

            /**/

            inter_indices.data[store_index - 0] = indices.x;
            inter_indices.data[store_index - 1] = indices.y;
            inter_indices.data[store_index - 2] = indices.z;

        } else {

            const uint inter_base_index = atomicAdd(intermediate.inter_draw.indexCount, 3);
            indirect_draw_vertices = true;

            inter_indices.data[inter_base_index + 0] = indices.x;
            inter_indices.data[inter_base_index + 1] = indices.y;
            inter_indices.data[inter_base_index + 2] = indices.z;
        }
    }

    memoryBarrierShared();

    if (gl_LocalInvocationIndex == 0u && indirect_dispatch_primitives) {
        intermediate.inter_dispatch.z = 1u;
        intermediate.inter_dispatch.y = 1u;
        atomicMax(intermediate.inter_dispatch.x, (intermediate.meta.virtual_primitive_count + 63u) / 64u);
        intermediate.meta.virtual_primitive_offset = 0;
        intermediate.meta.native_primitive_count = consts.primitives;
    }

    if (gl_LocalInvocationIndex == 1u && indirect_draw_vertices) {
        intermediate.inter_draw.instanceCount = 1;
    }
}