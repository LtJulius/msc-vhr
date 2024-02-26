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

layout(std430, set = 0, binding = 0) readonly buffer VertexData {
    Vertex data[];
} vertices;

layout(set = 0, binding = 1, r32ui) uniform restrict uimage2D depth_image;
//layout(set = 0, binding = 2, rgba8ui) uniform writeonly uimage2D target_image;
layout(set = 0, binding = 2, r32ui) uniform restrict uimage2D target_image;

layout(push_constant) uniform readonly Constants {
    uint32_t primitives;
    uint32_t padding;
    f32vec2 resolution;
}
consts;

/**/

float32_t perpDotProduct(f32vec2 va, f32vec2 vb) {
    return va.x * vb.y - va.y * vb.x;
}

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

shared uint shPrimitiveCount;
shared uint shVertexCount;
shared uint shVertexBase;

shared vec4 shVertexPosition[group_num_vertices];

/**/

void rasterize(uint i0, uint i1, uint i2) {

    vec4 v0 = shVertexPosition[i0].xyzw;
    vec4 v1 = shVertexPosition[i1].xyzw;
    vec4 v2 = shVertexPosition[i2].xyzw;
    
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

    [[branch]] if (aabbMax.x < 0.0f || aabbMax.y < 0.0f || aabbMin.x >= consts.resolution.x || aabbMin.y >= consts.resolution.y) {
        return;
    }

    // Clip candiate AABB
    uvec2 uiAabbMin = uvec2(aabbMin) + uvec2((fract(aabbMin.x) <= 0.5f ? 0u : 1u), (fract(aabbMin.y) <= 0.5f ? 0u : 1u));
    uvec2 uiAabbMax = uvec2(aabbMax) - uvec2((fract(aabbMax.x) <= 0.5f ? 1u : 0u), (fract(aabbMax.y) <= 0.5f ? 1u : 0u));

    aabbMin = clamp(uiAabbMin, uvec2(0), consts.resolution - 1.0f) + 0.5f;
    aabbMax = clamp(uiAabbMax, uvec2(0), consts.resolution - 1.0f) + 0.5f;

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

    uint32_t c0 = vertices.data[shVertexBase + i0].c;
    uint32_t c1 = vertices.data[shVertexBase + i1].c;
    uint32_t c2 = vertices.data[shVertexBase + i2].c;

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

            vec3 interBase = vec3(ssInterBase.x * v0.w, ssInterBase.y * v1.w, ssInterBase.z * v2.w);
            interBase = interBase / (interBase.x + interBase.y + interBase.z);

            float32_t depth = interBase.x * v0.z + interBase.y * v1.z + interBase.z * v2.z;

            // Drop pixel behind camera
            if(depth < 0.0 || depth >= 1.0) {
                continue;
            }
            
            u16vec2 tuv = u16vec2(x, y);
            const uint32_t stored = imageAtomicMin(depth_image, tuv, packF32(depth));

            if (unpackF32(stored) > depth) {
                f32vec3 interColor = (
                    interBase.x * unpackUnorm4x8(c0).rgb / v0.w +
                    interBase.y * unpackUnorm4x8(c1).rgb / v1.w +
                    interBase.z * unpackUnorm4x8(c2).rgb / v2.w
                );
                
                //imageStore(target_image, ivec2(x, y), uvec4(interColor, 1.0));
                //imageStore(target_image, tuv, u8vec4(vec4(interColor.rgb, 1.0f) * 255.0f));
                //imageAtomicExchange(target_image, tuv, packUnorm4x8(vec4(interColor.rgb, 1.0f)));
                imageAtomicExchange(target_image, tuv, pack32(u8vec4(u8vec3(interColor.bgr * 255.0f), 255u)));
            }
        }
    }
}

/**/

void main() {

    const uint groupOffset = gl_WorkGroupID.y * gl_NumWorkGroups.x;
    const uint groupId = gl_WorkGroupID.x + groupOffset;
    const uint localId = gl_LocalInvocationIndex;

    [[branch]] if ( localId == 0u ) {
        shPrimitiveCount = min(consts.primitives - (groupId * local_group_size), local_group_size /* Max : 1 Primitive ~ 1 Thread */);
        shVertexCount = shPrimitiveCount * primitive_vertex_count/* Triangle : 3 Vertices ~ 3 Indices */;
        shVertexBase = groupId * local_group_size * primitive_vertex_count;
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
            
            uint address = shVertexBase + index;
            vec4 vp = vec4(vertices.data[address].px, vertices.data[address].py, vertices.data[address].pz, 1.0);

            const mat4x4 mvp = mat4x4(
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            );

            vp = mvp * vp;
            shVertexPosition[index] = vec4(
                (vp.xy / vp.w) * (consts.resolution),
                vp.z,
                1.0f / vp.w
            );
        }

    }

    memoryBarrierShared();
    barrier();

    if ( localId < shPrimitiveCount ) {
        
        uint index = localId * primitive_vertex_count;
        rasterize(index, index + 1, index + 2);
    }
}