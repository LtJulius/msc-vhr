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

layout(set = 0, binding = 3) restrict buffer FwdProgData {
    uint32_t clusterId;
} fwd_prog;

layout(push_constant) uniform readonly Constants {
    uint32_t primitives;
    uint32_t padding;
    f32vec2 resolution;
} consts;

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
shared uint shVertexBase;

/**/

void rasterize(uint i0, uint i1, uint i2) {

    // Load Vertices
    Vertex v0 = vertices.data[i0];
    Vertex v1 = vertices.data[i1];
    Vertex v2 = vertices.data[i2];
    
    //v0.position = f32vec3(0.0, 0.0, 0.0);
    //v1.position = f32vec3(0.025, 0.0, 0.0);
    //v2.position = f32vec3(0.025, 0.033, 0.0);
    
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
    
    //vp0.xy = consts.resolution * (vp0.xy * f32vec2(0.5, -0.5) + 0.5);
    //vp1.xy = consts.resolution * (vp1.xy * f32vec2(0.5, -0.5) + 0.5);
    //vp2.xy = consts.resolution * (vp2.xy * f32vec2(0.5, -0.5) + 0.5);
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
    
            const i16vec2 tuv = i16vec2(x, y);
            const uint32_t stored = imageAtomicMin(depth_image, tuv, packF32(depth));
    
            if (unpackF32(stored) > depth) {
                const f32vec3 interColor = (
                    interBase.x * unpackUnorm4x8(v0.c).rgb /* / pp0.w */ +
                    interBase.y * unpackUnorm4x8(v1.c).rgb /* / pp1.w */ +
                    interBase.z * unpackUnorm4x8(v2.c).rgb /* / pp2.w */
                );
    
                //imageStore(target_image, ivec2(x, y), uvec4(interColor, 1.0));
                //imageStore(target_image, tuv, u8vec4(vec4(interColor.rgb, 1.0f) * 255.0f));
                //imageAtomicExchange(target_image, tuv, packUnorm4x8(f32vec4(interColor.rgb, 1.0f)));
                imageAtomicExchange(target_image, tuv, pack32(u8vec4(u8vec3(interColor.bgr * 255.0f), 255u)));
            }
        }
    }
}

/**/

void main() {

    const uint localId = gl_LocalInvocationIndex;

    /**/

    [[branch]] if ( localId == 0u ) {

        const uint32_t prevCluster = atomicAdd(fwd_prog.clusterId, 1u);
        shPrimitiveCount = (consts.primitives > prevCluster * local_group_size) ? min(consts.primitives - (prevCluster * local_group_size), local_group_size /* Max : 1 Primitive ~ 1 Thread */) : 0u;
        shVertexBase = prevCluster * local_group_size * primitive_vertex_count;    
    }

    /**/

    memoryBarrierShared();
    barrier();

    /**/

    while (shPrimitiveCount > 0u) {

        if ( localId < shPrimitiveCount ) {
        
            uint baseIndex = shVertexBase + localId * primitive_vertex_count;
            rasterize(baseIndex, baseIndex + 1u, baseIndex + 2u);
        }

        /**/

        barrier();

        /**/

        [[branch]] if ( gl_LocalInvocationIndex == 0u ) {

            const uint32_t prevCluster = atomicAdd(fwd_prog.clusterId, 1u);
            shPrimitiveCount = (consts.primitives > prevCluster * local_group_size) ? min(consts.primitives - (prevCluster * local_group_size), local_group_size /* Max : 1 Primitive ~ 1 Thread */) : 0u;
            shVertexBase = prevCluster * local_group_size * primitive_vertex_count;    
        }

        /**/

        memoryBarrierShared();
        barrier();
    }

}