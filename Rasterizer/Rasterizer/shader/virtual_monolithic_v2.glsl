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
layout(set = 0, binding = 2, r32ui) uniform coherent restrict uimage2D target_image;

layout(push_constant) uniform readonly Constants {
    uint32_t primitives;
    uint32_t padding;
    f32vec2 resolution;
} consts;

/**/

vec3 interpolate_vec3_f32(float i, float j, vec3 f0, vec3 dFd10, vec3 dFd20) {
    return dFd10 * i + dFd20 * j + f0;
}

vec3 pinterpolate_vec3_f32(float i, float j, vec3 f0, vec3 dFd10, float dFd10w, vec3 dFd20, float dFd20w) {
    return f0 + ((dFd10 * i / dFd10w + dFd20 * j / dFd20w) / (dFd10 / dFd10w + dFd20 / dFd20w));
}

/**/

uint32_t packF32(float32_t val_) {
    return uint32_t(round(clamp(val_, 0.0f, 1.0f) * 2147483647.0f));
}

float32_t unpackF32(uint32_t val_) {
    return float32_t(val_) / 2147483647.0f;
}

/**/

void main() {

    //uint threadId = gl_WorkGroupID.x * local_group_size + gl_LocalInvocationIndex;
    uint threadId = gl_LocalInvocationIndex + gl_WorkGroupID.x * local_group_size + gl_WorkGroupID.y * gl_NumWorkGroups.x * local_group_size;

    if ( threadId < consts.primitives ) {

        // Load Vertices
        u32vec3 indices = u32vec3(threadId * 3 + 0, threadId * 3 + 1, threadId * 3 + 2);
        Vertex v0 = vertices.data[indices.x];
        Vertex v1 = vertices.data[indices.y];
        Vertex v2 = vertices.data[indices.z];

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

        // Transform positions to clip space

        vec4 vp0 = vec4((pp0.xy / pp0.w) * (consts.resolution), pp0.z / pp0.w, pp0.w);
        vec4 vp1 = vec4((pp1.xy / pp1.w) * (consts.resolution), pp1.z / pp1.w, pp1.w);
        vec4 vp2 = vec4((pp2.xy / pp2.w) * (consts.resolution), pp2.z / pp2.w, pp2.w);

        // Determine Edge Equations

        vec3 edge10 = vp1.xyz - vp0.xyz;
        vec3 edge20 = vp2.xyz - vp0.xyz;

        float edgeDet = edge20.x * edge10.y - edge20.y * edge10.x;

        // Counter-Clockwise Backface-Culling

        [[branch]] if (edgeDet > 0.0) {
            return;
        }

        // Determine pixel AABB

        vec2 aabbMin = min(vp0.xy, min(vp1.xy, vp2.xy));
        vec2 aabbMax = max(vp0.xy, max(vp1.xy, vp2.xy));

        [[branch]] if (aabbMax.x < 0.0 || aabbMax.y < 0.0 || aabbMin.x >= consts.resolution.x || aabbMin.y >= consts.resolution.y) {
            return;
        }

        // Clip candiate AABB

        u16vec2 uiAabbMin = u16vec2(aabbMin.x + (fract(aabbMin.x) <= 0.5f ? 0u : 1u), aabbMin.y + (fract(aabbMin.y) <= 0.5f ? 0u : 1u));
        u16vec2 uiAabbMax = u16vec2(aabbMax.x - (fract(aabbMax.x) <= 0.5f ? 1u : 0u), aabbMax.y - (fract(aabbMax.y) <= 0.5f ? 1u : 0u));

        uiAabbMin = u16vec2(min(max(uiAabbMin, u16vec2(0,0)), consts.resolution - 1.0f));
        uiAabbMax = u16vec2(min(max(uiAabbMax, u16vec2(0,0)), consts.resolution - 1.0f));
        aabbMin = f32vec2(uiAabbMin) + 0.5f;
        aabbMax = f32vec2(uiAabbMax) + 0.5f;

        // Cull elements between pixel-probes
        if ((fract(aabbMin.x) > 0.5f) && (fract(aabbMax.x) <= 0.5f) && ((aabbMax.x - aabbMin.x) <= 0.0f)) {
            return;
        }

        if ((fract(aabbMin.y) > 0.5f) && (fract(aabbMax.y) <= 0.5f) && ((aabbMax.y - aabbMin.y) <= 0.0f)) {
            return;
        }

        //

        #define epsilon 0.00005f
        aabbMin -= vec2(epsilon);
        aabbMax += vec2(epsilon);

        //

        float edgeInvDet = 1.0f / edgeDet;

        vec2 ssDx = vec2(-edge20.y, edge10.y) * edgeInvDet;
        vec2 ssDy = vec2(edge20.x, -edge10.x) * edgeInvDet;
        vec2 ssx = ssDx * (aabbMin.x - vp0.x);
        vec2 ssy = ssDy * (aabbMin.y - vp0.y);

        /**/
        vec3 attr0 = unpackUnorm4x8(v0.c).rgb;
        vec3 attrD10 = unpackUnorm4x8(v1.c).rgb - attr0;
        vec3 attrD20 = unpackUnorm4x8(v2.c).rgb - attr0;
        /**/

        for ( uint16_t y = uiAabbMin.y; y <= uiAabbMax.y; ++y ) {

            vec2 ssp = ssx + ssy;

            for ( uint16_t x = uiAabbMin.x; x <= uiAabbMax.x; ++x ) {

                if (ssp.x >= 0.0 && ssp.y >= 0.0 && (ssp.x + ssp.y) <= 1.0) {
                    
                    float depth = edge10.z * ssp.x + edge20.z * ssp.y + vp0.z;

                    // Drop pixel behind camera
                    if(depth < 0.0 || depth >= 1.0) {
                        continue;
                    }
                    
                    ivec2 tuv = ivec2(x, y);
                    const uint32_t stored = imageAtomicMin(depth_image, tuv, packF32(depth));
                    
                    if (unpackF32(stored) > depth) {
                        vec3 interColor = interpolate_vec3_f32(ssp.x, ssp.y, attr0, attrD10, attrD20);
                        //vec3 interColor = attrD10 * ssp.x + attrD20 * ssp.y + attr0;
                        //imageStore(target_image, tuv, u8vec4(vec4(interColor.rgb, 1.0f) * 255.0f));
                        //imageAtomicExchange(target_image, ivec2(x, y), packUnorm4x8(vec4(interColor.rgb, 1.0f)));
                        imageAtomicExchange(target_image, tuv, pack32(u8vec4(u8vec3(interColor.bgr * 255.0f), 255u)));
                    }
                }

                ssp += ssDx;
            }

            ssy += ssDy;
        }
    }
}