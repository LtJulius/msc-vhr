glslangValidator.exe -V --target-env vulkan1.3 -S vert -v .\native_vertex.glsl -o .\native_vertex.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S frag -v .\native_fragment.glsl -o .\native_fragment.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\virtual_monolithic_v1.glsl -o .\virtual_monolithic_v1.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\virtual_monolithic_v2.glsl -o .\virtual_monolithic_v2.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\virtual_monolithic_v3.glsl -o .\virtual_monolithic_v3.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\virtual_monolithic_v4.glsl -o .\virtual_monolithic_v4.spirv

glslangValidator.exe -V --target-env vulkan1.3 -S vert -v .\hybrid_vertex_v1.glsl -o .\hybrid_vertex_v1.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S frag -v .\hybrid_fragment_v1.glsl -o .\hybrid_fragment_v1.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\hybrid_compute_v1.glsl -o .\hybrid_compute_v1.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\hybrid_analytic_v1.glsl -o .\hybrid_analytic_v1.spirv

glslangValidator.exe -V --target-env vulkan1.3 -S vert -v .\hybrid_vertex_v2.glsl -o .\hybrid_vertex_v2.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S frag -v .\hybrid_fragment_v2.glsl -o .\hybrid_fragment_v2.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\hybrid_compute_v2.glsl -o .\hybrid_compute_v2.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\hybrid_analytic_v2.glsl -o .\hybrid_analytic_v2.spirv

glslangValidator.exe -V --target-env vulkan1.3 -S vert -v .\hybrid_vertex_v3.glsl -o .\hybrid_vertex_v3.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S frag -v .\hybrid_fragment_v3.glsl -o .\hybrid_fragment_v3.spirv
glslangValidator.exe -V --target-env vulkan1.3 -S comp -v .\hybrid_analytic_v3.glsl -o .\hybrid_analytic_v3.spirv

spirv-opt -O .\virtual_monolithic_v1.spirv -o .\virtual_monolithic_v1.spirv
spirv-opt -O .\virtual_monolithic_v2.spirv -o .\virtual_monolithic_v2.spirv
spirv-opt -O .\virtual_monolithic_v3.spirv -o .\virtual_monolithic_v3.spirv
spirv-opt -O .\virtual_monolithic_v4.spirv -o .\virtual_monolithic_v4.spirv

spirv-opt -O .\hybrid_vertex_v1.spirv -o .\hybrid_vertex_v1.spirv
spirv-opt -O .\hybrid_fragment_v1.spirv -o .\hybrid_fragment_v1.spirv
spirv-opt -O .\hybrid_compute_v1.spirv -o .\hybrid_compute_v1.spirv
spirv-opt -O .\hybrid_analytic_v1.spirv -o .\hybrid_analytic_v1.spirv

spirv-opt -O .\hybrid_vertex_v2.spirv -o .\hybrid_vertex_v2.spirv
spirv-opt -O .\hybrid_fragment_v2.spirv -o .\hybrid_fragment_v2.spirv
spirv-opt -O .\hybrid_compute_v2.spirv -o .\hybrid_compute_v2.spirv
spirv-opt -O .\hybrid_analytic_v2.spirv -o .\hybrid_analytic_v2.spirv

spirv-opt -O .\hybrid_vertex_v3.spirv -o .\hybrid_vertex_v3.spirv
spirv-opt -O .\hybrid_fragment_v3.spirv -o .\hybrid_fragment_v3.spirv
spirv-opt -O .\hybrid_analytic_v3.spirv -o .\hybrid_analytic_v3.spirv