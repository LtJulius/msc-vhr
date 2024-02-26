#pragma once
#include "pipeline.hpp"

class hybrid_pipeline :
	public pipeline {
public:
	using this_type = hybrid_pipeline;

public:
	hybrid_pipeline();

	~hybrid_pipeline() noexcept override;

private:
	vk::Device _device = nullptr;

	vk::ShaderModule _virtual_shader_module = nullptr;
	vk::ShaderModule _analytic_shader_module = nullptr;
	vk::ShaderModule _vertex_shader_module = nullptr;
	vk::ShaderModule _fragment_shader_module = nullptr;

	vk::PipelineLayout _analytic_pipeline_layout = nullptr;
	vk::Pipeline _analytic_pipeline = nullptr;

	vk::PipelineLayout _native_pipeline_layout = nullptr;
	vk::Pipeline _native_pipeline = nullptr;

	vk::PipelineLayout _virtual_pipeline_layout = nullptr;
	vk::Pipeline _virtual_pipeline = nullptr;

	vk::DescriptorSetLayout _virtual_descriptor_layout = nullptr;
	vk::DescriptorSetLayout _native_descriptor_layout = nullptr;
	vk::DescriptorSetLayout _analytic_descriptor_layout = nullptr;
	vk::DescriptorPool _descriptor_pool = nullptr;

	std::vector<vk::DescriptorSet> _analytic_descriptor_sets = {};
	std::vector<vk::DescriptorSet> _native_descriptor_sets = {};
	std::vector<vk::DescriptorSet> _virtual_descriptor_sets = {};

	std::vector<vk::ImageView> _target_image_views = {};

	/**/

	std::vector<buffer> _intermediate_indices {};
	std::vector<buffer> _intermediate_draw {};

private:
	void ensure_intermediate(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_
	);

	vk::DescriptorSet allocate_analytic_descriptor_set(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_
	);

	vk::DescriptorSet allocate_virtual_descriptor_set(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_
	);

	vk::DescriptorSet allocate_native_descriptor_set(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_
	);

public:
	void setup(const context& context_, vk::Device device_) override;

	void record(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_,
		vk::CommandBuffer& rootCmdBuffer_
	) override;
};
