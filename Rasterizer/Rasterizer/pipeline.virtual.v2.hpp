#pragma once
#include "pipeline.hpp"

class virtual_pipeline_v2 :
	public pipeline {
public:
	using this_type = virtual_pipeline_v2;

public:
	virtual_pipeline_v2();

	~virtual_pipeline_v2() noexcept override;

private:
	vk::Device _device = nullptr;

	vk::ShaderModule _mono_shader_module = nullptr;

	vk::PipelineLayout _pipeline_layout = nullptr;
	vk::Pipeline _pipeline = nullptr;

	vk::DescriptorSetLayout _descriptor_layout = nullptr;
	vk::DescriptorPool _descriptor_pool = nullptr;

	std::vector<vk::DescriptorSet> _descriptor_sets = {};
	std::vector<vk::ImageView> _image_views = {};

private:
	vk::DescriptorSet allocate_descriptor_set(const context& context_, const uint32_t idx_, const dataset& dataset_);

public:
	void setup(const context& context_, vk::Device device_) override;

	void record(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_,
		vk::CommandBuffer& rootCmdBuffer_
	) override;
};
