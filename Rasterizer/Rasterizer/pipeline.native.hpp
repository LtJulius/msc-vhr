#pragma once
#include "pipeline.hpp"

class native_pipeline :
	public pipeline {
public:
	using this_type = native_pipeline;

public:
	native_pipeline();

	~native_pipeline() noexcept override;

private:
	vk::Device _device = nullptr;

	std::vector<vk::ClearValue> _clear_values = {};

	vk::ShaderModule _vertex_shader_module = nullptr;
	vk::ShaderModule _fragment_shader_module = nullptr;

	vk::PipelineLayout _pipeline_layout = nullptr;
	vk::Pipeline _pipeline = nullptr;

public:
	void setup(const context& context_, vk::Device device_) override;

	void record(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_,
		vk::CommandBuffer& rootCmdBuffer_
	) override;
};
