#include "stdafx.h"
#include "pipeline.native.hpp"

#include "context.hpp"
#include "shader.hpp"
#include "vertex.hpp"

native_pipeline::native_pipeline() {}

native_pipeline::~native_pipeline() noexcept {

	_device.destroyPipeline(_pipeline);
	_device.destroyPipelineLayout(_pipeline_layout);
	_device.destroyShaderModule(_fragment_shader_module);
	_device.destroyShaderModule(_vertex_shader_module);
}

void native_pipeline::setup(const context& context_, vk::Device device_) {

	_device = device_;

	#ifdef _WIN32
	auto vertex_shader_code = read_shader_file(L"./shader/native_vertex.spirv");
	auto fragment_shader_code = read_shader_file(L"./shader/native_fragment.spirv");
	#else
    auto vertex_shader_code = read_shader_file(R"(./shader/native_vertex.spirv)");
    auto fragment_shader_code = read_shader_file(R"(./shader/native_fragment.spirv)");
	#endif

	vk::ShaderModuleCreateInfo vsmci {
		vk::ShaderModuleCreateFlags {},
		vertex_shader_code.size() * sizeof(decltype(vertex_shader_code)::value_type),
		vertex_shader_code.data()
	};
	vk::ShaderModuleCreateInfo fsmci {
		vk::ShaderModuleCreateFlags {},
		fragment_shader_code.size() * sizeof(decltype(fragment_shader_code)::value_type),
		fragment_shader_code.data()
	};

	_vertex_shader_module = device_.createShaderModule(vsmci);
	_fragment_shader_module = device_.createShaderModule(fsmci);

	/**/

	vk::GraphicsPipelineCreateInfo gpci {};
	vk::PipelineLayoutCreateInfo plci {};

	/**/

	std::vector<vk::PipelineColorBlendAttachmentState> color_attachments {
		{
			VK_FALSE,
			vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
			vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
			vk::ColorComponentFlagBits::eR
			| vk::ColorComponentFlagBits::eG
			| vk::ColorComponentFlagBits::eB
			| vk::ColorComponentFlagBits::eA
		}
	};

	vk::PipelineColorBlendStateCreateInfo pcbsci {
		vk::PipelineColorBlendStateCreateFlags {},
		VK_FALSE,
		vk::LogicOp::eCopy,
		static_cast<uint32_t>(color_attachments.size()),
		color_attachments.data(),
		{ 0.F, 0.F, 0.F, 0.F }
	};
	gpci.pColorBlendState = &pcbsci;

	/**/

	vk::PipelineDepthStencilStateCreateInfo pdssci {
		vk::PipelineDepthStencilStateCreateFlags {},
		VK_TRUE,
		VK_TRUE,
		vk::CompareOp::eLessOrEqual,
		VK_FALSE,
		VK_FALSE,
		vk::StencilOpState {
			vk::StencilOp::eKeep,
			vk::StencilOp::eKeep,
			vk::StencilOp::eKeep,
			vk::CompareOp::eNever,
			{},
			{},
			{}
		},
		vk::StencilOpState {
			vk::StencilOp::eKeep,
			vk::StencilOp::eKeep,
			vk::StencilOp::eKeep,
			vk::CompareOp::eNever,
			{},
			{},
			{}
		},
		0.F,
		1.F
	};
	gpci.pDepthStencilState = &pdssci;

	/**/

	vk::PipelineDynamicStateCreateInfo pdsci {
		vk::PipelineDynamicStateCreateFlags {},
		0uL,
		nullptr
	};
	gpci.pDynamicState = &pdsci;

	/**/

	vk::PipelineInputAssemblyStateCreateInfo piasci {
		vk::PipelineInputAssemblyStateCreateFlags {},
		vk::PrimitiveTopology::eTriangleList,
		VK_FALSE
	};
	gpci.pInputAssemblyState = &piasci;

	/**/

	vk::PipelineMultisampleStateCreateInfo pmsci {
		vk::PipelineMultisampleStateCreateFlags {},
		vk::SampleCountFlagBits::e1,
		VK_FALSE,
		1.F,
		nullptr,
		VK_FALSE,
		VK_FALSE
	};
	gpci.pMultisampleState = &pmsci;

	/**/

	vk::PipelineRasterizationStateCreateInfo prsci {
		vk::PipelineRasterizationStateCreateFlags {},
		VK_FALSE,
		VK_FALSE,
		vk::PolygonMode::eFill,
		vk::CullModeFlagBits::eNone,
		vk::FrontFace::eClockwise,
		VK_FALSE,
		0.F,
		0.F,
		0.F,
		1.F
	};
	gpci.pRasterizationState = &prsci;

	/**/

	vk::PipelineShaderStageCreateInfo vertex_shader_ci {
		vk::PipelineShaderStageCreateFlags {},
		vk::ShaderStageFlagBits::eVertex,
		_vertex_shader_module,
		"main",
		nullptr
	};
	vk::PipelineShaderStageCreateInfo fragment_shader_ci {
		vk::PipelineShaderStageCreateFlags {},
		vk::ShaderStageFlagBits::eFragment,
		_fragment_shader_module,
		"main",
		nullptr
	};

	std::vector<vk::PipelineShaderStageCreateInfo> psscis {
		vertex_shader_ci,
		fragment_shader_ci
	};

	gpci.stageCount = static_cast<uint32_t>(psscis.size());
	gpci.pStages = psscis.data();

	/**/

	gpci.pTessellationState = nullptr;

	/**/

	std::vector<vk::VertexInputBindingDescription> vertex_bindings {
		{ 0uL, sizeof(vertex), vk::VertexInputRate::eVertex }
	};
	std::vector<vk::VertexInputAttributeDescription> vertex_attributes {
		{ 0uL, 0uL, vk::Format::eR32G32B32Sfloat, offsetof(vertex, position) },
		{ 1uL, 0uL, vk::Format::eR8G8B8A8Unorm, offsetof(vertex, color) }
	};

	vk::PipelineVertexInputStateCreateInfo pvisci {
		vk::PipelineVertexInputStateCreateFlags {},
		static_cast<uint32_t>(vertex_bindings.size()),
		vertex_bindings.data(),
		static_cast<uint32_t>(vertex_attributes.size()),
		vertex_attributes.data()
	};
	gpci.pVertexInputState = &pvisci;

	/**/

	std::vector<vk::Viewport> viewports {
		vk::Viewport {
			0.F, 0.F,
			static_cast<float>(context_.window.width), static_cast<float>(context_.window.height),
			0.F, 1.F
		}
	};
	std::vector<vk::Rect2D> scissors {
		vk::Rect2D {
			vk::Offset2D { 0L, 0L },
			vk::Extent2D { context_.window.width, context_.window.height }
		}
	};

	vk::PipelineViewportStateCreateInfo pvsci {
		vk::PipelineViewportStateCreateFlags {},
		static_cast<uint32_t>(viewports.size()),
		viewports.data(),
		static_cast<uint32_t>(scissors.size()),
		scissors.data()
	};
	gpci.pViewportState = &pvsci;

	/**/

	_pipeline_layout = device_.createPipelineLayout(plci);

	gpci.layout = _pipeline_layout;
	gpci.renderPass = context_.native_render_pass;
	gpci.subpass = 0;

	gpci.basePipelineHandle = nullptr;
	gpci.basePipelineIndex = -1L;

	auto result = device_.createGraphicsPipeline(nullptr, gpci);
	assert(result.result == vk::Result::eSuccess);

	_pipeline = result.value;

	/**/

	_clear_values.push_back(vk::ClearColorValue { 0.F, 0.F, 0.F, 1.F });
	_clear_values.push_back(vk::ClearDepthStencilValue { 1.F, 0uL });
}

void native_pipeline::record(
	const context& context_,
	const uint32_t idx_,
	const dataset& dataset_,
	vk::CommandBuffer& rootCmdBuffer_
) {

	rootCmdBuffer_.reset(vk::CommandBufferResetFlagBits::eReleaseResources);

	const vk::CommandBufferBeginInfo cbbi { vk::CommandBufferUsageFlagBits::eSimultaneousUse };
	rootCmdBuffer_.begin(cbbi);

	/**/

	rootCmdBuffer_.resetQueryPool(context_.swapchain.queries[idx_], 0uL, 2uL);
	rootCmdBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, context_.swapchain.queries[idx_], 0uL);

	/**/

	const vk::RenderPassBeginInfo rpbi {
		context_.native_render_pass,
		context_.swapchain.frames[idx_],
		vk::Rect2D { vk::Offset2D { 0L, 0L }, vk::Extent2D { context_.window.width, context_.window.height } },
		static_cast<uint32_t>(_clear_values.size()),
		_clear_values.data()
	};
	rootCmdBuffer_.beginRenderPass(rpbi, vk::SubpassContents::eInline);

	/**/

	rootCmdBuffer_.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

	if (dataset_.indices.vkBuffer) {
		rootCmdBuffer_.bindIndexBuffer(dataset_.indices.vkBuffer, dataset_.indices.offset, vk::IndexType::eUint32);
	}
	rootCmdBuffer_.bindVertexBuffers(0uL, 1uL, &dataset_.vertices.vkBuffer, &dataset_.vertices.offset);

	if (dataset_.indices.vkBuffer) {
		rootCmdBuffer_.drawIndexed(static_cast<uint32_t>(dataset_.indices.size / sizeof(uint32_t)), 1uL, 0uL, 0L, 0uL);
	} else {
		rootCmdBuffer_.draw(static_cast<uint32_t>(dataset_.vertices.size / sizeof(vertex)), 1uL, 0uL, 0uL);
		//rootCmdBuffer_.draw(3uL, 1uL, 0uL, 0uL);
	}

	/**/

	rootCmdBuffer_.endRenderPass();

	/**/

	rootCmdBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, context_.swapchain.queries[idx_], 1uL);

	/**/

	rootCmdBuffer_.end();
}
