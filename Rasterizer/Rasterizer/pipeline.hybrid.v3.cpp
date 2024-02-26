#include "stdafx.h"
#include "pipeline.hybrid.v3.hpp"

#include "context.hpp"
#include "shader.hpp"
#include "vertex.hpp"
#include "vk.hpp"

/**/

struct PushConstantPayload {
	uint32_t primitives;
	uint32_t __std_140_padding;
	glm::vec2 resolution;
};

struct IntermediateDrawPayload {
	vk::DrawIndirectCommand draw_native;
};

struct IntermediateMetaPayload {
	uint32_t native_primitive_offset;
	uint32_t native_primitive_count;
	/**/
	glm::vec2 resolution;
};

/**/

hybrid_pipeline_v3::hybrid_pipeline_v3() {}

hybrid_pipeline_v3::~hybrid_pipeline_v3() noexcept {

	_device.destroyPipeline(_analytic_pipeline);
	_device.destroyPipeline(_native_pipeline);
	_device.destroyPipelineLayout(_analytic_pipeline_layout);
	_device.destroyPipelineLayout(_native_pipeline_layout);

	_device.destroyShaderModule(_analytic_shader_module);
	_device.destroyShaderModule(_fragment_shader_module);
	_device.destroyShaderModule(_vertex_shader_module);

	_device.destroyDescriptorPool(_descriptor_pool);
	_device.destroyDescriptorSetLayout(_native_descriptor_layout);
	_device.destroyDescriptorSetLayout(_analytic_descriptor_layout);

	for (auto&& view : _target_image_views) {
		_device.destroyImageView(view);
	}
	_target_image_views.clear();

	for (auto&& buffer : _intermediate_draw) {
		_device.destroyBuffer(buffer.vkBuffer);
		_device.freeMemory(buffer.vkDeviceMemory);
	}
	_intermediate_draw.clear();

	for (auto&& buffer : _intermediate_vertices) {
		_device.destroyBuffer(buffer.vkBuffer);
		_device.freeMemory(buffer.vkDeviceMemory);
	}
	_intermediate_vertices.clear();
}

void hybrid_pipeline_v3::ensure_intermediate(const context& context_, const uint32_t idx_, const dataset& dataset_) {

	const bool draw_indexed = dataset_.indices.vkBuffer;

	/**/

	if (_intermediate_vertices.size() > idx_ && _intermediate_vertices[idx_].size <= dataset_.vertices.size) {

		_device.destroyBuffer(std::exchange(_intermediate_vertices[idx_].vkBuffer, nullptr));
		_device.freeMemory(std::exchange(_intermediate_vertices[idx_].vkDeviceMemory, nullptr));
		_intermediate_vertices[idx_].size = 0uLL;
		_intermediate_vertices[idx_].offset = 0uLL;
	}

	/**/

	if (_intermediate_draw.size() <= idx_) {

		buffer intermediate {};

		const auto bci = vk::BufferCreateInfo {
			vk::BufferCreateFlags {},
			vk::DeviceSize { sizeof(IntermediateDrawPayload) + sizeof(IntermediateMetaPayload) + sizeof(uint32_t) },
			vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
			vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::SharingMode::eExclusive,
			VK_QUEUE_FAMILY_IGNORED, nullptr
		};
		intermediate.vkBuffer = _device.createBuffer(bci);
		assert(intermediate.vkBuffer);

		/**/

		auto memory_requirements = _device.getBufferMemoryRequirements(intermediate.vkBuffer);

		const auto mai = vk::MemoryAllocateInfo {
			memory_requirements.size,
			vk_get_memory_index(
				context_.physical,
				memory_requirements.memoryTypeBits,
				vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
			)
		};
		intermediate.vkDeviceMemory = _device.allocateMemory(mai);
		assert(intermediate.vkDeviceMemory);

		intermediate.offset = 0uLL;
		intermediate.size = bci.size;

		_device.bindBufferMemory(intermediate.vkBuffer, intermediate.vkDeviceMemory, intermediate.offset);

		/**/

		if (_intermediate_draw.size() > idx_) {
			_intermediate_draw[idx_] = std::move(intermediate);
		} else {
			_intermediate_draw.emplace_back(std::move(intermediate));
		}
	}

	/**/

	if (_intermediate_vertices.size() <= idx_ || _intermediate_vertices[idx_].size < dataset_.vertices.size) {

		buffer intermediate {};

		const auto bci = vk::BufferCreateInfo {
			vk::BufferCreateFlags {},
			vk::DeviceSize { dataset_.vertices.size },
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
			vk::SharingMode::eExclusive,
			VK_QUEUE_FAMILY_IGNORED, nullptr
		};
		intermediate.vkBuffer = _device.createBuffer(bci);
		assert(intermediate.vkBuffer);

		/**/

		auto memory_requirements = _device.getBufferMemoryRequirements(intermediate.vkBuffer);

		const auto mai = vk::MemoryAllocateInfo {
			memory_requirements.size,
			vk_get_memory_index(
				context_.physical,
				memory_requirements.memoryTypeBits,
				vk::MemoryPropertyFlagBits::eDeviceLocal
			)
		};
		intermediate.vkDeviceMemory = _device.allocateMemory(mai);
		assert(intermediate.vkDeviceMemory);

		intermediate.offset = 0uLL;
		intermediate.size = bci.size;

		_device.bindBufferMemory(intermediate.vkBuffer, intermediate.vkDeviceMemory, intermediate.offset);

		/**/

		if (_intermediate_vertices.size() > idx_) {
			_intermediate_vertices[idx_] = std::move(intermediate);
		} else {
			_intermediate_vertices.emplace_back(std::move(intermediate));
		}
	}
}

vk::DescriptorSet hybrid_pipeline_v3::allocate_analytic_descriptor_set(
	const context& context_,
	const uint32_t idx_,
	const dataset& dataset_
) {

	if (_analytic_descriptor_sets.size() <= idx_) {

		vk::DescriptorSetAllocateInfo dsai {
			_descriptor_pool,
			1uL,
			&_analytic_descriptor_layout,
			nullptr
		};

		auto allocated = _device.allocateDescriptorSets(dsai);
		auto set = allocated.front();

		_analytic_descriptor_sets.emplace_back(set);
	}

	/**/

	vk::DescriptorBufferInfo vertex_buffer_info = {};
	vk::DescriptorBufferInfo intermediate_vertex_buffer_info = {};
	vk::DescriptorBufferInfo intermediate_draw_buffer_info = {};

	vk::DescriptorBufferInfo depth_buffer_info = {};
	vk::DescriptorImageInfo target_image_info = {};

	auto& set = _analytic_descriptor_sets[idx_];
	std::vector<vk::WriteDescriptorSet> writes {};

	/**/

	{
		vertex_buffer_info = vk::DescriptorBufferInfo {
			dataset_.vertices.vkBuffer, dataset_.vertices.offset, dataset_.vertices.size
		};
		writes.emplace_back(
			set,
			0uL,
			0uL,
			1uL,
			vk::DescriptorType::eStorageBuffer,
			nullptr,
			&vertex_buffer_info,
			nullptr
		);
	}

	{
		intermediate_draw_buffer_info = vk::DescriptorBufferInfo {
			_intermediate_draw[idx_].vkBuffer,
			_intermediate_draw[idx_].offset,
			_intermediate_draw[idx_].size
		};
		writes.emplace_back(
			set,
			1uL,
			0uL,
			1uL,
			vk::DescriptorType::eStorageBuffer,
			nullptr,
			&intermediate_draw_buffer_info,
			nullptr
		);
	}

	{
		intermediate_vertex_buffer_info = vk::DescriptorBufferInfo {
			_intermediate_vertices[idx_].vkBuffer,
			_intermediate_vertices[idx_].offset,
			_intermediate_vertices[idx_].size
		};
		writes.emplace_back(
			set,
			2uL,
			0uL,
			1uL,
			vk::DescriptorType::eStorageBuffer,
			nullptr,
			&intermediate_vertex_buffer_info,
			nullptr
		);
	}

	/**/

	{
		depth_buffer_info = vk::DescriptorBufferInfo {
			context_.swapchain.depth_buffers[idx_].vkBuffer,
			context_.swapchain.depth_buffers[idx_].offset,
			context_.swapchain.depth_buffers[idx_].size,
		};
		writes.emplace_back(
			set,
			3uL,
			0uL,
			1uL,
			vk::DescriptorType::eStorageBuffer,
			nullptr,
			&depth_buffer_info,
			nullptr

		);
	}

	{
		vk::ImageViewCreateInfo ivci {
			vk::ImageViewCreateFlags {},
			context_.swapchain.images[idx_].vkImage,
			vk::ImageViewType::e2D,
			vk::Format::eB8G8R8A8Uint,
			vk::ComponentMapping {
				vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA
			},
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		};
		vk::ImageView target_image_view = _device.createImageView(ivci);
		_target_image_views.emplace_back(target_image_view);
		target_image_info = vk::DescriptorImageInfo {
			nullptr,
			target_image_view, vk::ImageLayout::eGeneral
		};
		writes.emplace_back(
			set,
			4uL,
			0uL,
			1uL,
			vk::DescriptorType::eStorageImage,
			&target_image_info,
			nullptr,
			nullptr
		);
	}

	/**/

	_device.updateDescriptorSets(
		static_cast<uint32_t>(writes.size()),
		writes.data(),
		0uL,
		nullptr
	);

	/**/

	return set;
}

vk::DescriptorSet hybrid_pipeline_v3::allocate_native_descriptor_set(
	const context& context_,
	const uint32_t idx_,
	const dataset& dataset_
) {
	if (_native_descriptor_sets.size() <= idx_) {

		vk::DescriptorSetAllocateInfo dsai {
			_descriptor_pool,
			1uL,
			&_native_descriptor_layout,
			nullptr
		};

		auto allocated = _device.allocateDescriptorSets(dsai);
		auto set = allocated.front();

		_native_descriptor_sets.emplace_back(set);
	}

	/**/

	vk::DescriptorBufferInfo intermediate_meta_buffer_info = {};
	vk::DescriptorBufferInfo depth_buffer_info = {};
	vk::DescriptorImageInfo target_image_info = {};

	auto& set = _native_descriptor_sets[idx_];
	std::vector<vk::WriteDescriptorSet> writes {};

	/**/

	{
		intermediate_meta_buffer_info = vk::DescriptorBufferInfo {
			_intermediate_draw[idx_].vkBuffer,
			_intermediate_draw[idx_].offset + sizeof(IntermediateDrawPayload),
			/*_intermediate_draw[idx_].size*/ sizeof(IntermediateMetaPayload)
		};
		writes.emplace_back(
			set,
			0uL,
			0uL,
			1uL,
			vk::DescriptorType::eUniformBuffer,
			nullptr,
			&intermediate_meta_buffer_info,
			nullptr
		);
	}

	{
		depth_buffer_info = vk::DescriptorBufferInfo {
			context_.swapchain.depth_buffers[idx_].vkBuffer,
			context_.swapchain.depth_buffers[idx_].offset,
			context_.swapchain.depth_buffers[idx_].size,
		};
		writes.emplace_back(
			set,
			1uL,
			0uL,
			1uL,
			vk::DescriptorType::eStorageBuffer,
			nullptr,
			&depth_buffer_info,
			nullptr

		);
	}

	{
		const vk::ImageView target_image_view = _target_image_views[idx_];
		target_image_info = vk::DescriptorImageInfo {
			nullptr,
			target_image_view,
			vk::ImageLayout::eGeneral
		};
		writes.emplace_back(
			set,
			2uL,
			0uL,
			1uL,
			vk::DescriptorType::eStorageImage,
			&target_image_info,
			nullptr,
			nullptr
		);
	}

	/**/

	_device.updateDescriptorSets(
		static_cast<uint32_t>(writes.size()),
		writes.data(),
		0uL,
		nullptr
	);

	/**/

	return set;
}

void hybrid_pipeline_v3::setup(const context& context_, vk::Device device_) {

	_device = device_;

	/**
	 * Create Native Sub-Pipeline
	 */

	{
		#ifdef _WIN32
		auto vertex_shader_code = read_shader_file(L"./shader/hybrid_vertex_v3.spirv");
		auto fragment_shader_code = read_shader_file(L"./shader/hybrid_fragment_v3.spirv");
		#else
        auto vertex_shader_code = read_shader_file(R"(./shader/hybrid_vertex_v3.spirv)");
        auto fragment_shader_code = read_shader_file(R"(./shader/hybrid_fragment_v3.spirv)");
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

		std::vector<vk::PipelineColorBlendAttachmentState> color_attachments {};

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

		VkPipelineRasterizationStateRasterizationOrderAMD orderAMD = {};
		orderAMD.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_RASTERIZATION_ORDER_AMD;
		orderAMD.rasterizationOrder = VK_RASTERIZATION_ORDER_RELAXED_AMD;

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
			1.F,
			&orderAMD
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
			{ 0uL, 0uL, vk::Format::eR32G32B32A32Sfloat, offsetof(vertex, position) },
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

		std::vector<vk::DescriptorSetLayout> layouts {};
		std::vector<vk::DescriptorSetLayoutBinding> layout_bindings {};

		// Storage Buffer : Meta-Buffer
		layout_bindings.emplace_back(
			vk::DescriptorSetLayoutBinding {
				0uL,
				vk::DescriptorType::eUniformBuffer,
				1uL,
				vk::ShaderStageFlagBits::eFragment
			}
		);

		// Storage Buffer : Z-Buffer
		layout_bindings.emplace_back(
			vk::DescriptorSetLayoutBinding {
				1uL,
				vk::DescriptorType::eStorageBuffer,
				1uL,
				vk::ShaderStageFlagBits::eFragment
			}
		);

		// Texture : Target-Image
		layout_bindings.emplace_back(
			vk::DescriptorSetLayoutBinding {
				2uL,
				vk::DescriptorType::eStorageImage,
				1uL,
				vk::ShaderStageFlagBits::eFragment
			}
		);

		vk::DescriptorSetLayoutCreateInfo dslci {
			vk::DescriptorSetLayoutCreateFlags {},
			static_cast<uint32_t>(layout_bindings.size()),
			layout_bindings.data()
		};

		_native_descriptor_layout = device_.createDescriptorSetLayout(dslci);
		layouts.emplace_back(_native_descriptor_layout);

		/**/

		plci.flags = vk::PipelineLayoutCreateFlags {};
		plci.setLayoutCount = static_cast<uint32_t>(layouts.size());
		plci.pSetLayouts = layouts.data();

		/**/

		_native_pipeline_layout = device_.createPipelineLayout(plci);

		gpci.layout = _native_pipeline_layout;
		gpci.renderPass = context_.hybrid_render_pass;
		gpci.subpass = 0;

		gpci.basePipelineHandle = nullptr;
		gpci.basePipelineIndex = -1L;

		auto result = device_.createGraphicsPipeline(nullptr, gpci);
		assert(result.result == vk::Result::eSuccess);

		_native_pipeline = result.value;
	}

	/**
	 * Create Analytical Pipeline
	 */

	{
		#ifdef _WIN32
		auto mono_shader_code = read_shader_file(L"./shader/hybrid_analytic_v3.spirv");
		#else
        auto mono_shader_code = read_shader_file(R"(./shader/hybrid_analytic_v3.spirv)");
		#endif

		vk::ShaderModuleCreateInfo msmci {
			vk::ShaderModuleCreateFlags {},
			mono_shader_code.size() * sizeof(decltype(mono_shader_code)::value_type),
			mono_shader_code.data()
		};

		_analytic_shader_module = device_.createShaderModule(msmci);

		/**/

		vk::ComputePipelineCreateInfo cpci {};
		vk::PipelineLayoutCreateInfo plci {};

		/**/

		vk::PipelineShaderStageCreateInfo mono_shader_ci {
			vk::PipelineShaderStageCreateFlags {},
			vk::ShaderStageFlagBits::eCompute,
			_analytic_shader_module,
			"main",
			nullptr
		};

		cpci.stage = mono_shader_ci;

		/**/

		std::vector<vk::DescriptorSetLayout> layouts {};
		std::vector<vk::DescriptorSetLayoutBinding> layout_bindings {};

		// Storage Buffer : Vertex Data
		layout_bindings.emplace_back(
			vk::DescriptorSetLayoutBinding {
				0uL,
				vk::DescriptorType::eStorageBuffer,
				1uL,
				vk::ShaderStageFlagBits::eCompute
			}
		);

		// Storage Buffer : Index Data

		// Storage Buffer : Indirect Dispatch
		layout_bindings.emplace_back(
			1uL,
			vk::DescriptorType::eStorageBuffer,
			1uL,
			vk::ShaderStageFlagBits::eCompute
		);

		// Storage Buffer : Intermediate Vertex
		layout_bindings.emplace_back(
			2uL,
			vk::DescriptorType::eStorageBuffer,
			1uL,
			vk::ShaderStageFlagBits::eCompute
		);

		// Storage Buffer : Z-Buffer
		layout_bindings.emplace_back(
			vk::DescriptorSetLayoutBinding {
				3uL,
				vk::DescriptorType::eStorageBuffer,
				1uL,
				vk::ShaderStageFlagBits::eCompute
			}
		);

		// Texture : Target-Image
		layout_bindings.emplace_back(
			vk::DescriptorSetLayoutBinding {
				4uL,
				vk::DescriptorType::eStorageImage,
				1uL,
				vk::ShaderStageFlagBits::eCompute
			}
		);

		vk::DescriptorSetLayoutCreateInfo dslci {
			vk::DescriptorSetLayoutCreateFlags {},
			static_cast<uint32_t>(layout_bindings.size()),
			layout_bindings.data()
		};

		_analytic_descriptor_layout = device_.createDescriptorSetLayout(dslci);
		layouts.emplace_back(_analytic_descriptor_layout);

		/**/

		vk::PushConstantRange push_constant {
			vk::ShaderStageFlagBits::eCompute,
			0uL,
			sizeof(PushConstantPayload)
		};

		/**/

		plci.flags = vk::PipelineLayoutCreateFlags {};
		plci.setLayoutCount = static_cast<uint32_t>(layouts.size());
		plci.pSetLayouts = layouts.data();
		plci.pushConstantRangeCount = 1uL;
		plci.pPushConstantRanges = &push_constant;

		/**/

		_analytic_pipeline_layout = device_.createPipelineLayout(plci);

		cpci.flags = vk::PipelineCreateFlags {};
		cpci.layout = _analytic_pipeline_layout;

		cpci.basePipelineHandle = nullptr;
		cpci.basePipelineIndex = -1L;

		auto result = device_.createComputePipeline(nullptr, cpci);

		_analytic_pipeline = result.value;
	}

	/**
	 * Create Shared Descriptors
	 */

	{
		constexpr auto expected_sys_count = 3uL;
		constexpr auto expected_sets = 3uL;
		std::vector<vk::DescriptorPoolSize> pool_sizes {};

		pool_sizes.emplace_back(vk::DescriptorType::eStorageBuffer, expected_sys_count * expected_sets * 7uL);
		pool_sizes.emplace_back(vk::DescriptorType::eStorageImage, expected_sys_count * expected_sets * 7uL);
		pool_sizes.emplace_back(vk::DescriptorType::eUniformBuffer, expected_sys_count * expected_sets * 7uL);

		/**/

		vk::DescriptorPoolCreateInfo dpci {
			vk::DescriptorPoolCreateFlags {}, expected_sets * expected_sys_count,
			static_cast<uint32_t>(pool_sizes.size()), pool_sizes.data()
		};

		_descriptor_pool = device_.createDescriptorPool(dpci);
	}
}

void hybrid_pipeline_v3::record(
	const context& context_,
	const uint32_t idx_,
	const dataset& dataset_,
	vk::CommandBuffer& rootCmdBuffer_
) {

	rootCmdBuffer_.reset(vk::CommandBufferResetFlagBits::eReleaseResources);

	/**/

	ensure_intermediate(context_, idx_, dataset_);

	/**/

	const vk::CommandBufferBeginInfo cbbi { vk::CommandBufferUsageFlagBits::eSimultaneousUse };
	rootCmdBuffer_.begin(cbbi);

	/**/

	rootCmdBuffer_.resetQueryPool(context_.swapchain.queries[idx_], 0uL, 2uL);

	/**/

	std::vector<vk::ImageMemoryBarrier> pre_clear_barriers {};
	pre_clear_barriers.emplace_back(
		vk::ImageMemoryBarrier {
			vk::AccessFlagBits::eMemoryRead,
			vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			context_.swapchain.images[idx_].vkImage,
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		}
	);

	std::vector<vk::BufferMemoryBarrier> pre_clear_buf_barriers {};
	pre_clear_buf_barriers.emplace_back(
		vk::AccessFlags {},
		vk::AccessFlagBits::eTransferWrite,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		_intermediate_draw[idx_].vkBuffer,
		_intermediate_draw[idx_].offset,
		_intermediate_draw[idx_].size
	);
	pre_clear_buf_barriers.emplace_back(
		vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
		vk::AccessFlagBits::eTransferWrite,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		context_.swapchain.depth_buffers[idx_].vkBuffer,
		context_.swapchain.depth_buffers[idx_].offset,
		context_.swapchain.depth_buffers[idx_].size
	);

	rootCmdBuffer_.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands,
		vk::PipelineStageFlagBits::eTransfer,
		vk::DependencyFlagBits::eByRegion,
		0uL,
		nullptr,
		static_cast<uint32_t>(pre_clear_buf_barriers.size()),
		pre_clear_buf_barriers.data(),
		static_cast<uint32_t>(pre_clear_barriers.size()),
		pre_clear_barriers.data()
	);

	/**/

	rootCmdBuffer_.fillBuffer(
		_intermediate_draw[idx_].vkBuffer,
		_intermediate_draw[idx_].offset,
		_intermediate_draw[idx_].size,
		0x0uL
	);
	rootCmdBuffer_.clearColorImage(
		context_.swapchain.images[idx_].vkImage,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ClearColorValue(0.F, 0.F, 0.F, 1.F),
		vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
	);
	rootCmdBuffer_.fillBuffer(
		context_.swapchain.depth_buffers[idx_].vkBuffer,
		context_.swapchain.depth_buffers[idx_].offset,
		context_.swapchain.depth_buffers[idx_].size,
		0xFFFF'FFFEuL
	);

	/**/

	std::vector<vk::ImageMemoryBarrier> post_clear_barrier {};
	post_clear_barrier.emplace_back(
		vk::ImageMemoryBarrier {
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			context_.swapchain.images[idx_].vkImage,
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		}
	);

	std::vector<vk::BufferMemoryBarrier> post_clear_buf_barriers {};
	post_clear_buf_barriers.emplace_back(
		vk::AccessFlagBits::eTransferWrite,
		vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		_intermediate_draw[idx_].vkBuffer,
		_intermediate_draw[idx_].offset,
		_intermediate_draw[idx_].size
	);
	post_clear_buf_barriers.emplace_back(
		vk::AccessFlagBits::eTransferWrite,
		vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		context_.swapchain.depth_buffers[idx_].vkBuffer,
		context_.swapchain.depth_buffers[idx_].offset,
		context_.swapchain.depth_buffers[idx_].size
	);

	rootCmdBuffer_.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlagBits::eByRegion,
		0uL,
		nullptr,
		static_cast<uint32_t>(post_clear_buf_barriers.size()),
		post_clear_buf_barriers.data(),
		static_cast<uint32_t>(post_clear_barrier.size()),
		post_clear_barrier.data()
	);

	/**/

	rootCmdBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, context_.swapchain.queries[idx_], 0uL);

	/**
	 * Dispatch Analytical Pipeline
	 */

	{
		rootCmdBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, _analytic_pipeline);

		auto descriptor_set = allocate_analytic_descriptor_set(context_, idx_, dataset_);
		rootCmdBuffer_.bindDescriptorSets(
			vk::PipelineBindPoint::eCompute,
			_analytic_pipeline_layout,
			0uL,
			1uL,
			&descriptor_set,
			0uL,
			nullptr
		);

		/**/

		// We dispatch per wavefront group, which will allocate a single cluster group per cycle
		constexpr auto local_grp_size = 64uL;
		constexpr auto local_grp_ext = local_grp_size * 1uL * 1uL;

		const uint32_t primitives = static_cast<uint32_t>(dataset_.vertices.size / sizeof(vertex) / 3uL);
		const uint32_t max_batches = (primitives / local_grp_ext) + (primitives % local_grp_ext > 0 ? 1uL : 0uL);

		// TODO: Query possible scaling count
		const uint32_t shader_engines = 4uL;
		const uint32_t cu_per_shader_engine = 14uL;
		const uint32_t simd_per_cu = 4uL;
		const uint32_t latency_hiding_target = 10uL;

		const uint32_t max_concurrent_wavefronts = simd_per_cu * cu_per_shader_engine * shader_engines;
		// TODO: Use max occupancy to scale target wavefronts
		const auto current_occupancy_limit = 5.F / 10.F;
		const uint32_t target_concurrent_wavefronts = max_concurrent_wavefronts *
			static_cast<uint32_t>(static_cast<float>(latency_hiding_target) * current_occupancy_limit);

		const uint32_t batches = min(max_batches, target_concurrent_wavefronts);

		const PushConstantPayload payload {
			primitives,
			0,
			{ context_.window.width, context_.window.height }
		};
		rootCmdBuffer_.pushConstants(
			_analytic_pipeline_layout,
			vk::ShaderStageFlagBits::eCompute,
			0uL,
			sizeof(PushConstantPayload),
			&payload
		);
		rootCmdBuffer_.dispatch(batches, 1uL, 1uL);
	}

	/**
	 * Dispatch Barrier
	 */

	{
		const auto indirect_buffer_barrier = std::vector<vk::BufferMemoryBarrier> {
			vk::BufferMemoryBarrier {
				vk::AccessFlagBits::eShaderWrite,
				vk::AccessFlagBits::eIndirectCommandRead,
				VK_QUEUE_FAMILY_IGNORED,
				VK_QUEUE_FAMILY_IGNORED,
				_intermediate_draw[idx_].vkBuffer,
				0uL,
				sizeof(IntermediateDrawPayload),
				nullptr
			},
			vk::BufferMemoryBarrier {
				vk::AccessFlagBits::eShaderWrite,
				vk::AccessFlagBits::eVertexAttributeRead,
				VK_QUEUE_FAMILY_IGNORED,
				VK_QUEUE_FAMILY_IGNORED,
				_intermediate_vertices[idx_].vkBuffer,
				_intermediate_vertices[idx_].offset,
				_intermediate_vertices[idx_].size,
				nullptr
			}
		};

		rootCmdBuffer_.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader,
			vk::PipelineStageFlagBits::eDrawIndirect,
			vk::DependencyFlagBits::eByRegion,
			0uL,
			nullptr,
			0uL,
			nullptr,
			0uL,
			nullptr
		);
	}

	/**
	 * Dispatch Native Sub-Pipeline
	 */

	{
		const vk::RenderPassBeginInfo rpbi {
			context_.hybrid_render_pass,
			context_.hybrid_empty_frame_buffer,
			vk::Rect2D { vk::Offset2D { 0L, 0L }, vk::Extent2D { context_.window.width, context_.window.height } },
			0uL,
			nullptr
		};
		rootCmdBuffer_.beginRenderPass(rpbi, vk::SubpassContents::eInline);

		/**/

		rootCmdBuffer_.bindPipeline(vk::PipelineBindPoint::eGraphics, _native_pipeline);

		auto descriptor_set = allocate_native_descriptor_set(context_, idx_, dataset_);
		rootCmdBuffer_.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			_native_pipeline_layout,
			0uL,
			1uL,
			&descriptor_set,
			0uL,
			nullptr
		);

		/**/

		if (not _intermediate_indices.empty()) {
			rootCmdBuffer_.bindIndexBuffer(
				_intermediate_indices[idx_].vkBuffer,
				_intermediate_indices[idx_].offset,
				vk::IndexType::eUint32
			);
		}
		rootCmdBuffer_.bindVertexBuffers(
			0uL,
			1uL,
			&_intermediate_vertices[idx_].vkBuffer,
			&_intermediate_vertices[idx_].offset
		);

		if (_intermediate_indices.empty()) {
			rootCmdBuffer_.drawIndirect(
				_intermediate_draw[idx_].vkBuffer,
				offsetof(IntermediateDrawPayload, draw_native),
				1uL,
				sizeof(vk::DrawIndirectCommand)
			);

		} else {
			rootCmdBuffer_.drawIndirect(
				_intermediate_draw[idx_].vkBuffer,
				offsetof(IntermediateDrawPayload, draw_native),
				1uL,
				sizeof(vk::DrawIndirectCommand)
			);
		}

		/**/

		rootCmdBuffer_.endRenderPass();
	}

	/**/

	rootCmdBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, context_.swapchain.queries[idx_], 1uL);

	/**/

	std::vector<vk::ImageMemoryBarrier> post_dispatch_barrier {};
	post_dispatch_barrier.emplace_back(
		vk::AccessFlagBits::eShaderWrite,
		vk::AccessFlagBits::eMemoryRead,
		vk::ImageLayout::eGeneral,
		vk::ImageLayout::ePresentSrcKHR,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		context_.swapchain.images[idx_].vkImage,
		vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
	);

	rootCmdBuffer_.pipelineBarrier(
		vk::PipelineStageFlagBits::eFragmentShader,
		vk::PipelineStageFlagBits::eBottomOfPipe,
		vk::DependencyFlagBits::eByRegion,
		0uL,
		nullptr,
		0uL,
		nullptr,
		static_cast<uint32_t>(post_dispatch_barrier.size()),
		post_dispatch_barrier.data()
	);

	/**/

	rootCmdBuffer_.end();
}
