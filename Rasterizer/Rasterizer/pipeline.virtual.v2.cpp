#include "stdafx.h"
#include "pipeline.virtual.v2.hpp"

#include "context.hpp"
#include "shader.hpp"
#include "vertex.hpp"

/**/

struct PushConstantPayload {
	uint32_t primitives;
	uint32_t __std_140_padding;
	glm::vec2 resolution;
};

/**/

virtual_pipeline_v2::virtual_pipeline_v2() {}

virtual_pipeline_v2::~virtual_pipeline_v2() noexcept {

	_device.destroyPipeline(_pipeline);
	_device.destroyPipelineLayout(_pipeline_layout);
	_device.destroyShaderModule(_mono_shader_module);

	_device.destroyDescriptorPool(_descriptor_pool);
	_device.destroyDescriptorSetLayout(_descriptor_layout);

	for (auto&& view : _image_views) {
		_device.destroyImageView(view);
	}
	_image_views.clear();
}

vk::DescriptorSet virtual_pipeline_v2::allocate_descriptor_set(
	const context& context_,
	const uint32_t idx_,
	const dataset& dataset_
) {

	if (_descriptor_sets.size() <= idx_) {

		vk::DescriptorSetAllocateInfo dsai {
			_descriptor_pool,
			1uL,
			&_descriptor_layout,
			nullptr
		};

		auto allocated = _device.allocateDescriptorSets(dsai);
		auto set = allocated.front();

		_descriptor_sets.emplace_back(set);
	}

	/**/

	vk::DescriptorBufferInfo vertex_buffer_info = {};
	vk::DescriptorImageInfo depth_image_info = {};
	vk::DescriptorImageInfo target_image_info = {};

	auto& set = _descriptor_sets[idx_];
	std::vector<vk::WriteDescriptorSet> writes {};

	{
		vertex_buffer_info = vk::DescriptorBufferInfo {
			dataset_.vertices.vkBuffer, dataset_.vertices.offset, dataset_.vertices.size
		};
		writes.emplace_back(
			vk::WriteDescriptorSet {
				set, 0uL, 0uL, 1uL, vk::DescriptorType::eStorageBuffer, nullptr, &vertex_buffer_info, nullptr
			}
		);
	}

	{
		vk::ImageViewCreateInfo ivci {
			vk::ImageViewCreateFlags {},
			context_.swapchain.alias_depth[idx_],
			vk::ImageViewType::e2D,
			vk::Format::eR32Uint,
			vk::ComponentMapping { vk::ComponentSwizzle::eR },
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		};
		vk::ImageView depth_image_view = _device.createImageView(ivci);
		_image_views.emplace_back(depth_image_view);
		depth_image_info = vk::DescriptorImageInfo {
			nullptr,
			depth_image_view, vk::ImageLayout::eGeneral
		};
		writes.emplace_back(
			vk::WriteDescriptorSet {
				set, 1uL, 0uL, 1uL, vk::DescriptorType::eStorageImage, &depth_image_info, nullptr, nullptr
			}
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
		_image_views.emplace_back(target_image_view);
		target_image_info = vk::DescriptorImageInfo {
			nullptr,
			target_image_view, vk::ImageLayout::eGeneral
		};
		writes.emplace_back(
			vk::WriteDescriptorSet {
				set, 2uL, 0uL, 1uL, vk::DescriptorType::eStorageImage, &target_image_info, nullptr, nullptr
			}
		);
	}

	_device.updateDescriptorSets(
		static_cast<uint32_t>(writes.size()),
		writes.data(),
		0uL,
		nullptr
	);

	/**/

	return set;
}

void virtual_pipeline_v2::setup(const context& context_, vk::Device device_) {

	_device = device_;

	#ifdef _WIN32
	auto mono_shader_code = read_shader_file(L"./shader/virtual_monolithic_v2.spirv");
	#else
    auto mono_shader_code = read_shader_file(R"(./shader/virtual_monolithic_v2.spirv)");
	#endif

	vk::ShaderModuleCreateInfo msmci {
		vk::ShaderModuleCreateFlags {},
		mono_shader_code.size() * sizeof(decltype(mono_shader_code)::value_type),
		mono_shader_code.data()
	};

	_mono_shader_module = device_.createShaderModule(msmci);

	/**/

	vk::ComputePipelineCreateInfo cpci {};
	vk::PipelineLayoutCreateInfo plci {};

	/**/

	vk::PipelineShaderStageCreateInfo mono_shader_ci {
		vk::PipelineShaderStageCreateFlags {},
		vk::ShaderStageFlagBits::eCompute,
		_mono_shader_module,
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

	// Texture : Z-Buffer
	layout_bindings.emplace_back(
		vk::DescriptorSetLayoutBinding {
			1uL,
			vk::DescriptorType::eStorageImage,
			1uL,
			vk::ShaderStageFlagBits::eCompute
		}
	);

	// Texture : Target-Image
	layout_bindings.emplace_back(
		vk::DescriptorSetLayoutBinding {
			2uL,
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

	_descriptor_layout = device_.createDescriptorSetLayout(dslci);
	layouts.emplace_back(_descriptor_layout);

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

	_pipeline_layout = device_.createPipelineLayout(plci);

	cpci.flags = vk::PipelineCreateFlags {};
	cpci.layout = _pipeline_layout;

	cpci.basePipelineHandle = nullptr;
	cpci.basePipelineIndex = -1L;

	auto result = device_.createComputePipeline(nullptr, cpci);

	_pipeline = result.value;

	/**/

	constexpr auto expected_sets = 3uL;
	std::vector<vk::DescriptorPoolSize> pool_sizes {};

	pool_sizes.emplace_back(vk::DescriptorPoolSize { vk::DescriptorType::eStorageBuffer, expected_sets * 2uL });
	pool_sizes.emplace_back(vk::DescriptorPoolSize { vk::DescriptorType::eStorageImage, expected_sets * 2uL });

	/**/

	vk::DescriptorPoolCreateInfo dpci {
		vk::DescriptorPoolCreateFlags {}, expected_sets, static_cast<uint32_t>(pool_sizes.size()), pool_sizes.data()
	};

	_descriptor_pool = device_.createDescriptorPool(dpci);
}

void virtual_pipeline_v2::record(
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
	pre_clear_barriers.emplace_back(
		vk::ImageMemoryBarrier {
			vk::AccessFlagBits::eMemoryRead,
			vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			context_.swapchain.alias_depth[idx_],
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		}
	);

	rootCmdBuffer_.pipelineBarrier(
		vk::PipelineStageFlagBits::eComputeShader,
		vk::PipelineStageFlagBits::eTransfer,
		vk::DependencyFlagBits::eByRegion,
		0uL,
		nullptr,
		0uL,
		nullptr,
		static_cast<uint32_t>(pre_clear_barriers.size()),
		pre_clear_barriers.data()
	);

	/**/

	rootCmdBuffer_.clearColorImage(
		context_.swapchain.images[idx_].vkImage,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ClearColorValue(0.F, 0.F, 0.F, 1.F),
		vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
	);
	rootCmdBuffer_.clearColorImage(
		context_.swapchain.alias_depth[idx_],
		vk::ImageLayout::eTransferDstOptimal,
		vk::ClearColorValue(1.F, 1.F, 1.F, 1.F),
		vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
	);

	/**/

	std::vector<vk::ImageMemoryBarrier> post_clear_barrier {};
	post_clear_barrier.emplace_back(
		vk::ImageMemoryBarrier {
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderWrite,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			context_.swapchain.images[idx_].vkImage,
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		}
	);
	post_clear_barrier.emplace_back(
		vk::ImageMemoryBarrier {
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			context_.swapchain.alias_depth[idx_],
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		}
	);

	rootCmdBuffer_.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer,
		vk::PipelineStageFlagBits::eComputeShader,
		vk::DependencyFlagBits::eByRegion,
		0uL,
		nullptr,
		0uL,
		nullptr,
		static_cast<uint32_t>(post_clear_barrier.size()),
		post_clear_barrier.data()
	);

	/**/

	rootCmdBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, context_.swapchain.queries[idx_], 0uL);

	/**/

	rootCmdBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

	auto descriptor_set = allocate_descriptor_set(context_, idx_, dataset_);
	rootCmdBuffer_.bindDescriptorSets(
		vk::PipelineBindPoint::eCompute,
		_pipeline_layout,
		0uL,
		1uL,
		&descriptor_set,
		0uL,
		nullptr
	);

	// We dispatch per primitive and forward the process without further branching
	constexpr auto local_grp_size = 64uL;
	constexpr auto local_grp_ext = local_grp_size * 1uL * 1uL;

	const uint32_t primitives = static_cast<uint32_t>(dataset_.vertices.size / sizeof(vertex) / 3uL);
	const uint32_t batches = (primitives / local_grp_ext) + (primitives % local_grp_ext > 0 ? 1uL : 0uL);

	const uint32_t batch_grp_y = 1uL + (batches / 65565uL);
	const uint32_t batch_grp_x = (batches / batch_grp_y) + (batches % batch_grp_y > 0 ? 1uL : 0uL);

	const PushConstantPayload payload {
		primitives,
		0,
		{ context_.window.width, context_.window.height }
	};
	rootCmdBuffer_.pushConstants(
		_pipeline_layout,
		vk::ShaderStageFlagBits::eCompute,
		0uL,
		sizeof(PushConstantPayload),
		&payload
	);
	rootCmdBuffer_.dispatch(batch_grp_x, batch_grp_y, 1uL);

	/**/

	rootCmdBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, context_.swapchain.queries[idx_], 1uL);

	/**/

	std::vector<vk::ImageMemoryBarrier> post_dispatch_barrier {};
	post_dispatch_barrier.emplace_back(
		vk::ImageMemoryBarrier {
			vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eMemoryRead, vk::ImageLayout::eGeneral,
			vk::ImageLayout::ePresentSrcKHR, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			context_.swapchain.images[idx_].vkImage,
			vk::ImageSubresourceRange { vk::ImageAspectFlagBits::eColor, 0uL, 1uL, 0uL, 1uL }
		}
	);

	rootCmdBuffer_.pipelineBarrier(
		vk::PipelineStageFlagBits::eComputeShader,
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
