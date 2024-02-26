#include "stdafx.h"
#include "model.load.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include <random>

#include "tiny_obj_loader.h"
#include "vertex.hpp"
#include "vk.hpp"

static dataset transfer(const context& context_, std::vector<vertex>&& vertices_) {

	const vk::DeviceSize buffer_size = vertices_.size() * sizeof(vertex);

	vk::BufferCreateInfo tbci {
		vk::BufferCreateFlags {},
		buffer_size,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::SharingMode::eExclusive,
		0uL, nullptr
	};

	vk::BufferCreateInfo fbci {
		vk::BufferCreateFlags {},
		buffer_size,
		vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst |
		vk::BufferUsageFlagBits::eStorageBuffer,
		vk::SharingMode::eExclusive,
		0uL, nullptr
	};

	const auto stage_buffer = context_.device.createBuffer(tbci);
	const auto final_buffer = context_.device.createBuffer(fbci);

	const auto tmr = context_.device.getBufferMemoryRequirements(stage_buffer);
	const auto fmr = context_.device.getBufferMemoryRequirements(final_buffer);

	const vk::MemoryAllocateInfo tmai {
		tmr.size,
		vk_get_memory_index(context_.physical, tmr.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible)
	};

	const vk::MemoryAllocateInfo fmai {
		fmr.size,
		vk_get_memory_index(context_.physical, fmr.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
	};

	const auto stage_memory = context_.device.allocateMemory(tmai);
	const auto final_memory = context_.device.allocateMemory(fmai);

	constexpr vk::DeviceSize stage_offset = 0uLL;
	constexpr vk::DeviceSize final_offset = 0uLL;
	context_.device.bindBufferMemory(stage_buffer, stage_memory, stage_offset);
	context_.device.bindBufferMemory(final_buffer, final_memory, final_offset);

	/**/

	auto mapped_stage_memory = context_.device.mapMemory(
		stage_memory,
		stage_offset,
		buffer_size,
		vk::MemoryMapFlags {}
	);

	memcpy(mapped_stage_memory, vertices_.data(), buffer_size);

	const vk::MappedMemoryRange mapped_range { stage_memory, stage_offset, buffer_size };
	context_.device.flushMappedMemoryRanges(1uL, &mapped_range);
	context_.device.unmapMemory(stage_memory);

	/**/

	const vk::CommandBufferAllocateInfo cbai { context_.general_pool, vk::CommandBufferLevel::ePrimary, 1uL };
	auto cmds = context_.device.allocateCommandBuffers(cbai);
	auto& cmd = cmds.front();

	cmd.begin(vk::CommandBufferBeginInfo {});

	const vk::BufferCopy copy {
		0uL,
		0uL,
		buffer_size
	};
	cmd.copyBuffer(stage_buffer, final_buffer, 1uL, &copy);

	cmd.end();

	/**/

	const auto fence = context_.device.createFence({});

	vk::SubmitInfo submit { 0uL, nullptr, nullptr, 1uL, &cmd, 0uL, nullptr };
	auto result = context_.transfer_queue.submit(1uL, &submit, fence);
	assert(result == vk::Result::eSuccess);

	result = context_.device.waitForFences(1uL, &fence, VK_TRUE, UINT64_MAX);
	assert(result == vk::Result::eSuccess);

	context_.device.destroyFence(fence);

	/**/

	context_.device.freeCommandBuffers(context_.general_pool, 1uL, &cmd);

	/**/

	context_.device.destroyBuffer(stage_buffer);
	context_.device.freeMemory(stage_memory);

	/**/

	return dataset {
		{},
		{ .vkBuffer = final_buffer, .vkDeviceMemory = final_memory, .size = buffer_size, .offset = 0uL },
		{}
	};
}

dataset model_load_file(const context& context_, const std::filesystem::path& load_model_) {

	const auto cwd = std::filesystem::current_path();
	const auto load_model_path = std::filesystem::absolute(load_model_);

	if (not std::filesystem::exists(load_model_path)) {
		throw std::runtime_error("scene file not found.");
	}

	/**/

	tinyobj::attrib_t attrib {};
	std::vector<tinyobj::shape_t> shapes {};
	std::vector<tinyobj::material_t> materials {};

	std::string error {};

	const auto success = tinyobj::LoadObj(
		&attrib,
		&shapes,
		&materials,
		&error,
		load_model_path.string().c_str()
	);

	if (not success || not error.empty()) {
		std::cerr << error << std::endl;
		throw std::runtime_error("error file loading scene file.");
	}

	/**/

	std::vector<vertex> vertices {};

	glm::vec4 min_pos {}, max_pos {};

	{
		const auto ffv = shapes[0].mesh.indices[0].vertex_index;

		glm::vec4 def {};
		def.x = attrib.vertices[3 * ffv + 0];
		def.y = attrib.vertices[3 * ffv + 1];
		def.z = attrib.vertices[3 * ffv + 2];

		min_pos = def;
		max_pos = def;
	}

	/**/

	for (const auto& shape : shapes) {

		size_t index_offset = 0uLL;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {

			const auto fv = shape.mesh.num_face_vertices[f];
			for (size_t v = 0; v < fv; ++v) {

				vertex vert {};

				/**/

				const auto idx = shape.mesh.indices[index_offset + v];
				vert.position.x = attrib.vertices[3 * idx.vertex_index + 0];
				vert.position.y = attrib.vertices[3 * idx.vertex_index + 1];
				vert.position.z = attrib.vertices[3 * idx.vertex_index + 2];

				/**/

				vert.color.r = 127;
				vert.color.g = 127;
				vert.color.b = 127;
				vert.color.a = 255;

				/**/

				max_pos = (glm::max)(max_pos, vert.position);
				min_pos = (glm::min)(min_pos, vert.position);
				vertices.emplace_back(std::move(vert));
			}

			index_offset += fv;
		}

	}

	/**/

	std::random_device r {};
	std::mt19937 rng(r());
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);

	const auto extent = max_pos - min_pos;
	const auto model_center = extent / 2.F;

	const auto model_to_normalized_size = glm::vec4(0.5F, 0.5F, 0.5F, 0.0F) / extent;

	const auto model_to_global_scale = (std::min)(
		model_to_normalized_size.x,
		(std::min)(model_to_normalized_size.y, model_to_normalized_size.z)
	);

	const auto global_center = glm::vec4(0.75F, 0.25F, 0.5F, 0.0F);
	const auto global_factor = glm::vec4(1.5F, -1.5F, 1.0F, 1.F);

	auto color_used = 0u;
	auto color = glm::u8vec4(dist(rng), dist(rng), dist(rng), 255u);

	for (auto& vert : vertices) {

		const auto vec = vert.position - model_center;
		vert.position = global_center + vec * model_to_global_scale * global_factor;

		//const auto greybase = 127.5F + ((1.0F - vert.position.z) * 255.F - 127.5F) * 2.F;
		//const auto greybase = (1.0F - vert.position.z) * 255.F;
		//const auto greyscale = glm::clamp(greybase, 0.F, 255.F);
		//vert.color.r = static_cast<uint8_t>(greyscale);
		//vert.color.g = static_cast<uint8_t>(greyscale);
		//vert.color.b = static_cast<uint8_t>(greyscale);

		if (++color_used > 3u) {
			color = glm::u8vec4(dist(rng), dist(rng), dist(rng), 255u);
			color_used = 1u;
		}
		vert.color = color;
	}

	/**/

	return transfer(context_, std::move(vertices));
}
