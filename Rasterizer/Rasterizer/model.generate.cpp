#include "stdafx.h"
#include "model.generate.hpp"

#include "vertex.hpp"
#include "vk.hpp"

struct generator_params {
	glm::vec2 size;
	glm::vec2 padding;
	uint32_t limit;
};

/**/

generator_params parse(
	const context& context_,
	std::string_view generator_,
	std::string_view param_x_,
	std::string_view param_y_,
	std::string_view param_limit_
);

std::vector<vertex> generate(const context& context_, const generator_params& params_);

dataset transfer(const context& context_, std::vector<vertex>&& vertices_);

/**/

dataset model_load_generate(const context& context_, const std::filesystem::path& load_model_) {

	const auto prefix = R"(gen://)";
	const auto stringified = load_model_.string().substr(strlen(prefix));

	const auto first_delim = stringified.find_first_of('-');
	const auto generator = stringified.substr(0, first_delim);

	const auto second_delim = stringified.find('-', first_delim + 1);
	const auto str_param_x = stringified.substr(first_delim + 1, second_delim - (first_delim + 1));

	const auto limit_delim = stringified.find('-', second_delim + 1);
	const auto str_param_y = stringified.substr(second_delim + 1, limit_delim - (second_delim + 1));

	std::string str_param_limit = {};
	if (limit_delim < stringified.size()) {
		str_param_limit = stringified.substr(limit_delim + 1, stringified.size() - (limit_delim + 1));
	}

	/**/

	auto params = parse(context_, generator, str_param_x, str_param_y, str_param_limit);
	auto vertices = generate(context_, params);

	auto result = transfer(context_, std::move(vertices));

	/**/

	return result;
}

/**/

generator_params parse(
	const context& context_,
	std::string_view generator_,
	std::string_view param_x_,
	std::string_view param_y_,
	std::string_view param_limit_
) {

	generator_params result {};

	if (param_x_.contains('w')) {
		result.size.x = static_cast<float>(context_.window.width);
	} else {
		result.size.x = std::atof(param_x_.data());
	}

	if (param_y_.contains('h')) {
		result.size.y = static_cast<float>(context_.window.height);
	} else {
		result.size.y = std::atof(param_y_.data());
	}

	if (generator_.contains("ip")) {
		result.padding.x = -(result.size.x / 2);
		result.padding.y = -(result.size.y / 2);
	} else {
		result.padding.x = 0.F;
		result.padding.y = 0.F;
	}

	if (not param_limit_.empty()) {
		result.limit = std::atol(param_limit_.data());
	} else {
		result.limit = 0uL;
	}

	return result;
}

std::vector<vertex> generate(const context& context_, const generator_params& params_) {

	std::vector<vertex> result {};

	auto shift_x = -0.1f / static_cast<float>(context_.window.width);
	auto shift_y = 0.1f / static_cast<float>(context_.window.height);

	auto frag_base = params_.size + params_.padding;
	auto frag_count = glm::ceil(glm::vec2 { context_.window.width, context_.window.height } / frag_base);

	auto frag_size = params_.size;
	frag_base /= glm::vec2 { context_.window.width, context_.window.height };
	frag_size /= glm::vec2 { context_.window.width, context_.window.height };

	if (params_.limit > 0uL) {
		const auto base_vertex_count = frag_count.y * frag_count.x * 2.F * 3.F;
		const auto frag_factor = std::sqrtf(base_vertex_count / static_cast<float>(params_.limit));
		frag_count /= frag_factor;

		shift_x += (static_cast<float>(context_.window.width) * frag_factor - frag_count.x) * frag_base.x / 2.F;
		shift_y += (static_cast<float>(context_.window.height) * frag_factor - frag_count.y) * frag_base.y / 2.F;
	}

	const auto hc = static_cast<uint32_t>(frag_count.y);
	const auto wc = static_cast<uint32_t>(frag_count.x);
	result.reserve(wc * hc * 2uL * 3uL);

	float const_z = 0.49F;

	for (uint32_t h = 0; h < hc; ++h) {

		const auto line_start_y = shift_y + frag_base.y * static_cast<float>(h);
		const auto line_stop_y = shift_y + frag_size.y * static_cast<float>(h + 1uL);

		for (uint32_t w = 0; w < wc; ++w) {

			const auto line_start_x = shift_x + frag_base.x * static_cast<float>(w);
			const auto line_stop_x = shift_x + frag_size.x * static_cast<float>(w + 1uL);

			const_z -= 0.0000001F;

			/**/

			result.emplace_back(
				glm::vec4(line_start_x, line_start_y, const_z, 0.F),
				glm::u8vec4(255, 0, 0, 255)
			);

			result.emplace_back(
				glm::vec4(line_stop_x, line_start_y, const_z, 0.F),
				glm::u8vec4(0, 255, 0, 255)
			);

			result.emplace_back(
				glm::vec4(line_stop_x, line_stop_y, const_z, 0.F),
				glm::u8vec4(0, 0, 255, 255)
			);

			/**/

			result.emplace_back(
				glm::vec4(line_stop_x, line_stop_y, const_z, 0.F),
				glm::u8vec4(0, 0, 255, 255)
			);

			result.emplace_back(
				glm::vec4(line_start_x, line_stop_y, const_z, 0.F),
				glm::u8vec4(0, 0, 0, 255)
			);

			result.emplace_back(
				glm::vec4(line_start_x, line_start_y, const_z, 0.F),
				glm::u8vec4(255, 0, 0, 255)
			);
		}
	}

	return result;
}

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
