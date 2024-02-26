#pragma once

#include "stdafx.h"

#include "benchmark.hpp"
#include "image.hpp"
#include "pipeline.hpp"

struct window {
	uint32_t width = 0;
	uint32_t height = 0;

	SDL_Window* native = nullptr;

	vk::SurfaceKHR surface = nullptr;
	vk::Format format = vk::Format::eUndefined;
};

struct swapchain {
	vk::SwapchainKHR native = nullptr;

	std::vector<image> images = {};
	std::vector<vk::Image> depths = {};
	std::vector<vk::Image> alias_depth = {};
	std::vector<vk::DeviceMemory> depth_memory = {};
	std::vector<std::array<vk::ImageView, 2>> attachments = {};
	std::vector<vk::Framebuffer> frames = {};

	/**/

	std::vector<buffer> depth_buffers {};

	/**/

	std::vector<vk::QueryPool> queries = {};
};

struct context {
	uint32_t otf_count = 3uL;

	vk::Instance instance = nullptr;

	vk::PhysicalDevice physical = nullptr;
	vk::Device device = nullptr;

	uint32_t compute_queue_index = ~0uL;
	vk::Queue compute_queue = nullptr;
	uint32_t transfer_queue_index = ~0uL;
	vk::Queue transfer_queue = nullptr;
	uint32_t graphics_queue_index = ~0uL;
	vk::Queue graphics_queue = nullptr;

	vk::CommandPool general_pool = nullptr;
	vk::RenderPass native_render_pass = nullptr;
	vk::RenderPass hybrid_render_pass = nullptr;

	vk::Framebuffer hybrid_empty_frame_buffer = nullptr;

	window window = {};
	swapchain swapchain = {};

	std::map<benchmark_pipeline_bits, std::shared_ptr<pipeline>> pipelines = {};
};
