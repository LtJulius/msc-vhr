#pragma once
#include "stdafx.h"

#include <iostream>
#include <thread>

#include "pipeline.hpp"

struct frame_pass {
	vk::Fence otf_cpu_gpu = nullptr;
	vk::Semaphore otf_finish = nullptr;

	vk::Semaphore otf_image = nullptr;
	vk::PipelineStageFlags otf_image_stages = {};

	vk::CommandBuffer root = nullptr;
	vk::SubmitInfo root_submit = {};
};

struct stats {
	uint64_t frame = 0;
	uint64_t last_query_time = 0;
	uint64_t last_frame_time = 0;

	std::array<uint64_t, 4096uLL> query_accumulate = {};
	std::array<uint64_t, 4096uLL> frame_accumulate = {};
	uint16_t accumulate_index = 0;
};

class renderer {
public:
	using this_type = renderer;

public:
	renderer(const vk::Device& device_) noexcept :
		_device(device_),
		_queue(),
		_cmd_pool(),
		_frame_passes(),
		_stats() {}

	~renderer() {
		tidy();
	}

private:
	vk::Device _device = nullptr;
	vk::Queue _queue = nullptr;
	vk::CommandPool _cmd_pool = nullptr;

	std::vector<frame_pass> _frame_passes = {};
	stats _stats = {};

private:
	void reset_stats() {
		_stats.frame = 0uLL;
		_stats.last_frame_time = 0uLL;
		_stats.accumulate_index = 0u;
	}

	void tidy() {

		for (auto& frame_pass : _frame_passes) {

			_device.waitForFences(1uL, &frame_pass.otf_cpu_gpu, VK_TRUE, UINT64_MAX);

			_device.destroySemaphore(frame_pass.otf_finish);
			_device.destroySemaphore(frame_pass.otf_image);
			_device.destroyFence(frame_pass.otf_cpu_gpu);
			_device.freeCommandBuffers(_cmd_pool, 1uL, &frame_pass.root);
		}

		_device.destroyCommandPool(_cmd_pool);
	}

	void wait() {
		for (auto& frame_pass : _frame_passes) {
			_device.waitForFences(1uL, &frame_pass.otf_cpu_gpu, VK_TRUE, UINT64_MAX);
		}
	}

public:
	void build(const context& context_, size_t otf_count_) {

		_queue = context_.graphics_queue;
		_frame_passes.resize(otf_count_);

		/**/
		vk::CommandPoolCreateInfo cpci {
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			context_.graphics_queue_index
		};
		_cmd_pool = _device.createCommandPool(cpci);

		vk::CommandBufferAllocateInfo cbai {
			_cmd_pool,
			vk::CommandBufferLevel::ePrimary,
			static_cast<uint32_t>(otf_count_)
		};
		const auto vk_cmd_roots = _device.allocateCommandBuffers(cbai);

		/**/

		for (size_t idx = 0; idx < otf_count_; ++idx) {

			auto& frame_pass = _frame_passes[idx];
			frame_pass.root = vk_cmd_roots[idx];

			vk::SemaphoreCreateInfo sci {};
			#if USE_ALIAS_SWAPCHAIN == FALSE
			frame_pass.otf_image = _device.createSemaphore(sci);
			frame_pass.otf_finish = _device.createSemaphore(sci);
			#endif

			vk::FenceCreateInfo fci { vk::FenceCreateFlagBits::eSignaled };
			frame_pass.otf_cpu_gpu = _device.createFence(fci);

			frame_pass.otf_image_stages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		}

		/**/

		for (size_t idx = 0; idx < otf_count_; ++idx) {

			auto& frame_pass = _frame_passes[idx];
			const auto& prev_frame_pass = _frame_passes[(idx + _frame_passes.size() - 1uLL) % _frame_passes.size()];

			frame_pass.root_submit = vk::SubmitInfo {
				#if USE_ALIAS_SWAPCHAIN == FALSE
				1uL, &frame_pass.otf_image, &frame_pass.otf_image_stages,
				1uL, &frame_pass.root,
				1uL, &frame_pass.otf_finish
				#else
                0uL, nullptr, nullptr,
                1uL, &frame_pass.root,
                0uL, nullptr
				#endif
			};

		}
	}

	void record(const context& context_, const std::shared_ptr<pipeline>& pipeline_, const dataset& dataset_) {
		for (uint32_t i = 0; i < context_.otf_count; ++i) {
			auto& frame_pass = _frame_passes[i];
			pipeline_->record(context_, i, dataset_, frame_pass.root);
		}
	}

	void execute(const context& context_) {

		const auto mod = _frame_passes.size();
		reset_stats();

		// TODO: Check how we want to check performance, cause full round-trip has a lot of cpu work involved
		auto start = std::chrono::high_resolution_clock::now();
		auto end = std::chrono::high_resolution_clock::now();

		auto auto_start = std::chrono::high_resolution_clock::now();

		vk::Result result = vk::Result::eSuccess;
		size_t prev_pass = mod - 1uLL;
		size_t next_pass = 0uLL;

		bool continuing = true;
		while (continuing) {

			auto& prev = _frame_passes[prev_pass];
			auto& next = _frame_passes[next_pass];

			/**/

			start = std::chrono::high_resolution_clock::now();

			/**/

			result = _device.waitForFences(1uL, &prev.otf_cpu_gpu, VK_TRUE, UINT64_MAX);
			assert(result == vk::Result::eSuccess);
			result = _device.resetFences(1uL, &next.otf_cpu_gpu);
			assert(result == vk::Result::eSuccess);

			/**/

			#if USE_ALIAS_SWAPCHAIN == FALSE
			const auto index = context_.device.acquireNextImageKHR(
				context_.swapchain.native,
				UINT64_MAX,
				next.otf_image,
				nullptr
			);
			assert(index.result == vk::Result::eSuccess);
			#else
            const auto index = vk::ResultValue<uint32_t>(vk::Result::eSuccess, next_pass);
			#endif

			const bool realign = index.value != next_pass;
			if (realign) {
				prev_pass = (index.value + mod - 1) % mod;
				next_pass = index.value;
				#ifdef _DEBUG
                std::cout << "Realigned swapchain cycle \n";
				#endif
			}

			/**/

			if (not realign) {
				std::array<uint64_t, 2uLL> query_times {};
				result = context_.device.getQueryPoolResults(
					context_.swapchain.queries[next_pass],
					0uL,
					2uL,
					query_times.size() * sizeof(uint64_t),
					query_times.data(),
					sizeof(uint64_t),
					vk::QueryResultFlagBits::e64
				);

				auto diff_ticks = query_times[1] - query_times[0];

				if (result == vk::Result::eNotReady) {
					diff_ticks = _stats.query_accumulate[
						(_stats.accumulate_index + _stats.frame_accumulate.size() - 1) % _stats.frame_accumulate.size()
					];
				}

				// Will actually write results from previous swap cycle
				_stats.query_accumulate[_stats.accumulate_index] = diff_ticks;
			}

			/**/

			result = _queue.submit(1uL, &next.root_submit, next.otf_cpu_gpu);
			assert(result == vk::Result::eSuccess);

			#if USE_ALIAS_SWAPCHAIN == FALSE

			/**/

			vk::PresentInfoKHR pikhr {
				1uL,
				&next.otf_finish,
				1uL,
				&context_.swapchain.native,
				&index.value
			};

			result = _queue.presentKHR(pikhr);
			assert(result == vk::Result::eSuccess);

			/**/

			#endif

			result = _device.waitForFences(1uL, &next.otf_cpu_gpu, VK_TRUE, UINT64_MAX);
			assert(result == vk::Result::eSuccess);

			/**/

			end = std::chrono::high_resolution_clock::now();

			/**/

			++prev_pass;
			prev_pass = prev_pass % mod;
			++next_pass;
			next_pass = next_pass % mod;

			/**/

			++_stats.frame;
			_stats.last_frame_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
			_stats.accumulate_index = ++_stats.accumulate_index % _stats.frame_accumulate.size();
			_stats.frame_accumulate[_stats.accumulate_index] = _stats.last_frame_time;

			/**/

			SDL_Event event;
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
					continuing = false;
					break;
				}
				if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_f) {

					double avg = 0.0;
					for (const uint64_t time : _stats.frame_accumulate) {
						avg += time;
					}
					avg /= _stats.frame_accumulate.size();

					std::cout << "Frame " << _stats.frame
						<< " | Last Frame Time " << _stats.last_frame_time
						<< "ns | Average Frame Time " << avg
						<< "ns | Frame Rate " << ((1000.0 * 1000.0 * 1000.0) / avg) << std::endl;
				}
			}

			/**/

			if (std::chrono::duration_cast<std::chrono::seconds>(
				(std::chrono::high_resolution_clock::now() - auto_start)
			).count() > 5) {
				continuing = false;
			}
		}

		/**/

		{
			const auto probes_frames = (std::min)(_stats.frame, _stats.frame_accumulate.size());

			double avg_time = 0.0;
			for (auto i = 0uLL; i < probes_frames; ++i) {
				const auto time = _stats.frame_accumulate[i];
				avg_time += static_cast<double>(time);
			}
			avg_time /= static_cast<double>(probes_frames);

			double avg_ticks = 0.0;
			for (auto i = 0uLL; i < probes_frames; ++i) {
				const auto ticks = _stats.query_accumulate[i];
				avg_ticks += static_cast<double>(ticks);
			}
			avg_ticks /= static_cast<double>(probes_frames);

			const auto text = std::format(
				"Exit with {} frames | Last Frame Time {}ns | Avg Frame Time {}ns | Avg Ticks {} | Frame Rate {}\n",
				_stats.frame,
				_stats.last_frame_time,
				avg_time,
				avg_ticks,
				((1000.0 * 1000.0 * 1000.0) / avg_time)
			);

			std::cout << text;
			OutputDebugStringA(text.c_str());
		}

		/**/

		wait();
	}
};
