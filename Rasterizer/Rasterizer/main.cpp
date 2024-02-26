#include "stdafx.h"

#include <set>

#include "benchmark.hpp"
#include "context.hpp"
#include "model.generate.hpp"
#include "model.load.hpp"
#include "pipeline.hpp"
#include "pipeline.hybrid.hpp"
#include "pipeline.hybrid.v2.hpp"
#include "pipeline.hybrid.v3.hpp"
#include "pipeline.native.hpp"
#include "pipeline.virtual.hpp"
#include "pipeline.virtual.v2.hpp"
#include "pipeline.virtual.v3.hpp"
#include "pipeline.virtual.v4.hpp"
#include "renderer.hpp"
#include "vertex.hpp"
#include "vk.hpp"

/**/

// @formatter:off
static std::vector<benchmark_set> runs {
    { "gen://p-w-h", eNative | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-16-16", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-4-4", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-2-2", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-1-1", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-0.5-0.5", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-16-16", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-4-4", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-2-2", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-1-1", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-0.5-0.5", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    //
    { "gen://p-1-1-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-0.5-0.5-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-0.25-0.25-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://p-0.125-0.125-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-2-2-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-1-1-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-0.5-0.5-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "gen://ip-0.25-0.25-1843200", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "./models/stanford-bunny.obj", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "./models/happy.obj", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
    { "./models/clustered.obj", eNative | eVirtualV1 | eVirtualV2 | eVirtualV3 | eVirtualV4 | eHybridV1 | eHybridV2 | eHybridFinal },
};
// @formatter:on

static std::vector<const char*> vk_validation_layers = {
	#ifdef _DEBUG
	"VK_LAYER_KHRONOS_validation",
	#endif
	"VK_LAYER_KHRONOS_synchronization2",
	"VK_LAYER_KHRONOS_shader_object"
};

static std::vector<const char*> vk_extension_names = {
	VK_KHR_SURFACE_EXTENSION_NAME,
	VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
	#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
	#elif defined(VK_USE_PLATFORM_MI_KHR)
    VK_KHR_MIR_SURFACE_EXTENSION_NAME,
	#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
    VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
	#elif defined(VK_USE_PLATFORM_WIN32_KHR)
	VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
	#elif defined(VK_USE_PLATFORM_XLIB_KHR)
    VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
	#endif
	#if defined(_DEBUG) || TRUE
	VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
	VK_EXT_DEBUG_REPORT_EXTENSION_NAME
	#endif
};

static std::vector<const char*> vk_device_extensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME,
	VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
	VK_KHR_RELAXED_BLOCK_LAYOUT_EXTENSION_NAME,
	VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
	VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME
};

static std::vector<const char*> vk_device_validations = {
	#if defined(_DEBUG) || TRUE
	"VK_LAYER_LUNARG_standard_validation",
	"VK_LAYER_KHRONOS_validation"
	#endif
};

static vk::Format vk_depth_format = vk::Format::eD32Sfloat;

/**/

void setup_application(context& context_);

void setup_surface(context& context_);

void setup_device(context& context_);

void setup_renderpass(context& context_);

void setup_framebuffer(context& context_);

void setup_pipelines(context& context_);

void destroy(context& context_);

[[nodiscard]] dataset resolve_dataset(
	const context& context_,
	const std::filesystem::path& load_model_
);

[[nodiscard]] std::vector<std::shared_ptr<pipeline>> resolve_pipelines(
	const context& context_,
	benchmark_pipeline_bit_flag flag_bits_
);

/**/

int mainEntry() {

	context context {};

	/**/

	setup_application(context);
	setup_surface(context);
	setup_device(context);
	setup_renderpass(context);
	setup_framebuffer(context);
	setup_pipelines(context);

	/**/

	{
		renderer renderer { context.device };
		renderer.build(context, context.otf_count);

		/**/

		for (const auto& set : runs) {

			auto data = resolve_dataset(context, set.load_model);
			auto pipelines = resolve_pipelines(context, set.pipelines);

			/**/

			const auto text = std::format(
				"Started run {} with {} vertices.\n",
				set.load_model.string(),
				(data.vertices.size / sizeof(vertex))
			);

			std::cout << text;
			OutputDebugStringA(text.c_str());

			/**/

			for (const auto& pipeline : pipelines) {
				renderer.record(context, pipeline, data);
				renderer.execute(context);
			}

			/**/

			if (data.vertices.vkBuffer) {
				context.device.destroyBuffer(data.vertices.vkBuffer);
				context.device.freeMemory(data.vertices.vkDeviceMemory);
			}

			if (data.indices.vkBuffer) {
				context.device.destroyBuffer(data.indices.vkBuffer);
				context.device.freeMemory(data.indices.vkDeviceMemory);
			}

			if (data.camera.vkBuffer) {
				context.device.destroyBuffer(data.camera.vkBuffer);
				context.device.freeMemory(data.camera.vkDeviceMemory);
			}
		}
	}

	/**/

	destroy(context);
	return 0;
}

int main() {
	return mainEntry();
}

int WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
	return mainEntry();
}

/**/

void setup_application(context& context_) {

	uint32_t layer_count = 0uL;
	vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

	auto* layers = static_cast<VkLayerProperties*>(malloc(sizeof(VkLayerProperties) * layer_count));
	vkEnumerateInstanceLayerProperties(&layer_count, layers);

	uint32_t extension_count = 0uL;
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

	std::vector<VkExtensionProperties> properties {};
	properties.resize(extension_count);
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, properties.data());

	/**/

	free(layers);

	/**/

	vk::ApplicationInfo ai {
		"Rasterizer",
		0,
		"Rasterizer",
		0,
		VK_API_VERSION_1_3
	};

	vk::InstanceCreateInfo ici {
		vk::InstanceCreateFlags {},
		&ai,
		static_cast<uint32_t>(vk_validation_layers.size()),
		vk_validation_layers.data(),
		static_cast<uint32_t>(vk_extension_names.size()),
		vk_extension_names.data()
	};

	context_.instance = vk::createInstance(ici);
	assert(context_.instance);
}

void setup_surface(context& context_) {

	const uint32_t width = 640uL;
	const uint32_t height = 480uL;

	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::exit(-1);
	}

	SDL_Window* window = SDL_CreateWindow(
		"Rasterizer",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		width,
		height,
		SDL_WINDOW_VULKAN
	);
	assert(window);

	context_.window.native = window;

	/**/

	VkSurfaceKHR surface;
	if (not SDL_Vulkan_CreateSurface(window, context_.instance, &surface)) {
		std::exit(-1);
	}

	context_.window.surface = surface;
	context_.window.format = vk::Format::eB8G8R8A8Unorm;

	context_.window.width = width;
	context_.window.height = height;
}

void setup_device(context& context_) {

	const auto candidates = context_.instance.enumeratePhysicalDevices();
	for (const auto& candidate : candidates) {

		bool extension_support = false;
		bool surface_support = false;

		/**/

		const auto properties = candidate.enumerateDeviceExtensionProperties(nullptr);
		std::set<std::string> required_properties { vk_device_extensions.begin(), vk_device_extensions.end() };

		for (const auto& property : properties) {
			required_properties.erase(property.extensionName);
		}

		extension_support = required_properties.empty();

		/**/

		const auto capabilities = candidate.getSurfaceCapabilitiesKHR(context_.window.surface);
		const auto formats = candidate.getSurfaceFormatsKHR(context_.window.surface);
		const auto modes = candidate.getSurfacePresentModesKHR(context_.window.surface);

		surface_support = not formats.empty() && not modes.empty();

		/**/

		if (extension_support && surface_support) {
			context_.physical = candidate;
			break;
		}
	}

	assert(context_.physical);

	/**/

	std::vector<vk::DeviceQueueCreateInfo> dqci {};

	constexpr float default_queue_priority = 0.0F;
	const auto requested_queue_types = vk::QueueFlagBits::eGraphics;

	if (requested_queue_types & vk::QueueFlagBits::eGraphics) {

		context_.graphics_queue_index = vk_get_queue_family(context_.physical, vk::QueueFlagBits::eGraphics);
		dqci.emplace_back(
			vk::DeviceQueueCreateFlags {},
			context_.graphics_queue_index,
			1uL,
			&default_queue_priority
		);
	}

	if (requested_queue_types & vk::QueueFlagBits::eCompute) {

		context_.compute_queue_index = vk_get_queue_family(context_.physical, vk::QueueFlagBits::eCompute);
		dqci.emplace_back(
			vk::DeviceQueueCreateFlags {},
			context_.compute_queue_index,
			1uL,
			&default_queue_priority
		);
	}

	if (requested_queue_types & vk::QueueFlagBits::eTransfer) {

		context_.transfer_queue_index = vk_get_queue_family(context_.physical, vk::QueueFlagBits::eTransfer);

		if (
			context_.transfer_queue_index != context_.graphics_queue_index &&
			context_.transfer_queue_index != context_.compute_queue_index
		) {
			dqci.emplace_back(
				vk::DeviceQueueCreateFlags {},
				context_.transfer_queue_index,
				1uL,
				&default_queue_priority
			);
		} else {
			context_.transfer_queue_index = context_.graphics_queue_index;
		}
	}

	/**/

	const auto features = context_.physical.getFeatures();

	vk::PhysicalDeviceShaderAtomicInt64Features a64f {};
	a64f.shaderBufferInt64Atomics = VK_TRUE;
	a64f.shaderSharedInt64Atomics = VK_TRUE;

	vk::PhysicalDeviceVulkan13Features v13f {};
	v13f.maintenance4 = VK_TRUE;
	v13f.pNext = &a64f;

	const vk::DeviceCreateInfo dci {
		vk::DeviceCreateFlags {},
		static_cast<uint32_t>(dqci.size()),
		dqci.data(),
		static_cast<uint32_t>(vk_device_validations.size()),
		vk_device_validations.data(),
		static_cast<uint32_t>(vk_device_extensions.size()),
		vk_device_extensions.data(),
		&features,
		&v13f
	};

	context_.device = context_.physical.createDevice(dci);
	assert(context_.device);

	assert(context_.physical.getSurfaceSupportKHR(context_.graphics_queue_index, context_.window.surface));

	/**/

	context_.graphics_queue = context_.device.getQueue(context_.graphics_queue_index, 0uL);

	if (context_.graphics_queue_index == context_.transfer_queue_index) {
		context_.transfer_queue = context_.graphics_queue;
	}

	if (requested_queue_types & vk::QueueFlagBits::eCompute) {

		if (context_.graphics_queue_index == context_.compute_queue_index) {
			context_.compute_queue = context_.graphics_queue;
		}

		if (not context_.compute_queue) {
			context_.compute_queue = context_.device.getQueue(context_.compute_queue_index, 0uL);
		}

	} else {
		context_.compute_queue_index = context_.graphics_queue_index;
		context_.compute_queue = context_.graphics_queue;
	}

	if (requested_queue_types & vk::QueueFlagBits::eTransfer) {

		if (context_.compute_queue_index == context_.transfer_queue_index) {
			context_.transfer_queue = context_.compute_queue;
		}

		if (not context_.transfer_queue) {
			context_.transfer_queue = context_.device.getQueue(context_.transfer_queue_index, 0uL);
		}

	} else {
		context_.transfer_queue_index = context_.graphics_queue_index;
		context_.transfer_queue = context_.graphics_queue;
	}

	assert(context_.graphics_queue);
	assert(context_.compute_queue);
	assert(context_.transfer_queue);

	/**/

	vk::CommandPoolCreateInfo cpci {
		vk::CommandPoolCreateFlagBits::eTransient,
		context_.graphics_queue_index
	};
	context_.general_pool = context_.device.createCommandPool(cpci);
}

void setup_renderpass(context& context_) {

	std::vector<vk::AttachmentDescription> attachments {};
	std::vector<vk::SubpassDescription> subpasses {};
	std::vector<vk::SubpassDependency> dependencies {};

	/**/

	attachments.emplace_back(
		vk::AttachmentDescriptionFlags {},
		context_.window.format,
		vk::SampleCountFlagBits::e1,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::ePresentSrcKHR
	);

	attachments.emplace_back(
		vk::AttachmentDescriptionFlags {},
		vk_depth_format,
		vk::SampleCountFlagBits::e1,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal
	);

	vk::AttachmentReference subpass_color {
		0uL,
		vk::ImageLayout::eColorAttachmentOptimal
	};
	vk::AttachmentReference subpass_depth {
		1uL,
		vk::ImageLayout::eDepthStencilAttachmentOptimal
	};

	subpasses.emplace_back(
		vk::SubpassDescriptionFlags {},
		vk::PipelineBindPoint::eGraphics,
		0uL,
		nullptr,
		1uL,
		&subpass_color,
		nullptr,
		&subpass_depth,
		0uL,
		nullptr
	);

	dependencies.emplace_back(
		0uL,
		0uL,
		vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
		vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
		vk::AccessFlags {},
		vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
		vk::DependencyFlagBits::eByRegion
	);

	/**/

	vk::RenderPassCreateInfo rpci {
		vk::RenderPassCreateFlags {},
		static_cast<uint32_t>(attachments.size()), attachments.data(),
		static_cast<uint32_t>(subpasses.size()), subpasses.data(),
		static_cast<uint32_t>(dependencies.size()), dependencies.data()
	};
	auto render_pass = context_.device.createRenderPass(rpci);

	context_.native_render_pass = render_pass;

	/**/

	subpasses.clear();
	subpasses.emplace_back(
		vk::SubpassDescriptionFlags {},
		vk::PipelineBindPoint::eGraphics,
		0uL,
		nullptr,
		0uL,
		nullptr,
		nullptr,
		nullptr,
		0uL,
		nullptr
	);

	rpci = vk::RenderPassCreateInfo {
		vk::RenderPassCreateFlags {},
		0uL, nullptr,
		static_cast<uint32_t>(subpasses.size()), subpasses.data(),
		0uL, nullptr
	};
	render_pass = context_.device.createRenderPass(rpci);

	context_.hybrid_render_pass = render_pass;
}

void setup_framebuffer(context& context_) {

	std::vector<vk::Format> formats {};
	formats.emplace_back(context_.window.format);
	formats.emplace_back(vk::Format::eB8G8R8A8Uint);
	vk::ImageFormatListCreateInfo formatList { static_cast<uint32_t>(formats.size()), formats.data() };

	#if USE_ALIAS_SWAPCHAIN == FALSE

	vk::SwapchainCreateInfoKHR sci {
		vk::SwapchainCreateFlagBitsKHR::eMutableFormat,
		context_.window.surface,
		context_.otf_count,
		context_.window.format,
		vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear,
		vk::Extent2D { context_.window.width, context_.window.height },
		1uL,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
		vk::ImageUsageFlagBits::eStorage,
		vk::SharingMode::eExclusive,
		0uL,
		nullptr,
		vk::SurfaceTransformFlagBitsKHR::eIdentity,
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		vk::PresentModeKHR::eImmediate,
		VK_TRUE,
		nullptr,
		&formatList
	};
	auto swapchain = context_.device.createSwapchainKHR(sci);

	context_.swapchain.native = swapchain;

	const auto images = context_.device.getSwapchainImagesKHR(swapchain);
	for (auto&& image : images) {
		context_.swapchain.images.emplace_back(image, nullptr, 0uLL, 0uLL);
	}
	context_.otf_count = static_cast<uint32_t>(context_.swapchain.images.size());

	#else

	for (uint32_t i = 0; i < context_.otf_count; ++i) {

		const auto ici = vk::ImageCreateInfo {
			vk::ImageCreateFlagBits::eMutableFormat,
			vk::ImageType::e2D,
			vk::Format::eB8G8R8A8Uint,
			vk::Extent3D { context_.window.width, context_.window.height },
			1uL,
			1uL,
			vk::SampleCountFlagBits::e1,
			vk::ImageTiling::eLinear,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
			vk::ImageUsageFlagBits::eStorage,
			vk::SharingMode::eExclusive,
			0uL,
			nullptr,
			vk::ImageLayout::eUndefined, nullptr
		};
		const auto image = context_.device.createImage(ici);
		assert(image);

		const auto memory_requirements = context_.device.getImageMemoryRequirements(image);

		const auto mai = vk::MemoryAllocateInfo {
			memory_requirements.size,
			vk_get_memory_index(
				context_.physical,
				memory_requirements.memoryTypeBits,
				vk::MemoryPropertyFlagBits::eDeviceLocal
			)
		};
		const auto memory = context_.device.allocateMemory(mai);
		assert(memory);

		context_.device.bindImageMemory(image, memory, 0uL);

		context_.swapchain.images.emplace_back(image, memory, memory_requirements.size, 0uL);
		context_.otf_count = static_cast<uint32_t>(context_.swapchain.images.size());
	}

	#endif

	/**/

	context_.swapchain.depths.resize(context_.otf_count);
	context_.swapchain.alias_depth.resize(context_.otf_count);
	context_.swapchain.depth_memory.resize(context_.otf_count);
	context_.swapchain.depth_buffers.resize(context_.otf_count);
	context_.swapchain.attachments.resize(context_.otf_count);
	context_.swapchain.frames.resize(context_.otf_count);
	context_.swapchain.queries.resize(context_.otf_count);

	/**/

	for (uint32_t i = 0; i < context_.otf_count; ++i) {

		{
			vk::ImageCreateInfo ici {
				vk::ImageCreateFlags {},
				vk::ImageType::e2D,
				vk_depth_format,
				vk::Extent3D { context_.window.width, context_.window.height, 1uL },
				1uL,
				1uL,
				vk::SampleCountFlagBits::e1,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferDst,
				vk::SharingMode::eExclusive,
				0uL,
				nullptr,
				vk::ImageLayout::eUndefined
				//vk::ImageLayout::eDepthStencilAttachmentOptimal
			};
			auto depth_image = context_.device.createImage(ici);

			context_.swapchain.depths[i] = depth_image;

			/**/

			vk::ImageCreateInfo aici {
				vk::ImageCreateFlags {},
				vk::ImageType::e2D,
				vk::Format::eR32Uint,
				vk::Extent3D { context_.window.width, context_.window.height, 1uL },
				1uL,
				1uL,
				vk::SampleCountFlagBits::e1,
				vk::ImageTiling::eLinear,
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
				vk::SharingMode::eExclusive,
				0uL,
				nullptr,
				vk::ImageLayout::eUndefined
				//vk::ImageLayout::eDepthStencilAttachmentOptimal
			};
			auto alias_depth_image = context_.device.createImage(aici);

			context_.swapchain.alias_depth[i] = alias_depth_image;

			/**/

			auto memory_requirements = context_.device.getImageMemoryRequirements(depth_image);

			vk::MemoryAllocateInfo mai {
				memory_requirements.size,
				vk_get_memory_index(
					context_.physical,
					memory_requirements.memoryTypeBits,
					vk::MemoryPropertyFlagBits::eDeviceLocal
				)
			};
			auto memory = context_.device.allocateMemory(mai);

			context_.swapchain.depth_memory[i] = memory;

			/**/

			context_.device.bindImageMemory(depth_image, memory, 0uLL);
			context_.device.bindImageMemory(alias_depth_image, memory, 0uLL);
		}

		/**/

		{
			const auto dbbci = vk::BufferCreateInfo {
				vk::BufferCreateFlags {},
				sizeof(glm::u64) * context_.window.width * context_.window.height,
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				vk::SharingMode::eExclusive,
				VK_QUEUE_FAMILY_IGNORED,
				nullptr, nullptr
			};
			auto depth_buffer_x64 = context_.device.createBuffer(dbbci);
			assert(depth_buffer_x64);

			auto memory_requirements = context_.device.getBufferMemoryRequirements(depth_buffer_x64);

			const auto mai = vk::MemoryAllocateInfo {
				memory_requirements.size,
				vk_get_memory_index(
					context_.physical,
					memory_requirements.memoryTypeBits,
					vk::MemoryPropertyFlagBits::eDeviceLocal
				)
			};
			auto depth_mem_x64 = context_.device.allocateMemory(mai);
			assert(depth_mem_x64);

			/**/

			context_.device.bindBufferMemory(depth_buffer_x64, depth_mem_x64, 0uLL);

			/**/

			context_.swapchain.depth_buffers[i] = buffer {
				depth_buffer_x64,
				depth_mem_x64,
				dbbci.size,
				0uLL
			};
		}
	}

	/**/

	for (uint32_t i = 0; i < context_.otf_count; ++i) {

		vk::ImageViewCreateInfo civci {
			vk::ImageViewCreateFlags {},
			context_.swapchain.images[i].vkImage,
			vk::ImageViewType::e2D,
			context_.window.format,
			vk::ComponentMapping {
				vk::ComponentSwizzle::eR,
				vk::ComponentSwizzle::eG,
				vk::ComponentSwizzle::eB,
				vk::ComponentSwizzle::eA
			},
			vk::ImageSubresourceRange {
				vk::ImageAspectFlagBits::eColor,
				0uL,
				1uL,
				0uL,
				1uL
			}
		};
		auto swap_image_view = context_.device.createImageView(civci);

		context_.swapchain.attachments[i][0] = swap_image_view;

		/**/

		vk::ImageViewCreateInfo divci {
			vk::ImageViewCreateFlags {},
			context_.swapchain.depths[i],
			vk::ImageViewType::e2D,
			vk_depth_format,
			vk::ComponentMapping {
				vk::ComponentSwizzle::eR,
				vk::ComponentSwizzle::eG,
				vk::ComponentSwizzle::eB,
				vk::ComponentSwizzle::eA
			},
			vk::ImageSubresourceRange {
				vk::ImageAspectFlagBits::eDepth,
				0uL,
				1uL,
				0uL,
				1uL
			}
		};
		auto depth_image_view = context_.device.createImageView(divci);

		context_.swapchain.attachments[i][1] = depth_image_view;

		/**/

		vk::FramebufferCreateInfo fci {
			vk::FramebufferCreateFlags {},
			context_.native_render_pass,
			static_cast<uint32_t>(context_.swapchain.attachments[i].size()),
			context_.swapchain.attachments[i].data(),
			context_.window.width,
			context_.window.height,
			1uL
		};
		auto framebuffer = context_.device.createFramebuffer(fci);

		context_.swapchain.frames[i] = framebuffer;

		/**/

		vk::QueryPoolCreateInfo qpci {
			vk::QueryPoolCreateFlags {},
			vk::QueryType::eTimestamp,
			2uL
		};
		auto query_pool = context_.device.createQueryPool(qpci);

		context_.swapchain.queries[i] = query_pool;
	}

	/**/

	{
		vk::FramebufferCreateInfo fci {
			vk::FramebufferCreateFlags {},
			context_.hybrid_render_pass,
			0uL,
			nullptr,
			context_.window.width,
			context_.window.height,
			1uL
		};

		context_.hybrid_empty_frame_buffer = context_.device.createFramebuffer(fci);
		assert(context_.hybrid_empty_frame_buffer);
	}
}

void setup_pipelines(context& context_) {

	auto native_pipe = std::make_shared<native_pipeline>();
	auto virtual_pipe = std::make_shared<virtual_pipeline>();
	auto virtual_pipe_v2 = std::make_shared<virtual_pipeline_v2>();
	auto virtual_pipe_v3 = std::make_shared<virtual_pipeline_v3>();
	auto virtual_pipe_v4 = std::make_shared<virtual_pipeline_v4>();
	auto hybrid_pipe = std::make_shared<hybrid_pipeline>();
	auto hybrid_pipe_v2 = std::make_shared<hybrid_pipeline_v2>();
	auto hybrid_pipe_final = std::make_shared<hybrid_pipeline_v3>();

	/**/

	native_pipe->setup(context_, context_.device);
	virtual_pipe->setup(context_, context_.device);
	virtual_pipe_v2->setup(context_, context_.device);
	virtual_pipe_v3->setup(context_, context_.device);
	virtual_pipe_v4->setup(context_, context_.device);
	hybrid_pipe->setup(context_, context_.device);
	hybrid_pipe_v2->setup(context_, context_.device);
	hybrid_pipe_final->setup(context_, context_.device);

	context_.pipelines[benchmark_pipeline_bits::eNative] = _STD move(native_pipe);
	context_.pipelines[benchmark_pipeline_bits::eVirtualV1] = _STD move(virtual_pipe);
	context_.pipelines[benchmark_pipeline_bits::eVirtualV2] = _STD move(virtual_pipe_v2);
	context_.pipelines[benchmark_pipeline_bits::eVirtualV3] = _STD move(virtual_pipe_v3);
	context_.pipelines[benchmark_pipeline_bits::eVirtualV4] = _STD move(virtual_pipe_v4);
	context_.pipelines[benchmark_pipeline_bits::eHybridV1] = _STD move(hybrid_pipe);
	context_.pipelines[benchmark_pipeline_bits::eHybridV2] = _STD move(hybrid_pipe_v2);
	context_.pipelines[benchmark_pipeline_bits::eHybridFinal] = _STD move(hybrid_pipe_final);
}

dataset resolve_dataset(const context& context_, const std::filesystem::path& load_model_) {

	#ifdef _WIN32
	if (load_model_.native().contains(L"gen://")) {
		#else
    if (load_model_.native().contains(R"(gen://)")) {
		#endif
		return model_load_generate(context_, load_model_);
	}

	return model_load_file(context_, load_model_);
}

std::vector<std::shared_ptr<pipeline>> resolve_pipelines(
	const context& context_,
	benchmark_pipeline_bit_flag flag_bits_
) {

	std::vector<std::shared_ptr<pipeline>> collected {};

	if (flag_bits_ & eNative) {
		collected.push_back(context_.pipelines.at(eNative));
	}

	if (flag_bits_ & eVirtualV1) {
		collected.push_back(context_.pipelines.at(eVirtualV1));
	}

	if (flag_bits_ & eVirtualV2) {
		collected.push_back(context_.pipelines.at(eVirtualV2));
	}

	if (flag_bits_ & eVirtualV3) {
		collected.push_back(context_.pipelines.at(eVirtualV3));
	}

	if (flag_bits_ & eVirtualV4) {
		collected.push_back(context_.pipelines.at(eVirtualV4));
	}

	if (flag_bits_ & eHybridV1) {
		collected.push_back(context_.pipelines.at(eHybridV1));
	}

	if (flag_bits_ & eHybridV2) {
		collected.push_back(context_.pipelines.at(eHybridV2));
	}

	if (flag_bits_ & eHybridFinal) {
		collected.push_back(context_.pipelines.at(eHybridFinal));
	}

	return collected;
}

void destroy(context& context_) {

	context_.pipelines.clear();

	for (uint32_t i = 0; i < context_.otf_count; ++i) {
		context_.device.destroyFramebuffer(context_.swapchain.frames[i]);
		context_.device.destroyImageView(context_.swapchain.attachments[i][0]);
		context_.device.destroyImageView(context_.swapchain.attachments[i][1]);
		context_.device.destroyImage(context_.swapchain.depths[i]);
		context_.device.destroyImage(context_.swapchain.alias_depth[i]);
		context_.device.freeMemory(context_.swapchain.depth_memory[i]);
		context_.device.destroyBuffer(context_.swapchain.depth_buffers[i].vkBuffer);
		context_.device.freeMemory(context_.swapchain.depth_buffers[i].vkDeviceMemory);
		context_.device.destroyQueryPool(context_.swapchain.queries[i]);
	}

	context_.device.destroyFramebuffer(context_.hybrid_empty_frame_buffer);

	#if USE_ALIAS_SWAPCHAIN == FALSE
	context_.device.destroySwapchainKHR(context_.swapchain.native);
	#else
	for (auto&& image : context_.swapchain.images) {
		context_.device.destroyImage(std::exchange(image.vkImage, nullptr));
		context_.device.freeMemory(std::exchange(image.vkDeviceMemory, nullptr));
	}
	context_.swapchain.images.clear();
	#endif

	context_.device.destroyRenderPass(context_.native_render_pass);
	context_.device.destroyRenderPass(context_.hybrid_render_pass);

	context_.device.destroyCommandPool(context_.general_pool);

	context_.transfer_queue;
	context_.compute_queue;
	context_.graphics_queue;

	vkDestroyDevice(context_.device, nullptr);

	vkDestroySurfaceKHR(context_.instance, context_.window.surface, nullptr);
	//SDL_DestroyWindowSurface(context_.window.native);
	SDL_DestroyWindow(context_.window.native);

	vkDestroyInstance(context_.instance, nullptr);
}
