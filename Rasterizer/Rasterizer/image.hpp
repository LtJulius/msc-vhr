#pragma once

#include "stdafx.h"

struct image {
	vk::Image vkImage = nullptr;
	vk::DeviceMemory vkDeviceMemory = nullptr;

	size_t size = 0;
	size_t offset = 0;
};
