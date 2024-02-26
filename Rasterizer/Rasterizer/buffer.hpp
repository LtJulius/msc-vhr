#pragma once

#include "stdafx.h"

struct buffer {
	vk::Buffer vkBuffer = nullptr;
	vk::DeviceMemory vkDeviceMemory = nullptr;

	size_t size = 0;
	size_t offset = 0;
};
