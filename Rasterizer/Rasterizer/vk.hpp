#pragma once

#include "stdafx.h"

inline uint32_t vk_get_memory_index(
	const vk::PhysicalDevice physical_,
	const uint32_t types_,
	const vk::MemoryPropertyFlags flags_
) {

	const auto memory_properties = physical_.getMemoryProperties();

	for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
		if ((types_ & (1 << i)) && (memory_properties.memoryTypes[i].propertyFlags & flags_) == flags_) {
			return i;
		}
	}

	throw std::runtime_error("Unable to find suitable memory type.");
}

inline uint32_t vk_get_queue_family(vk::PhysicalDevice physical_, vk::QueueFlagBits flag_) {

	const auto properties = physical_.getQueueFamilyProperties();

	// Dedicated Compute
	if (flag_ & vk::QueueFlags { vk::QueueFlagBits::eCompute }) {
		for (uint32_t i = 0; i < properties.size(); ++i) {
			if (properties[i].queueFlags & flag_ && !(properties[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
				return i;
			}
		}
	}

	// Dedicated Transfer
	if (flag_ & vk::QueueFlags { vk::QueueFlagBits::eTransfer }) {
		for (uint32_t i = 0; i < properties.size(); ++i) {
			if (
				properties[i].queueFlags & flag_ &&
				!(properties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
				!(properties[i].queueFlags & vk::QueueFlagBits::eCompute)
			) {
				return i;
			}
		}
	}

	// Shared Transfer & Compute
	if (flag_ & vk::QueueFlagBits::eTransfer || flag_ & vk::QueueFlagBits::eCompute) {
		for (uint32_t i = 0; i < properties.size(); ++i) {
			if (properties[i].queueFlags & flag_ && !(properties[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
				return i;
			}
		}
	}

	// First Match
	for (uint32_t i = 0; i < properties.size(); ++i) {
		if (properties[i].queueCount > 0 && properties[i].queueFlags & flag_) {
			return i;
		}
	}

	throw std::runtime_error("Could not find matching queue family.");
}
