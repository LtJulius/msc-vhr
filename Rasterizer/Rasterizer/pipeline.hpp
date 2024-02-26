#pragma once
#include "stdafx.h"
#include "dataset.hpp"

struct context;

class pipeline {
public:
	using this_type = pipeline;

public:
	constexpr pipeline() noexcept = default;

	constexpr virtual ~pipeline() noexcept = default;

public:
	virtual void setup(const context& context_, vk::Device device_) = 0;

	virtual void record(
		const context& context_,
		const uint32_t idx_,
		const dataset& dataset_,
		vk::CommandBuffer& rootCmdBuffer_
	) = 0;
};
