#pragma once

#include "stdafx.h"

struct vertex {
	glm::vec4 position = {};
	glm::u8vec4 color = {};
};

static_assert(sizeof(vertex) == sizeof(float) * 4 + sizeof(uint8_t) * 4);
static_assert(alignof(vertex) == alignof(float));
