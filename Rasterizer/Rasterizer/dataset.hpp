#pragma once

#include "stdafx.h"
#include "buffer.hpp"

struct dataset {
	buffer camera = {};
	buffer vertices = {};
	buffer indices = {};
};
