#pragma once

#include "stdafx.h"

enum benchmark_pipeline_bits : unsigned char {
	eNative = 0b0000'0001,
	eVirtualV1 = 0b0000'0010,
	eVirtualV2 = 0b0000'0100,
	eVirtualV3 = 0b0000'1000,
	eVirtualV4 = 0b0001'0000,
	eHybridV1 = 0b0010'0000,
	eHybridV2 = 0b0100'0000,
	eHybridFinal = 0b1000'0000
};

using benchmark_pipeline_bit_flag = std::underlying_type_t<benchmark_pipeline_bits>;

struct benchmark_set {
	std::filesystem::path load_model = {};
	benchmark_pipeline_bit_flag pipelines = {};
};
