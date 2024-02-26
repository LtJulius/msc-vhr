#include "stdafx.h"
#include "shader.hpp"

#include <fstream>

std::vector<uint32_t> read_shader_file(std::filesystem::path path_) {

	path_ = std::filesystem::absolute(path_);
	if (not std::filesystem::exists(path_)) {
		assert(false);
		std::exit(-1);
	}

	auto stream = std::ifstream(path_.native(), std::ios::in | std::ios::binary);

	stream.seekg(0, std::ios::end);
	auto fsize = stream.tellg();
	stream.seekg(0, std::ios::beg);

	std::vector<uint32_t> spirv_code {};
	spirv_code.resize(fsize / sizeof(uint32_t), 0uL);

	stream.read(reinterpret_cast<char*>(spirv_code.data()), fsize);
	assert(not stream.fail());

	return spirv_code;
}
