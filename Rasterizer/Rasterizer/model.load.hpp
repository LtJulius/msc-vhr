#pragma once
#include "stdafx.h"

#include "context.hpp"
#include "dataset.hpp"

extern dataset model_load_file(const context& context_, const std::filesystem::path& load_model_);
