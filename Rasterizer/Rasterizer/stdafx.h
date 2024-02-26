#pragma once

#include <concepts>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

/**/

#define USE_ALIAS_SWAPCHAIN FALSE

/* GLM */

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/trigonometric.hpp>

/* Vulkan API */

#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

// #define VULKAN_HPP_DISABLE_ENHANCED_MODE
// #define VULKAN_HPP_NO_CONSTRUCTORS
// #define VULKAN_HPP_NO_SETTERS
// #define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_SPACESHIP_OPERATOR
#include <vulkan/vulkan.hpp>

/* SDL */

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
