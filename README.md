# MSc. Vulkan Hybrid Rasterization - Implementation

## Requirements

* Visual Studio 2022
* [Vulkan SDK](https://vulkan.lunarg.com)

## Quick Use

The solution can be opened by visual studio 2022 and later.

There are multiple switches related to debug and release build. If you want to run the code in headless auto-benchmark mode, just redefine the `USE_SWAPCHAIN_ALIAS` macro respectively.

In addition there is a powershell script to compile the referenced glsl-code into spirv files using the `glslangValidator` of the Vulkan SDK by using environment variables to find the required executables.

```powershell
powershell -executionpolicy bypass -File .\compile.ps1
```

In addition, if you want to run the executable, please make sure that you have the `shader`-directory and `model`-directory within the current working directory, as the executable will try to find the corresponding spirv files and obj model files using relative file-resolving.
