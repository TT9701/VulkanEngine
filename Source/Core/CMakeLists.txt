find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 20)

set (UTILITIES_SRC 
    Utilities/Logger.hpp
    Utilities/VulkanUtilities.hpp
    Utilities/VulkanUtilities.cpp
    Utilities/MemoryPool.hpp
    Utilities/MemoryPool.cpp
    Utilities/Defines.hpp
    Utilities/Camera.hpp
    Utilities/Camera.cpp
)
set (UTILITIES_SRC_GROUP_NAME "Utilities")

set (VULKAN_CORE_SRC 
    Vulkan/VulkanHelper.hpp
    Vulkan/EngineCore.hpp
    Vulkan/EngineCore.cpp
    Vulkan/Instance.hpp
    Vulkan/Instance.cpp
    Vulkan/DebugUtils.hpp
    Vulkan/DebugUtils.cpp
    Vulkan/Surface.hpp
    Vulkan/Surface.cpp
    Vulkan/PhysicalDevice.hpp
    Vulkan/PhysicalDevice.cpp
    Vulkan/Device.hpp
    Vulkan/Device.cpp
    Vulkan/Context.hpp
    Vulkan/Context.cpp
    Vulkan/MemoryAllocator.hpp
    Vulkan/MemoryAllocator.cpp
    Vulkan/Swapchain.hpp
    Vulkan/Swapchain.cpp
    Vulkan/Texture.hpp
    Vulkan/Texture.cpp
    Vulkan/SyncStructures.hpp
    Vulkan/SyncStructures.cpp
    Vulkan/Commands.hpp
    Vulkan/Commands.cpp
    Vulkan/CommandManager.cpp
    Vulkan/CommandManager.hpp
    Vulkan/Descriptors.hpp
    Vulkan/Descriptors.cpp
    Vulkan/Pipeline.hpp
    Vulkan/Pipeline.cpp
    Vulkan/Buffer.hpp
    Vulkan/Buffer.cpp
    Vulkan/Sampler.hpp
    Vulkan/Sampler.cpp
    Vulkan/Shader.hpp
    Vulkan/Shader.cpp
    Vulkan/RenderResource.cpp
    Vulkan/RenderResource.hpp
    Vulkan/RenderResourceManager.hpp
    Vulkan/RenderResourceManager.cpp
)
set (VULKAN_CORE_SRC_GROUP_NAME "Vulkan")

set (PLATFORM_SRC 
    Platform/Window.hpp 
    Platform/Window.cpp
)
set (PLATFORM_SRC_GROUP_NAME "Platform")

set (MODEL_SRC
    Model/MeshType.hpp
    Model/Mesh.hpp
    Model/Mesh.cpp
    Model/Model.cpp
    Model/Model.hpp
    Model/CISDI_3DModelConverter.hpp
    Model/CISDI_3DModelConverter.cpp
    Model/CISDI_3DModelData.hpp
    Model/CISDI_3DModelData.cpp
)
set (MODEL_SRC_GROUP_NAME "Model")

source_group(${UTILITIES_SRC_GROUP_NAME} FILES ${UTILITIES_SRC})
source_group(${VULKAN_CORE_SRC_GROUP_NAME} FILES ${VULKAN_CORE_SRC})
source_group(${PLATFORM_SRC_GROUP_NAME} FILES ${PLATFORM_SRC})
source_group(${MODEL_SRC_GROUP_NAME} FILES ${MODEL_SRC})

add_library(Core STATIC
    ${UTILITIES_SRC}
    ${PLATFORM_SRC}
    ${VULKAN_CORE_SRC}
    ${MODEL_SRC}
)

set_target_properties(Core PROPERTIES FOLDER ${LIB_FOLDER_NAME})

if(MSVC AND NOT USE_CLANG)
    target_compile_options(Core 
        PUBLIC 
            "/MP")
endif()

target_compile_definitions(Core 
    PUBLIC 
        GLM_ENABLE_EXPERIMENTAL
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS)

# <TODO>
target_compile_definitions(Core 
    PUBLIC 
        VK_USE_PLATFORM_WIN32_KHR
        NOMINMAX)

target_compile_definitions(Core 
    PUBLIC 
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 )

target_link_libraries(Core 
    PUBLIC
        Vulkan::Headers
        Vulkan::shaderc_combined)

target_link_libraries(CUDA
    PUBLIC
        GPUOpen::VulkanMemoryAllocator)

target_include_directories(Core 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/Source)

# target_include_directories(Core 
#     PUBLIC 
#         ${PROJECT_SOURCE_DIR}/External/Output/Include)

# include("../../External/Output/cmake/SDL2Config.cmake")
target_link_libraries(Core
    PUBLIC 
        SDL2::SDL2)

# include("../../External/Output/lib/cmake/assimp-5.4/assimpConfig.cmake")
target_link_libraries(Core
    PUBLIC
        assimp::assimp)

target_link_libraries(Core
    PUBLIC
        meshoptimizer)

set (INTELLIGENDESIGN_CODE_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository")

target_include_directories(Core
    PUBLIC
        ${INTELLIGENDESIGN_CODE_PATH}
)

add_library(INTELLI_DS_MemoryPool_DEBUG STATIC IMPORTED)
add_library(INTELLI_DS_MemoryPool_RELEASE STATIC IMPORTED)

if (MSVC)
    set_target_properties(INTELLI_DS_MemoryPool_DEBUG 
        PROPERTIES
            IMPORTED_LOCATION ${INTELLIGENDESIGN_CODE_PATH}/LibraryOutputs/x64/Debug/LibMemoryPool.lib)
    set_target_properties(INTELLI_DS_MemoryPool_RELEASE 
        PROPERTIES
            IMPORTED_LOCATION ${INTELLIGENDESIGN_CODE_PATH}/LibraryOutputs/x64/Release/LibMemoryPool.lib)
endif(MSVC)

target_link_libraries(Core
    PUBLIC
        debug INTELLI_DS_MemoryPool_DEBUG
        optimized INTELLI_DS_MemoryPool_RELEASE)

target_precompile_headers(Core 
    PUBLIC 
        <functional>
        <deque>
        <optional> 
        <vector> 
        <memory> 
        <string> 
        <tuple>
        <thread>
        <span>
        <fstream>
        <iostream>
        <variant>
        <filesystem>
        <unordered_map>)