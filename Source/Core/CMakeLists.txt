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
    Utilities/Functor.hpp
    Utilities/Functor.cpp
)
set (UTILITIES_SRC_GROUP_NAME "Utilities")

set (VULKAN_CORE_SRC 
    Vulkan/Native/Instance.hpp
    Vulkan/Native/Instance.cpp
    Vulkan/Native/DebugUtils.hpp
    Vulkan/Native/DebugUtils.cpp
    Vulkan/Native/Surface.hpp
    Vulkan/Native/Surface.cpp
    Vulkan/Native/PhysicalDevice.hpp
    Vulkan/Native/PhysicalDevice.cpp
    Vulkan/Native/Device.hpp
    Vulkan/Native/Device.cpp
    Vulkan/Native/MemoryAllocator.hpp
    Vulkan/Native/MemoryAllocator.cpp
    Vulkan/Native/Swapchain.hpp
    Vulkan/Native/Swapchain.cpp
    Vulkan/Native/Texture.hpp
    Vulkan/Native/Texture.cpp
    Vulkan/Native/SyncStructures.hpp
    Vulkan/Native/SyncStructures.cpp
    Vulkan/Native/Commands.hpp
    Vulkan/Native/Commands.cpp
    Vulkan/Native/Descriptors.hpp
    Vulkan/Native/Descriptors.cpp
    Vulkan/Native/Pipeline.hpp
    Vulkan/Native/Pipeline.cpp
    Vulkan/Native/Buffer.hpp
    Vulkan/Native/Buffer.cpp
    Vulkan/Native/Sampler.hpp
    Vulkan/Native/Sampler.cpp
    Vulkan/Native/Shader.hpp
    Vulkan/Native/Shader.cpp
    Vulkan/Native/RenderResource.cpp
    Vulkan/Native/RenderResource.hpp
    Vulkan/Native/DrawCallMetaData.h
    Vulkan/Native/DrawCallMetaData.cpp
    
    Vulkan/Manager/Context.hpp
    Vulkan/Manager/Context.cpp
    Vulkan/Manager/CommandManager.cpp
    Vulkan/Manager/CommandManager.hpp
    Vulkan/Manager/RenderResourceManager.hpp
    Vulkan/Manager/RenderResourceManager.cpp
    Vulkan/Manager/PipelineManager.hpp
    Vulkan/Manager/PipelineManager.cpp
    Vulkan/Manager/ShaderManager.hpp
    Vulkan/Manager/ShaderManager.cpp
    Vulkan/Manager/DescriptorManager.hpp
    Vulkan/Manager/DescriptorManager.cpp
    Vulkan/Manager/DrawCallManager.h
    Vulkan/Manager/DrawCallManager.cpp

    Vulkan/EngineCore.hpp
    Vulkan/EngineCore.cpp
    
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

target_link_libraries(Core
    PUBLIC
        spirv-cross-c)


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