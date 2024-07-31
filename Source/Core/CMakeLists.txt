find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 20)

set (UTILITIES_SRC 
    Utilities/Logger.hpp
    Utilities/VulkanUtilities.hpp
    Utilities/VulkanUtilities.cpp
    Utilities/MemoryPool.hpp
    Utilities/MemoryPool.cpp
    Utilities/Defines.hpp
)
set (UTILITIES_SRC_GROUP_NAME "Utilities")

set (VULKAN_CORE_SRC 
    VulkanCore/VulkanHelper.hpp
    VulkanCore/VulkanEngine.hpp
    VulkanCore/VulkanEngine.cpp
    VulkanCore/VulkanInstance.hpp
    VulkanCore/VulkanInstance.cpp
    VulkanCore/VulkanDebugUtils.hpp
    VulkanCore/VulkanDebugUtils.cpp
    VulkanCore/VulkanSurface.hpp
    VulkanCore/VulkanSurface.cpp
    VulkanCore/VulkanPhysicalDevice.hpp
    VulkanCore/VulkanPhysicalDevice.cpp
    VulkanCore/VulkanDevice.hpp
    VulkanCore/VulkanDevice.cpp
    VulkanCore/VulkanContext.hpp
    VulkanCore/VulkanContext.cpp
    VulkanCore/VulkanMemoryAllocator.hpp
    VulkanCore/VulkanMemoryAllocator.cpp
    VulkanCore/VulkanSwapchain.hpp
    VulkanCore/VulkanSwapchain.cpp
    VulkanCore/VulkanImage.hpp
    VulkanCore/VulkanImage.cpp
    VulkanCore/VulkanSyncStructures.hpp
    VulkanCore/VulkanSyncStructures.cpp
    VulkanCore/VulkanCommands.hpp
    VulkanCore/VulkanCommands.cpp
    VulkanCore/VulkanCommandManager.cpp
    VulkanCore/VulkanCommandManager.hpp
    VulkanCore/VulkanFrameObjects.hpp
    VulkanCore/VulkanFrameObjects.cpp
    VulkanCore/VulkanDescriptors.hpp
    VulkanCore/VulkanDescriptors.cpp
    VulkanCore/VulkanPipeline.hpp
    VulkanCore/VulkanPipeline.cpp
    VulkanCore/VulkanBuffer.hpp
    VulkanCore/VulkanBuffer.cpp
    VulkanCore/VulkanSampler.hpp
    VulkanCore/VulkanSampler.cpp
    VulkanCore/VulkanShader.hpp
    VulkanCore/VulkanShader.cpp
)
set (VULKAN_CORE_SRC_GROUP_NAME "VulkanCore")

set (PLATFORM_SRC 
    Platform/Window.hpp 
    Platform/Window.cpp
)
set (PLATFORM_SRC_GROUP_NAME "Platform")

set (MODEL_SRC
    Model/MeshType.hpp
    Model/Mesh.hpp
    Model/Mesh.cpp
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
        Vulkan::Headers)

target_include_directories(Core 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/Source)

target_include_directories(Core 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/External/Output/Include/SDL/include)
        
add_library(SDL_DEBUG SHARED IMPORTED)
add_library(SDL_RELEASE SHARED IMPORTED)

if (MSVC)
    set_target_properties(SDL_DEBUG PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/External/Output/Bin/SDL2d.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/External/Output/Lib/Debug/SDL2d.lib)
    set_target_properties(SDL_RELEASE PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/External/Output/Bin/SDL2.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/External/Output/Lib/Release/SDL2.lib)
endif(MSVC)

target_link_libraries(Core
    PUBLIC
        debug SDL_DEBUG
        optimized SDL_RELEASE)


set (INTELLIGENDESIGN_CODE_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository")

target_include_directories(Core
    PUBLIC
        ${INTELLIGENDESIGN_CODE_PATH}
)

add_library(INTELLI_DS_MemoryPool_DEBUG STATIC IMPORTED)
add_library(INTELLI_DS_MemoryPool_RELEASE STATIC IMPORTED)

set (INTELLIGENDESIGN_DEMO_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsDemo/Repository")

if (MSVC)
    set_target_properties(INTELLI_DS_MemoryPool_DEBUG 
        PROPERTIES
            IMPORTED_LOCATION ${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Debug/LibMemoryPool.lib
            IMPORTED_IMPLIB ${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Debug/LibMemoryPool.lib)
    set_target_properties(INTELLI_DS_MemoryPool_RELEASE 
        PROPERTIES
            IMPORTED_LOCATION ${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Release/LibMemoryPool.lib
            IMPORTED_IMPLIB $${INTELLIGENDESIGN_DEMO_PATH}/LibraryOutputs/x64/Release/LibMemoryPool.lib)
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
        <thread>
        <span>
        <fstream>
        <unordered_map>)