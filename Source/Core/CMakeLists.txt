find_package (Vulkan REQUIRED)

set (CMAKE_CXX_STANDARD 20)

set (UTILITIES_SRC
        Utilities/Logger.h
        Utilities/VulkanUtilities.h
        Utilities/VulkanUtilities.cpp
        Utilities/MemoryPool.h
        Utilities/MemoryPool.cpp
        Utilities/Defines.h
        Utilities/Camera.h
        Utilities/Camera.cpp
        Utilities/Functor.hpp
        Utilities/Timer.h
        Utilities/Timer.cpp
        Utilities/GUI.h
        Utilities/GUI.cpp
        Utilities/Threading/Thread.hpp
        Utilities/Profiler.h
        Utilities/Profiler.cpp
)
set (UTILITIES_SRC_GROUP_NAME "Utilities")

set (APPLICATION_SRC
        Application/Application.h
        Application/Application.cpp
        Application/EntryPoint.h
        Application/EntryPoint.cpp
)
set (APPLICATION_SRC_GROUP_NAME "Application")

set (VULKAN_CORE_SRC
        Vulkan/Native/Instance.h
        Vulkan/Native/Instance.cpp
        Vulkan/Native/DebugUtils.h
        Vulkan/Native/DebugUtils.cpp
        Vulkan/Native/Surface.h
        Vulkan/Native/Surface.cpp
        Vulkan/Native/PhysicalDevice.h
        Vulkan/Native/PhysicalDevice.cpp
        Vulkan/Native/Device.h
        Vulkan/Native/Device.cpp
        Vulkan/Native/Queue.h
        Vulkan/Native/Queue.cpp
        Vulkan/Native/MemoryAllocator.h
        Vulkan/Native/MemoryAllocator.cpp
        Vulkan/Native/Swapchain.h
        Vulkan/Native/Swapchain.cpp
        Vulkan/Native/Texture.h
        Vulkan/Native/Texture.cpp
        Vulkan/Native/SyncStructures.h
        Vulkan/Native/SyncStructures.cpp
        Vulkan/Native/Commands.h
        Vulkan/Native/Commands.cpp
        Vulkan/Native/Descriptors.h
        Vulkan/Native/Descriptors.cpp
        Vulkan/Native/DescriptorSetAllocator.h
        Vulkan/Native/DescriptorSetAllocator.cpp
        Vulkan/Native/Pipeline.h
        Vulkan/Native/Pipeline.cpp
        Vulkan/Native/Image.h
        Vulkan/Native/Image.cpp
        Vulkan/Native/Buffer.h
        Vulkan/Native/Buffer.cpp
        Vulkan/Native/Sampler.h
        Vulkan/Native/Sampler.cpp
        Vulkan/Native/Shader.h
        Vulkan/Native/Shader.cpp
        Vulkan/Native/RenderResource.cpp
        Vulkan/Native/RenderResource.h
        Vulkan/Native/Resource.cpp
        Vulkan/Native/Resource.h
        Vulkan/Native/DrawCallMetaData.h
        Vulkan/Native/DrawCallMetaData.cpp
        Vulkan/Native/DGCSeqLayout.h
        Vulkan/Native/DGCSeqLayout.cpp
        Vulkan/Native/DGCSequence.cpp
        Vulkan/Native/DGCSequence.h
        Vulkan/Native/DGCSeqRenderLayout.cpp
        Vulkan/Native/DGCSeqRenderLayout.h
        Vulkan/Native/DGCSeqDataBufPoolResource.cpp
        Vulkan/Native/DGCSeqDataBufPoolResource.h
        Vulkan/Native/QueryPool.h
        Vulkan/Native/QueryPool.cpp

        Vulkan/Manager/VulkanContext.h
        Vulkan/Manager/VulkanContext.cpp
        Vulkan/Manager/RenderContext.h
        Vulkan/Manager/RenderContext.cpp
        Vulkan/Manager/CommandManager.cpp
        Vulkan/Manager/CommandManager.h
        Vulkan/Manager/RenderResourceManager.h
        Vulkan/Manager/RenderResourceManager.cpp
        Vulkan/Manager/PipelineManager.h
        Vulkan/Manager/PipelineManager.cpp
        Vulkan/Manager/ShaderManager.h
        Vulkan/Manager/ShaderManager.cpp
        Vulkan/Manager/DrawCallManager.h
        Vulkan/Manager/DrawCallManager.cpp
        Vulkan/Manager/RenderFrame.h
        Vulkan/Manager/RenderFrame.cpp
        Vulkan/Manager/DGCSeqManager.cpp
        Vulkan/Manager/DGCSeqManager.h

        Vulkan/RenderGraph/RenderPassBindingInfo.h
        Vulkan/RenderGraph/RenderPassBindingInfo.cpp
        Vulkan/RenderGraph/RenderSequence.h
        Vulkan/RenderGraph/RenderSequence.cpp
        Vulkan/RenderGraph/RenderSequenceConfig.h
        Vulkan/RenderGraph/RenderSequenceConfig.cpp
)
set (VULKAN_CORE_SRC_GROUP_NAME "Vulkan")

set (PLATFORM_SRC
        Platform/Window.h
        Platform/Window.cpp
        Platform/Input.h
        Platform/Input.cpp
)
set (PLATFORM_SRC_GROUP_NAME "Platform")

set (MODEL_SRC
        Model/Mesh.h
        Model/Mesh.cpp
        Model/GPUGeometryData.cpp
        Model/GPUGeometryData.h
        Model/GPUGeometryDataManager.cpp
        Model/GPUGeometryDataManager.h
        Model/ModelDataManager.h
        Model/ModelDataManager.cpp
)
set (MODEL_SRC_GROUP_NAME "Model")

set (SCENE_GRAPH_SRC
        SceneGraph/Scene.h
        SceneGraph/Scene.cpp
        SceneGraph/Node.h
        SceneGraph/Node.cpp
        SceneGraph/NodeProxy.hpp
)
set (SCENE_GRAPH_SRC_GROUP_NAME "SceneGraph")

source_group (${UTILITIES_SRC_GROUP_NAME} FILES ${UTILITIES_SRC})
source_group (${VULKAN_CORE_SRC_GROUP_NAME} FILES ${VULKAN_CORE_SRC})
source_group (${PLATFORM_SRC_GROUP_NAME} FILES ${PLATFORM_SRC})
source_group (${MODEL_SRC_GROUP_NAME} FILES ${MODEL_SRC})
source_group (${APPLICATION_SRC_GROUP_NAME} FILES ${APPLICATION_SRC})
source_group (${SCENE_GRAPH_SRC_GROUP_NAME} FILES ${SCENE_GRAPH_SRC})

add_library (Core STATIC
        ${UTILITIES_SRC}
        ${PLATFORM_SRC}
        ${VULKAN_CORE_SRC}
        ${MODEL_SRC}
        ${APPLICATION_SRC}
        ${SCENE_GRAPH_SRC}
        ../../External/tracy/public/TracyClient.cpp
)

set_target_properties (Core PROPERTIES FOLDER ${LIB_FOLDER_NAME})

if (MSVC AND NOT USE_CLANG)
        target_compile_options (Core
                PUBLIC
                "/MP")
endif ()

# <TODO>
target_compile_definitions (Core
        PUBLIC
        VK_USE_PLATFORM_WIN32_KHR
        NOMINMAX)

target_compile_definitions (Core
        PUBLIC
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        TRACY_ENABLE=1
        TRACY_VK_USE_SYMBOL_TABLE=1)

target_link_libraries (Core
        PUBLIC
        Vulkan::Headers
        Vulkan::shaderc_combined)

target_include_directories (Core
        PUBLIC
        ${PROJECT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/External/tracy/public)

# target_include_directories(Core
# PUBLIC
# ${PROJECT_SOURCE_DIR}/External/Output/Include)

# include("../../External/Output/cmake/SDL2Config.cmake")
target_link_libraries (Core
        PUBLIC
        SDL2::SDL2
        spirv-cross-c
        ImGuiFileDialog
        ImGuizmo)

# include("../../External/Output/lib/cmake/assimp-5.4/assimpConfig.cmake")
set (INTELLIGENDESIGN_CODE_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository")
set (INTELLIGENDESIGN_LIB_PATH "${PROJECT_SOURCE_DIR}/../IntelligenDesign/Git-IntelligenDesignToolsCode/Repository/LibraryOutputs/x64")

target_include_directories (Core
        PUBLIC
        ${INTELLIGENDESIGN_CODE_PATH}
)

add_library (INTELLI_DS_MemoryPool_DEBUG STATIC IMPORTED)
add_library (INTELLI_DS_MemoryPool_RELEASE STATIC IMPORTED)

# if (MSVC)
set_target_properties (INTELLI_DS_MemoryPool_DEBUG
        PROPERTIES
        IMPORTED_LOCATION ${INTELLIGENDESIGN_LIB_PATH}/Debug/LibMemoryPool.lib)
set_target_properties (INTELLI_DS_MemoryPool_RELEASE
        PROPERTIES
        IMPORTED_LOCATION ${INTELLIGENDESIGN_LIB_PATH}/Release/LibMemoryPool.lib)

# endif(MSVC)
target_link_libraries (Core
        PUBLIC
        debug INTELLI_DS_MemoryPool_DEBUG
        optimized INTELLI_DS_MemoryPool_RELEASE)

add_subdirectory (CISDIModel)

target_link_libraries (Core
        PUBLIC
        CISDI_ModelData)