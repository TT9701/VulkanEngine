/**
 * @file CISDI_3DModelData.h
 * @author 
 * @brief 采用 pmr allocator 的 CISDI_3DModel 数据结构，使用 FBXSDK 与 Assimp 结合的方式导入 FBX 文件，其余文件使用 Assimp 导入。
          CISDI_3DModel 包含渲染所需的所有数据，包括节点、网格、材质、各级包围盒等。
 * @version 0.4.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "Source/Common/Common.h"

#ifdef CISDI_MODEL_DATA_EXPORTS
#define CISDI_MODEL_DATA_API __declspec(dllexport)
#else
#define CISDI_MODEL_DATA_API __declspec(dllimport)
#endif

constexpr bool bUseCombinedImport = true;

namespace IntelliDesign_NS::ModelData {

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;

constexpr Version CISDI_3DModel_VERSION = {0ui8, 4ui8, 2ui16};

/**
 * @brief CISDI_3DModel 包含渲染所需的所有数据，包括节点、网格、材质、各级包围盒等。
 */
struct CISDI_3DModel {
    /**
     * @brief 使用 pmr allocator 构造 CISDI_3DModel 对象
     * @param pMemPool pmr memory resource 指针
     */
    CISDI_3DModel(::std::pmr::memory_resource* pMemPool);

    /**
     * @brief CISDI_3DModel::Header 包含模型的基本信息，包括版本号、节点总数、网格总数、材质总数。
     */
    struct Header {
        uint64_t header {0};         ///<- 标识头部信息
        Version version {};          ///<- 版本号
        uint32_t nodeCount {0};      ///<- 节点总数
        uint32_t meshCount {0};      ///<- 网格总数
        uint32_t materialCount {0};  ///<- 材质总数
    };

    /**
     * @brief CISDI_Mesh 包含网格的所有信息，包括网格头部信息、网格的所有 meshlet 信息、网格的包围盒。
     */
    struct CISDI_Mesh {

        /**
         * @brief CISDI_Mesh::MeshHeader 包含网格的头部信息，包括顶点总数、meshlet 总数、meshlet 三角形总数。
         */
        struct MeshHeader {
            uint32_t vertexCount {0};           ///<- 该mesh的顶点总数
            uint32_t meshletCount {0};          ///<- 该mesh的 meshlet 总数
            uint32_t meshletTriangleCount {0};  ///<- 该mesh的 meshlet 三角形总数
        };

        MeshHeader header {};  ///<- 网格头部信息

        CISDI_Meshlets meshlets {};  ///<- 网格的所有 meshlet 信息

        AABoundingBox boundingBox {};  ///<- 网格的包围盒
    };

    Header header {};  ///<- 模型的头部信息

    Type_STLString name;  ///<- 模型的名称

    Type_STLVector<CISDI_Node> nodes;  ///<- 模型的所有节点

    Type_STLVector<CISDI_Mesh> meshes;  ///<- 模型的所有网格

    Type_STLVector<CISDI_Material> materials;  ///<- 模型的所有材质

    AABoundingBox boundingBox {};  ///<- 模型的包围盒
};

/**
 * @brief 将任意 3D 模型文件转换为 CISDI_3DModel 对象，支持的文件格式包括 FBX、OBJ、GLTF、STL 等。FBX 文件使用 FBXSDK 与 Assimp 结合的方式导入，其余文件使用 Assimp 导入。
 * @param path 输入文件路径
 * @param flipYZ 是否翻转 YZ 轴
 * @param pMemPool pmr memory resource 指针
 * @param output 输出文件路径
 * @return CISDI_3DModel 对象
 */
CISDI_MODEL_DATA_API CISDI_3DModel
Convert(const char* path, bool flipYZ, ::std::pmr::memory_resource* pMemPool,
        const char* output = nullptr);

/**
 * @brief 加载 .cisdi 文件，返回 CISDI_3DModel 对象
 * @param path .cisdi 文件路径
 * @param pMemPool pmr memory resource 指针
 * @return CISDI_3DModel 对象
 */
CISDI_MODEL_DATA_API CISDI_3DModel Load(const char* path,
                                        ::std::pmr::memory_resource* pMemPool);

}  // namespace IntelliDesign_NS::ModelData