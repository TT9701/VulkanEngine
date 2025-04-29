/**
 * @file Assimp_Importer.h
 * @author 
 * @brief 使用 Assimp 导入非 FBX 文件
 * @version 0.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

namespace IntelliDesign_NS::ModelImporter {

class CombinedImporter;

namespace Assimp {

/**
 * @brief 使用 Assimp 导入非 FBX 文件
 *
 */
class Importer {
    INTELLI_DS_DEFINE_STANDARD_CLASS_UNIQUE(Importer);

    using Type_InternalMeshDatas =
        ModelData::Type_STLVector<ModelData::InternalMeshData>;
    using Type_Indices =
        ModelData::Type_STLVector<ModelData::Type_STLVector<uint32_t>>;

public:
    /**
     * @brief 构造 Assimp Importer
     * @param pMemPool memory resource 指针
     * @param path 源文件路径
     * @param flipYZ 是否翻转 YZ 轴
     * @param outData 导出部分的 CISDI_3DModel 中的信息
     * @param tmpVertices 暂存的顶点数据
     * @param outIndices 输出的索引数据
     */
    Importer(::std::pmr::memory_resource* pMemPool, const char* path,
             bool flipYZ, ModelData::CISDI_3DModel& outData,
             Type_InternalMeshDatas& tmpVertices, Type_Indices& outIndices);

    ~Importer();

private:
    friend CombinedImporter;

    /**
     * @brief 使用 aiImporter 导入aiScene
     * @param path 源文件路径
     */
    void ImportScene(const char* path);

    /**
     * @brief 根据 aiScene 初始化 CISDI_3DModel总体数据
     * @param outData 输出的 CISDI_3DModel 对象
     * @param path 源文件路径
     */
    void InitializeData(ModelData::CISDI_3DModel& outData, const char* path);

    /**
     * @brief 根据 aiScene 导出材质信息
     * @param outData 输出的 CISDI_3DModel 对象
     */
    void ExtractMaterials(ModelData::CISDI_3DModel& outData);

    /**
     * @brief 根据 aiScene 处理节点
     * @param outData 输出的 CISDI_3DModel 对象
     * @param parentNodeIdx 父结点索引
     * @param node aiNode 节点
     * @param flipYZ 是否翻转 YZ 轴
     * @return 当前节点索引
     */
    uint32_t ProcessNode(ModelData::CISDI_3DModel& outData,
                         uint32_t parentNodeIdx, aiNode* node, bool flipYZ);

    /**
     * @brief 根据 aiMesh 处理网格信息
     * @param cisdiNode 当前节点所属的 CISDI_Node 对象
     * @param mesh aiMesh 网格
     * @param flipYZ 是否翻转 YZ 轴
     */
    void ProcessMesh(ModelData::CISDI_Node& cisdiNode, aiMesh* mesh,
                     bool flipYZ);

    /**
     * @brief 根据 aiNode 处理节点用户自定义属性
     * @param node aiNode 节点
     * @param cisdiNode 对应的 CISDI_Node 对象
     */
    void ProcessNodeProperties(aiNode* node, ModelData::CISDI_Node& cisdiNode);

private:
    ::std::pmr::memory_resource* pMemPool;  ///<- pmr memory resource 指针

    ::Assimp::Importer importer {};  ///<- Assimp library importer
    aiScene* mScene {nullptr};       ///<- aiScene 对象

    Type_InternalMeshDatas& mTmpVertices;  ///<- 暂存的顶点数据
    Type_Indices& mOutIndices;             ///<- 输出的索引数据
};

}  // namespace Assimp
}  // namespace IntelliDesign_NS::ModelImporter
