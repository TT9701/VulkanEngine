/**
 * @file FBX_Importer.h
 * @author 
 * @brief 使用 FBX SDK 导入 FBX 文件
 * @version 0.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <fbxsdk.h>

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

namespace IntelliDesign_NS::ModelImporter {

class CombinedImporter;

namespace FBXSDK {

/**
 * @brief 使用 FBX SDK 导入 FBX 文件
 *
 */
class Importer {
    using Type_InternalMeshDatas =
        ModelData::Type_STLVector<ModelData::InternalMeshData>;
    using Type_Indices =
        ModelData::Type_STLVector<ModelData::Type_STLVector<uint32_t>>;
    using Type_Materials = ModelData::Type_STLVector<ModelData::CISDI_Material>;

public:
    /**
     * @brief 构造 FBX SDK Importer
     * @param pMemPool memory resource 指针
     * @param path 源文件路径
     * @param flipYZ 是否翻转 YZ 轴
     * @param outData 导出部分的 CISDI_3DModel 中的信息
     * @param tmpVertices 暂存的顶点数据
     * @param outIndices 输出的索引数据
     * @param meshData 是否导入网格数据
     */
    Importer(::std::pmr::memory_resource* pMemPool, const char* path,
             bool flipYZ, ModelData::CISDI_3DModel& outData,
             Type_InternalMeshDatas& tmpVertices, Type_Indices& outIndices,
             bool meshData = true);

    ~Importer();

private:
    friend CombinedImporter;

    /**
     * @brief 初始化 FBX SDK manager 和 scene
     */
    void InitializeSdkObjects();

    /**
     * @brief 导入 FBX 文件
     * @param path 源文件路径
     */
    void ImportScene(const char* path);

    /**
     * @brief 初始化 CISDI_3DModel 总体数据
     * @param data CISDI_3DModel 对象
     * @param path 源文件路径
     */
    void InitializeData(ModelData::CISDI_3DModel& data, const char* path);

    /**
     * @brief 修改模型坐标系统、坐标系缩放因子、以及三角化
     */
    void ModifyGeometry();

    /**
     * @brief 导出材质信息
     * @param data CISDI_3DModel 对象
     */
    void ExtractMaterials(ModelData::CISDI_3DModel& data);

    /**
     * @brief 根据 FbxNode 处理节点信息
     * @param data CISDI_3DModel 对象
     * @param pNode FbxNode 节点
     * @param parentNodeIdx 父结点索引
     * @param flipYZ 是否翻转 YZ 轴
     * @param meshData 是否导入网格数据
     * @return 当前节点索引
     */
    int ProcessNode(ModelData::CISDI_3DModel& data, FbxNode* pNode,
                    int parentNodeIdx, bool flipYZ, bool meshData);

    /**
     * @brief 根据 FbxMesh 处理网格信息
     * @param pMesh FbxMesh 网格
     * @param flipYZ 是否翻转 YZ 轴
     * @param meshData 是否导入网格数据
     * @return 当前网格索引
     */
    int ProcessMesh(FbxMesh* pMesh, bool flipYZ, bool meshData);

    /**
     * @brief 根据 FbxNode 处理节点用户自定义属性
     * @param pNode FbxNode 节点
     * @param cisdiNode 对应的 CISDI_Node 对象
     */
    void ProcessUserDefinedProperties(FbxNode const* pNode,
                                      ModelData::CISDI_Node& cisdiNode);

private:
    ::std::pmr::memory_resource* pMemPool;  ///<- pmr memory resource 指针

    FbxManager* mSdkManager {nullptr};  ///<- FBX SDK manager
    FbxScene* mScene {nullptr};         ///<- FBX scene

    Type_InternalMeshDatas& mTmpVertices;  ///<- 暂存的顶点数据
    Type_Indices& mOutIndices;             ///<- 输出的索引数据
};

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
