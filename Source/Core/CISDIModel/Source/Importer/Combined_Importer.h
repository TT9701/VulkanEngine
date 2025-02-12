/**
 * @file Combined_Importer.h
 * @author 
 * @brief 使用 FBX SDK 和 Assimp 结合导入 FBX 文件
 * @version 0.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

template <class T>
using UniquePtr = ::std::unique_ptr<T>;

template <typename T, typename... Types>
UniquePtr<T> MakeUnique(Types&&... val) {
    return ::std::make_unique<T>(::std::forward<Types>(val)...);
}

namespace IntelliDesign_NS::ModelImporter {

namespace FBXSDK {
class Importer;
}

namespace Assimp {
class Importer;
}

/**
 * @brief 使用 FBX SDK 与 Assimp 结合导入 FBX 文件
 *
 */
class CombinedImporter {
    using Type_InternalMeshDatas =
        ModelData::Type_STLVector<ModelData::InternalMeshData>;
    using Type_Indices =
        ModelData::Type_STLVector<ModelData::Type_STLVector<uint32_t>>;

public:
    /**
     * @brief 构造 Combined Importer
     * @param pMemPool memory resource 指针
     * @param path 源文件路径
     * @param flipYZ 是否翻转 YZ 轴
     * @param outData 导出部分的 CISDI_3DModel 中的信息
     * @param tmpVertices 暂存的顶点数据
     * @param outIndices 输出的索引数据
     */
    CombinedImporter(::std::pmr::memory_resource* pMemPool, const char* path,
                     bool flipYZ, ModelData::CISDI_3DModel& outData,
                     Type_InternalMeshDatas& tmpVertices,
                     Type_Indices& outIndices);

    ~CombinedImporter();

private:
    /**
     * @brief 处理 FBX SDK 导入的 Node 与 Assimp 导入的 Node 的对应关系
     * @param outData 输出的 CISDI_3DModel 对象
     * @param tmpAssimpData 暂存的 Assimp 导入的 CISDI_3DModel 对象
     * @param tmpVertices 暂存的顶点数据
     * @param outIndices 输出的索引数据
     */
    void ProcessNode(ModelData::CISDI_3DModel& outData,
                     ModelData::CISDI_3DModel const& tmpAssimpData,
                     Type_InternalMeshDatas& tmpVertices,
                     Type_Indices& outIndices);

private:
    UniquePtr<FBXSDK::Importer> mFBXImporter;     ///<- FBX SDK Importer
    UniquePtr<Assimp::Importer> mAssimpImporter;  ////<- Assimp Importer
};

}  // namespace IntelliDesign_NS::ModelImporter