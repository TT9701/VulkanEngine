#pragma once

#include <fbxsdk.h>

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

namespace IntelliDesign_NS::ModelImporter {

class CombinedImporter;

namespace FBXSDK {

class Importer {
    using Type_InternalMeshDatas =
        ModelData::Type_STLVector<ModelData::InternalMeshData>;
    using Type_Indices =
        ModelData::Type_STLVector<ModelData::Type_STLVector<uint32_t>>;
    using Type_Materials = ModelData::Type_STLVector<ModelData::CISDI_Material>;

public:
    Importer(::std::pmr::memory_resource* pMemPool, const char* path,
             bool flipYZ, ModelData::CISDI_3DModel& outData,
             Type_InternalMeshDatas& tmpVertices, Type_Indices& outIndices,
             bool meshData = true);

    ~Importer();

private:
    friend CombinedImporter;

    void InitializeSdkObjects();

    void ImportScene(const char* path);

    void InitializeData(ModelData::CISDI_3DModel& data, const char* path);

    /**
     * @brief Change axis system, system scale factor. Triangulate.
     */
    void ModifyGeometry();

    void ExtractMaterials(ModelData::CISDI_3DModel& data);

    int ProcessNode(ModelData::CISDI_3DModel& data, FbxNode* pNode,
                    int parentNodeIdx, bool flipYZ, bool meshData);

    int ProcessMesh(FbxMesh* pMesh, bool flipYZ, bool meshData);

    void ProcessUserDefinedProperties(FbxNode const* pNode,
                                      ModelData::CISDI_Node& cisdiNode);

private:
    ::std::pmr::memory_resource* pMemPool;

    FbxManager* mSdkManager {nullptr};
    FbxScene* mScene {nullptr};

    Type_InternalMeshDatas& mTmpVertices;
    Type_Indices& mOutIndices;
};

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
