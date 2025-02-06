#pragma once

#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

namespace IntelliDesign_NS::ModelImporter {

class CombinedImporter;

namespace Assimp {

class Importer {
    using Type_InternalMeshDatas =
        ModelData::Type_STLVector<ModelData::InternalMeshData>;
    using Type_Indices =
        ModelData::Type_STLVector<ModelData::Type_STLVector<uint32_t>>;

public:
    Importer(::std::pmr::memory_resource* pMemPool, const char* path,
             bool flipYZ, ModelData::CISDI_3DModel& outData,
             Type_InternalMeshDatas& tmpVertices, Type_Indices& outIndices);

    ~Importer();

private:
    friend CombinedImporter;

    void ImportScene(const char* path);

    void InitializeData(ModelData::CISDI_3DModel& outData, const char* path);

    void ExtractMaterials(ModelData::CISDI_3DModel& outData);

    uint32_t ProcessNode(ModelData::CISDI_3DModel& outData,
                         uint32_t parentNodeIdx, aiNode* node, bool flipYZ);

    void ProcessMesh(ModelData::CISDI_Node& cisdiNode, aiMesh* mesh,
                     bool flipYZ);

    void ProcessNodeProperties(aiNode* node, ModelData::CISDI_Node& cisdiNode);

private:
    ::std::pmr::memory_resource* pMemPool;

    ::Assimp::Importer importer {};
    aiScene* mScene {nullptr};

    Type_InternalMeshDatas& mTmpVertices;
    Type_Indices& mOutIndices;
};

}  // namespace Assimp
}  // namespace IntelliDesign_NS::ModelImporter
