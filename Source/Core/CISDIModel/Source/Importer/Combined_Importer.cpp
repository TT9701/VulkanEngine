#include "Combined_Importer.h"

#include "Assimp_Importer.h"
#include "FBX_Importer.h"

namespace IntelliDesign_NS::ModelImporter {

CombinedImporter::CombinedImporter(std::pmr::memory_resource* pMemPool,
                                   const char* path, bool flipYZ,
                                   ModelData::CISDI_3DModel& outData,
                                   Type_InternalMeshDatas& tmpVertices,
                                   Type_Indices& outIndices) {
    ModelData::CISDI_3DModel tmpAssimpData {pMemPool};

    mFBXImporter = MakeUnique<FBXSDK::Importer>(pMemPool, path, flipYZ, outData,
                                                tmpVertices, outIndices, false);
    tmpVertices.clear();
    mAssimpImporter = MakeUnique<Assimp::Importer>(
        pMemPool, path, flipYZ, tmpAssimpData, tmpVertices, outIndices);

    outData.header.meshCount = (uint32_t)tmpVertices.size();
    outData.meshes.resize(outData.header.meshCount);

    ProcessNode(outData, tmpAssimpData, tmpVertices, outIndices);
}

CombinedImporter::~CombinedImporter() = default;

void CombinedImporter::ProcessNode(
    ModelData::CISDI_3DModel& outData,
    ModelData::CISDI_3DModel const& tmpAssimpData,
    Type_InternalMeshDatas& tmpVertices, Type_Indices& outIndices) {
    ::std::unordered_map<ModelData::Type_STLString, int> meshIDtoIdxMap;
    ::std::unordered_map<int, ModelData::Type_STLString> meshIDxtoIDMap;

    const char* cisdiMeshIDName = "CISDI_Mesh_ID";
    for (auto const& node : tmpAssimpData.nodes) {
        if (node.meshIdx != -1) {
            if (!node.userProperties.contains(cisdiMeshIDName)) {
                return;
            }
            ::std::visit(
                [&](auto&& v) {
                    using T = ::std::decay_t<decltype(v)>;
                    if constexpr (::std::is_same_v<T,
                                                   ModelData::Type_STLString>) {
                        meshIDtoIdxMap[v] = node.meshIdx;
                        meshIDxtoIDMap[node.meshIdx] = v;
                    } else {
                        throw ::std::runtime_error(
                            "ERROR::COMBINED_IMPORTER::PROCESSNODE: "
                            "CISDI_Mesh_ID should be Type_STLString!");
                    }
                },
                node.userProperties.at(cisdiMeshIDName));
        }
    }

    for (auto const& node : outData.nodes) {

        if (node.meshIdx != -1) {
            ::std::visit(
                [&](auto&& v) {
                    using T = ::std::decay_t<decltype(v)>;
                    if constexpr (::std::is_same_v<T,
                                                   ModelData::Type_STLString>) {
                        auto assimpID = meshIDtoIdxMap.at(v);
                        if (assimpID != node.meshIdx) {
                            ::std::swap(tmpVertices[assimpID],
                                        tmpVertices[node.meshIdx]);
                            ::std::swap(outIndices[assimpID],
                                        outIndices[node.meshIdx]);

                            auto prevMeshID = meshIDxtoIDMap.at(node.meshIdx);
                            meshIDtoIdxMap[prevMeshID] = assimpID;
                        }
                    }
                },
                node.userProperties.at(cisdiMeshIDName));
        }
    }
}

}  // namespace IntelliDesign_NS::ModelImporter