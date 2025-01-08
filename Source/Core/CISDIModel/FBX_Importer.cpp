#include "FBX_Importer.h"

#include "CISDI_3DModelData.h"
#include "Common.h"

#include <fbxsdk.h>

#include <Windows.h>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <map>
#include <unordered_set>

using namespace IntelliDesign_NS::ModelData;

namespace IntelliDesign_NS::ModelImporter {
namespace FBXSDK {

namespace {

std::string GbkToUtf8(const std::string& gbkStr) {
    typedef std::codecvt_byname<wchar_t, char, std::mbstate_t> F;
#ifdef _WIN32
    std::wstring_convert<F> gbtowide(new F("zh_CN"));
#else
    std::wstring_convert<F> gbtowide(new F("zh_CN.GB18030"));
#endif
    std::wstring wstr = gbtowide.from_bytes(gbkStr);
    std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8_convert;
    std::string u8str = utf8_convert.to_bytes(wstr);
    return u8str;
}

void InitializeSdkObjects(FbxManager*& pManager, FbxScene*& pScene) {
    pManager = FbxManager::Create();
    if (!pManager) {
        FBXSDK_printf("Error: Unable to create FBX Manager!\n");
        exit(1);
    } else {
        FBXSDK_printf("Using Autodesk FBX SDK version %s\n",
                      pManager->GetVersion());
    }

    FbxIOSettings* ios = FbxIOSettings::Create(pManager, IOSROOT);
    pManager->SetIOSettings(ios);

    FbxString lPath = FbxGetApplicationDirectory();
    pManager->LoadPluginsDirectory(lPath.Buffer());

    pScene = FbxScene::Create(pManager, "");
    if (!pScene) {
        FBXSDK_printf("Error: Unable to create FBX scene!\n");
        exit(1);
    }
}

void DestroySdkObjects(FbxManager* pManager) {
    if (pManager)
        pManager->Destroy();
}

int ProcessMesh(FbxMesh* pMesh, CISDI_3DModel& data, bool flipYZ) {
    int meshIdx = (int)data.meshes.size();
    CISDI_3DModel::Mesh cisdiMesh {};

    FbxVector4* lControlPoints = pMesh->GetControlPoints();

    int triangleCount = pMesh->GetPolygonCount();
    int vertCount = triangleCount * 3;

    cisdiMesh.vertices.positions.resize(vertCount);
    cisdiMesh.vertices.normals.resize(vertCount);
    cisdiMesh.vertices.uvs.resize(vertCount);

    for (int i = 0; i < triangleCount; ++i) {
        int polySize = pMesh->GetPolygonSize(i);
        assert(polySize == 3);

        for (int j = 0; j < 3; ++j) {
            int vertIdx = i * 3 + j;
            // positions
            int lControlPointIndex = pMesh->GetPolygonVertex(i, j);
            FbxVector4 lControlPoint = lControlPoints[lControlPointIndex];
            Float4 pos {(float)lControlPoint[0], (float)lControlPoint[1],
                        (float)lControlPoint[2], 1.0f};

            if (flipYZ)
                ::std::swap(pos[1], pos[2]);

            cisdiMesh.vertices.positions[vertIdx] = pos;

            // normals
            if (pMesh->GetElementNormalCount() < 1)
                continue;

            auto leNormal = pMesh->GetElementNormal();
            auto& norm = cisdiMesh.vertices.normals[vertIdx];

            auto setNorm = [&norm](FbxVector4 const& normal) {
                auto f = ::std::sqrt(2.0f / (1.0f - (float)normal[0]));
                norm[0] = (float)normal[1] * f;
                norm[1] = (float)normal[2] * f;
            };

            switch (leNormal->GetMappingMode()) {
                case FbxGeometryElement::eByControlPoint: {
                    switch (leNormal->GetReferenceMode()) {
                        case FbxGeometryElement::eDirect: {
                            auto normal = leNormal->GetDirectArray().GetAt(
                                lControlPointIndex);
                            setNorm(normal);
                        } break;

                        case FbxGeometryElement::eIndexToDirect: {
                            int id = leNormal->GetIndexArray().GetAt(
                                lControlPointIndex);
                            auto normal = leNormal->GetDirectArray().GetAt(id);
                            setNorm(normal);
                        } break;

                        default: break;
                    }
                } break;

                case FbxGeometryElement::eByPolygonVertex: {
                    switch (leNormal->GetReferenceMode()) {
                        case FbxGeometryElement::eDirect: {
                            auto normal =
                                leNormal->GetDirectArray().GetAt(vertIdx);
                            setNorm(normal);
                        } break;

                        case FbxGeometryElement::eIndexToDirect: {
                            int id = leNormal->GetIndexArray().GetAt(vertIdx);
                            auto normal = leNormal->GetDirectArray().GetAt(id);
                            setNorm(normal);
                        } break;

                        default: break;
                    }
                } break;
            }

            if (flipYZ)
                ::std::swap(norm[0], norm[1]);
        }
    }

    cisdiMesh.header.vertexCount = vertCount;
    cisdiMesh.header.indexCount = vertCount;

    data.meshes.emplace_back(cisdiMesh);

    return meshIdx;
}

::std::map<FbxSurfaceMaterial*, uint32_t> materialIdxMap {};

int ProcessNode(FbxNode* pNode, int parentNodeIdx, CISDI_3DModel& data,
                bool flipYZ) {
    int nodeIdx = (int)data.nodes.size();
    int childCount = pNode->GetChildCount();

    CISDI_3DModel::Node cisdiNode {};
    cisdiNode.name = pNode->GetName();
    cisdiNode.parentIdx = parentNodeIdx;
    cisdiNode.childCount = childCount;
    cisdiNode.childrenIdx.reserve(childCount);

    if (FbxNodeAttribute* lNodeAttribute = pNode->GetNodeAttribute()) {
        switch (lNodeAttribute->GetAttributeType()) {
            case FbxNodeAttribute::eMesh:
                if (FbxMesh* mesh = pNode->GetMesh())
                    cisdiNode.meshIdx = ProcessMesh(mesh, data, flipYZ);
                break;
            case FbxNodeAttribute::eSkeleton: break;
            case FbxNodeAttribute::eLight: break;
            case FbxNodeAttribute::eCamera: break;
            default: break;
        }
    }

    // process material
    if (pNode->GetMaterialCount() > 0) {
        auto lMaterial = pNode->GetMaterial(0);
        cisdiNode.materialIdx = materialIdxMap.at(lMaterial);
    }

    auto& ref = data.nodes.emplace_back(::std::move(cisdiNode));

    for (int i = 0; i < childCount; i++) {
        ref.childrenIdx.emplace_back(
            ProcessNode(pNode->GetChild(i), nodeIdx, data, flipYZ));
    }

    return nodeIdx;
}

}  // namespace

CISDI_3DModel Convert(const char* path, bool flipYZ) {
    CISDI_3DModel data {};

    FbxManager* lSdkManager;
    FbxScene* lScene;
    InitializeSdkObjects(lSdkManager, lScene);

    FbxImporter* lImporter = FbxImporter::Create(lSdkManager, "");

    if (!lImporter->Initialize(path, -1, lSdkManager->GetIOSettings())) {
        printf("Call to FbxImporter::Initialize() failed.\n");
        printf("Error returned: %s\n\n",
               lImporter->GetStatus().GetErrorString());
        exit(-1);
    }

    if (lImporter->Import(lScene)) {
        // Check the scene integrity!
        FbxStatus status;
        FbxArray<FbxString*> details;
        FbxSceneCheckUtility sceneCheck(FbxCast<FbxScene>(lScene), &status,
                                        &details);
        bool lNotify =
            (!sceneCheck.Validate(FbxSceneCheckUtility::eCkeckData)
             && details.GetCount() > 0)
            || (lImporter->GetStatus().GetCode() != FbxStatus::eSuccess);
        if (lNotify) {
            FBXSDK_printf("\n");
            FBXSDK_printf(
                "**************************************************************"
                "******************\n");
            if (details.GetCount()) {
                FBXSDK_printf(
                    "Scene integrity verification failed with the following "
                    "errors:\n");

                for (int i = 0; i < details.GetCount(); i++)
                    FBXSDK_printf("   %s\n", details[i]->Buffer());

                FbxArrayDelete<FbxString*>(details);
            }

            if (lImporter->GetStatus().GetCode() != FbxStatus::eSuccess) {
                FBXSDK_printf("\n");
                FBXSDK_printf("WARNING:\n");
                FBXSDK_printf(
                    "   The importer was able to read the file but with "
                    "errors.\n");
                FBXSDK_printf("   Loaded scene may be incomplete.\n\n");
                FBXSDK_printf("   Last error message:'%s'\n",
                              lImporter->GetStatus().GetErrorString());
            }

            FBXSDK_printf(
                "**************************************************************"
                "******************\n");
            FBXSDK_printf("\n");
        }
    }

    lImporter->Destroy();

    FbxAxisSystem SceneAxisSystem = lScene->GetGlobalSettings().GetAxisSystem();
    FbxAxisSystem OurAxisSystem(FbxAxisSystem::eYAxis,
                                FbxAxisSystem::eParityOdd,
                                FbxAxisSystem::eRightHanded);
    if (SceneAxisSystem != OurAxisSystem) {
        OurAxisSystem.ConvertScene(lScene);
    }

    FbxSystemUnit SceneSystemUnit = lScene->GetGlobalSettings().GetSystemUnit();
    if (SceneSystemUnit.GetScaleFactor() != 1.0) {
        FbxSystemUnit::cm.ConvertScene(lScene);
    }

    // Convert mesh, NURBS and patch into triangle mesh
    FbxGeometryConverter lGeomConverter(lSdkManager);
    try {
        lGeomConverter.Triangulate(lScene, true);
    } catch (std::runtime_error&) {
        FBXSDK_printf("Scene integrity verification failed.\n");
    }

    data.header = {CISDI_3DModel_HEADER_UINT64,
                   CISDI_3DModel_VERSION,
                   (uint32_t)lScene->GetNodeCount(),
                   (uint32_t)lScene->GetGeometryCount(),
                   (uint32_t)lScene->GetMaterialCount(),
                   false};

    data.name = ::std::filesystem::path(path).stem().string();

    data.nodes.reserve(data.header.nodeCount);

    data.meshes.reserve(data.header.meshCount);

    data.materials.reserve(data.header.materialCount);

    for (int i = 0; i < lScene->GetMaterialCount(); ++i) {
        FbxSurfaceMaterial* lMaterial = lScene->GetMaterial(i);
        materialIdxMap.emplace(lMaterial, i);

        CISDI_3DModel::Material cisdiMaterial {};
        cisdiMaterial.name = lMaterial->GetName();

        if (lMaterial->GetClassId().Is(FbxSurfacePhong::ClassId)) {

            auto lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Ambient;
            Float3 ambient {(float)lKFbxDouble3.Get()[0],
                            (float)lKFbxDouble3.Get()[1],
                            (float)lKFbxDouble3.Get()[2]};
            cisdiMaterial.ambient = ambient;

            lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Diffuse;
            Float3 diffuse {(float)lKFbxDouble3.Get()[0],
                            (float)lKFbxDouble3.Get()[1],
                            (float)lKFbxDouble3.Get()[2]};
            cisdiMaterial.diffuse = diffuse;

            // TODO: how to deal with specular?
            // lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Specular;
            // Float3 specular {(float)lKFbxDouble3.Get()[0],
            //                  (float)lKFbxDouble3.Get()[1],
            //                  (float)lKFbxDouble3.Get()[2]};

            lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Emissive;
            Float3 emissive {(float)lKFbxDouble3.Get()[0],
                             (float)lKFbxDouble3.Get()[1],
                             (float)lKFbxDouble3.Get()[2]};
            cisdiMaterial.emissive = emissive;

            float opacity = ((FbxSurfacePhong*)lMaterial)->TransparencyFactor;
            cisdiMaterial.opacity = opacity;

            // TODO: how to deal with Shininess?
            // float shininess = ((FbxSurfacePhong*)lMaterial)->Shininess;

            // TODO: how to deal with Reflectivity?
            // float reflectivity =
            //     ((FbxSurfacePhong*)lMaterial)->ReflectionFactor;

        } else if (lMaterial->GetClassId().Is(FbxSurfaceLambert::ClassId)) {
            auto lKFbxDouble3 = ((FbxSurfaceLambert*)lMaterial)->Ambient;
            Float3 ambient {(float)lKFbxDouble3.Get()[0],
                            (float)lKFbxDouble3.Get()[1],
                            (float)lKFbxDouble3.Get()[2]};
            cisdiMaterial.ambient = ambient;

            lKFbxDouble3 = ((FbxSurfaceLambert*)lMaterial)->Diffuse;
            Float3 diffuse {(float)lKFbxDouble3.Get()[0],
                            (float)lKFbxDouble3.Get()[1],
                            (float)lKFbxDouble3.Get()[2]};
            cisdiMaterial.diffuse = diffuse;

            lKFbxDouble3 = ((FbxSurfaceLambert*)lMaterial)->Emissive;
            Float3 emissive {(float)lKFbxDouble3.Get()[0],
                             (float)lKFbxDouble3.Get()[1],
                             (float)lKFbxDouble3.Get()[2]};
            cisdiMaterial.emissive = emissive;

            float opacity = ((FbxSurfaceLambert*)lMaterial)->TransparencyFactor;
            cisdiMaterial.opacity = opacity;
        }
        data.materials.emplace_back(::std::move(cisdiMaterial));
    }

    FbxNode* lRootNode = lScene->GetRootNode();
    if (lRootNode) {
        int childCount = lRootNode->GetChildCount();
        CISDI_3DModel::Node cisdiNode {};
        cisdiNode.name = lRootNode->GetName();
        cisdiNode.childCount = childCount;
        cisdiNode.childrenIdx.reserve(childCount);

        int rootNodeIdx = (int)data.nodes.size();
        for (int i = 0; i < childCount; i++) {
            cisdiNode.childrenIdx.emplace_back(
                ProcessNode(lRootNode->GetChild(i), rootNodeIdx, data, flipYZ));
        }

        data.nodes.emplace_back(::std::move(cisdiNode));
    }

    // TODO: process keyframes

    DestroySdkObjects(lSdkManager);

    return data;
}

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
