#include "FBX_Importer.h"

#include "CISDI_3DModelData.h"
#include "Common.h"

#include <fbxsdk.h>

#include <Windows.h>
#include <codecvt>
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

void DestroySdkObjects(FbxManager* pManager, bool pExitStatus) {
    if (pManager)
        pManager->Destroy();
    if (pExitStatus)
        FBXSDK_printf("Program Success!\n");
}

void ProcessMesh(FbxMesh* pMesh, CISDI_3DModel& data) {
    printf("    Processing Mesh.\n");

    CISDI_3DModel::Mesh cisdiMesh {};

    int controlPointsCount = pMesh->GetControlPointsCount();
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
            auto& pos = cisdiMesh.vertices.positions[vertIdx];
            pos[0] = (float)lControlPoint[0];
            pos[1] = (float)lControlPoint[1];
            pos[2] = (float)lControlPoint[2];
            pos[3] = 1.0f;

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
        }
    }

    // int numIndices = pMesh->GetPolygonVertexCount();
    // int* lIndices = pMesh->GetPolygonVertices();

    cisdiMesh.indices.resize(vertCount);
    // cisdiMesh.indices.assign(lIndices, lIndices + numIndices);
    for (size_t i = 0; i < vertCount; ++i) {
        cisdiMesh.indices[i] = i;
    }

    cisdiMesh.header.vertexCount = vertCount;
    cisdiMesh.header.indexCount = vertCount;

    data.meshes.emplace_back(cisdiMesh);
}

void ProcessNode(FbxNode* pNode, CISDI_3DModel& data) {
    auto name = pNode->GetName();
    printf("Processing Node: %s\n", name);

    // FbxDouble3 translation = pNode->LclTranslation.Get();
    // FbxDouble3 rotation = pNode->LclRotation.Get();
    // FbxDouble3 scaling = pNode->LclScaling.Get();
    //
    // printf(
    //     "    Node transforms: translation: (%f, %f, %f), rotation: (%f, %f, "
    //     "%f), scaling: (%f, %f, %f).\n",
    //     translation.mData[0], translation.mData[1], translation.mData[2],
    //     rotation.mData[0], rotation.mData[1], rotation.mData[2],
    //     scaling.mData[0], scaling.mData[1], scaling.mData[2]);

    if (FbxNodeAttribute* lNodeAttribute = pNode->GetNodeAttribute()) {
        switch (lNodeAttribute->GetAttributeType()) {
            case FbxNodeAttribute::eMesh:
                if (FbxMesh* mesh = pNode->GetMesh())
                    ProcessMesh(mesh, data);
                break;
            case FbxNodeAttribute::eSkeleton: break;
            case FbxNodeAttribute::eLight: break;
            case FbxNodeAttribute::eCamera: break;
            default: break;
        }
    }
    for (int i = 0; i < pNode->GetChildCount(); i++) {
        ProcessNode(pNode->GetChild(i), data);
    }
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

    uint32_t geoCount = lScene->GetGeometryCount();

    data.header = {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION, geoCount,
                   false};

    data.meshes.reserve(data.header.meshCount);

    FbxNode* lRootNode = lScene->GetRootNode();
    if (lRootNode) {
        for (int i = 0; i < lRootNode->GetChildCount(); i++) {
            ProcessNode(lRootNode->GetChild(i), data);
        }
    }

    // process materials

    // TODO: process keyframes

    DestroySdkObjects(lSdkManager, true);

    return data;
}

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
