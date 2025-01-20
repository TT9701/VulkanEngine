#include "FBX_Importer.h"

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

#include <fbxsdk.h>

#include <cassert>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

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

int ProcessMesh(FbxMesh* pMesh, bool flipYZ,
                Type_STLVector<InternalMeshData>& tmpVertices) {
    FbxVector4* lControlPoints = pMesh->GetControlPoints();

    int triangleCount = pMesh->GetPolygonCount();
    int vertCount = triangleCount * 3;

    int meshIdx = (int)tmpVertices.size();
    auto& tmpMeshVertices = tmpVertices.emplace_back();
    tmpMeshVertices.positions.resize(vertCount);
    tmpMeshVertices.normals.resize(vertCount);
    tmpMeshVertices.uvs.resize(vertCount);

    for (int i = 0; i < triangleCount; ++i) {
        int polySize = pMesh->GetPolygonSize(i);
        assert(polySize == 3);

        for (int j = 0; j < 3; ++j) {
            int vertIdx = i * 3 + j;
            // positions
            int lControlPointIndex = pMesh->GetPolygonVertex(i, j);
            FbxVector4 lControlPoint = lControlPoints[lControlPointIndex];
            Float32_3 fPos {(float)lControlPoint[0], (float)lControlPoint[1],
                            (float)lControlPoint[2]};

            if (flipYZ)
                ::std::swap(fPos[1], fPos[2]);

            tmpMeshVertices.positions[vertIdx] = fPos;

            // normals
            if (pMesh->GetElementNormalCount() < 1)
                continue;

            auto leNormal = pMesh->GetElementNormal();
            auto& norm = tmpMeshVertices.normals[vertIdx];

            auto setNorm = [&norm, flipYZ](FbxVector4 const& normal) {
                norm = Float32_3 {(float)normal[0], (float)normal[1],
                                  (float)normal[2]};
                if (flipYZ)
                    ::std::swap(norm[1], norm[2]);
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

            if (pMesh->GetElementUVCount() < 1)
                continue;
            auto leUV = pMesh->GetElementUV();

            auto& uv = tmpMeshVertices.uvs[vertIdx];

            auto setUV = [&uv](FbxVector2 const& texcoords) {
                uv = Float32_2 {(float)texcoords[0], (float)texcoords[1]};
            };

            switch (leUV->GetMappingMode()) {
                default: break;
                case FbxGeometryElement::eByControlPoint:
                    switch (leUV->GetReferenceMode()) {
                        case FbxGeometryElement::eDirect: {
                            auto texcoords = leUV->GetDirectArray().GetAt(
                                lControlPointIndex);
                            setUV(texcoords);
                        } break;
                        case FbxGeometryElement::eIndexToDirect: {
                            int id =
                                leUV->GetIndexArray().GetAt(lControlPointIndex);
                            auto texcoords = leUV->GetDirectArray().GetAt(id);
                            setUV(texcoords);
                        } break;
                        default: break;
                    }
                    break;

                case FbxGeometryElement::eByPolygonVertex: {
                    int lTextureUVIndex = pMesh->GetTextureUVIndex(i, j);
                    switch (leUV->GetReferenceMode()) {
                        case FbxGeometryElement::eDirect:
                        case FbxGeometryElement::eIndexToDirect: {
                            auto texcoords =
                                leUV->GetDirectArray().GetAt(lTextureUVIndex);
                            setUV(texcoords);
                        } break;
                        default: break;
                    }
                } break;
            }
        }
    }

    return meshIdx;
}

::std::map<FbxSurfaceMaterial*, uint32_t> materialIdxMap {};

void ProcessUserDefinedProperties(FbxNode* pNode,
                                  CISDI_3DModel::Node& cisdiNode) {
    if (!pNode)
        return;

    FbxProperty prop = pNode->GetFirstProperty();
    while (prop.IsValid()) {
        if (prop.GetFlag(FbxPropertyFlags::eUserDefined)
            || prop.GetFlag(FbxPropertyFlags::eImported)) {
            std::string propName = prop.GetNameAsCStr();
            FbxDataType dataType = prop.GetPropertyDataType();

            switch (dataType.GetType()) {
                case eFbxBool:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxBool>());
                    break;
                case eFbxChar:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxChar>());
                    break;
                case eFbxUChar:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxUChar>());
                    break;
                case eFbxInt:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxInt>());
                    break;
                case eFbxUInt:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxUInt>());
                    break;
                case eFbxLongLong:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxLongLong>());
                    break;
                case eFbxULongLong:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxULongLong>());
                    break;
                case eFbxFloat:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxFloat>());
                    break;
                case eFbxDouble:
                    cisdiNode.userProperties.emplace(propName,
                                                     prop.Get<FbxDouble>());
                    break;
                case eFbxString:
                    cisdiNode.userProperties.emplace(
                        propName, prop.Get<FbxString>().Buffer());
                    break;
                default:
                    throw ::std::runtime_error(
                        "ProcessUserDefinedProperties: Unknown property type");
            }
            cisdiNode.userPropertyCount++;
        }
        prop = pNode->GetNextProperty(prop);
    }
}

int ProcessNode(FbxNode* pNode, int parentNodeIdx, CISDI_3DModel& data,
                bool flipYZ, Type_STLVector<InternalMeshData>& tmpVertices) {
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
                    cisdiNode.meshIdx = ProcessMesh(mesh, flipYZ, tmpVertices);
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

    // process user properties
    ProcessUserDefinedProperties(pNode, cisdiNode);

    auto& ref = data.nodes.emplace_back(::std::move(cisdiNode));

    for (int i = 0; i < childCount; i++) {
        ref.childrenIdx.emplace_back(ProcessNode(pNode->GetChild(i), nodeIdx,
                                                 data, flipYZ, tmpVertices));
    }

    return nodeIdx;
}

}  // namespace

CISDI_3DModel Convert(const char* path, bool flipYZ,
                      Type_STLVector<InternalMeshData>& tmpVertices,
                      Type_STLVector<Type_STLVector<uint32_t>>& outIndices) {
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

    data.header = {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION,
                   (uint32_t)lScene->GetNodeCount(),
                   (uint32_t)lScene->GetGeometryCount(),
                   (uint32_t)lScene->GetMaterialCount()};

    data.name = ::std::filesystem::path(path).stem().string();

    data.nodes.reserve(data.header.nodeCount);

    data.meshes.resize(data.header.meshCount);
    tmpVertices.reserve(data.header.meshCount);

    data.materials.reserve(data.header.materialCount);

    for (int i = 0; i < lScene->GetMaterialCount(); ++i) {
        FbxSurfaceMaterial* lMaterial = lScene->GetMaterial(i);
        materialIdxMap.emplace(lMaterial, i);

        Material cisdiMaterial {};
        cisdiMaterial.name = lMaterial->GetName();

        if (lMaterial->GetClassId().Is(FbxSurfacePhong::ClassId)) {
            cisdiMaterial.data.shadingModel = Material::ShadingModel::Phong;

            auto lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Ambient;
            cisdiMaterial.data.ambient = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->AmbientFactor};

            lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Diffuse;
            cisdiMaterial.data.diffuse = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->DiffuseFactor};

            // TODO: how to deal with specular?
            lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Specular;
            cisdiMaterial.data.specular = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->SpecularFactor};

            lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Emissive;
            cisdiMaterial.data.emissive = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->EmissiveFactor};

            lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->TransparentColor;
            cisdiMaterial.data.transparency = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->TransparencyFactor};

            // TODO: how to deal with Shininess?
            float shininess = ((FbxSurfacePhong*)lMaterial)->Shininess;
            cisdiMaterial.data.shininess = shininess;

            // TODO: how to deal with Reflectivity?
            lKFbxDouble3 = ((FbxSurfacePhong*)lMaterial)->Reflection;
            cisdiMaterial.data.reflection = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->ReflectionFactor};

        } else if (lMaterial->GetClassId().Is(FbxSurfaceLambert::ClassId)) {
            cisdiMaterial.data.shadingModel =
                Material::ShadingModel::Lambert;

            auto lKFbxDouble3 = ((FbxSurfaceLambert*)lMaterial)->Ambient;
            cisdiMaterial.data.ambient = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->AmbientFactor};

            lKFbxDouble3 = ((FbxSurfaceLambert*)lMaterial)->Diffuse;
            cisdiMaterial.data.diffuse = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->DiffuseFactor};

            lKFbxDouble3 = ((FbxSurfaceLambert*)lMaterial)->Emissive;
            cisdiMaterial.data.emissive = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfacePhong*)lMaterial)->EmissiveFactor};

            lKFbxDouble3 = ((FbxSurfaceLambert*)lMaterial)->TransparentColor;
            cisdiMaterial.data.transparency = Float32_4 {
                (float)lKFbxDouble3.Get()[0], (float)lKFbxDouble3.Get()[1],
                (float)lKFbxDouble3.Get()[2],
                (float)((FbxSurfaceLambert*)lMaterial)->TransparencyFactor};
        }
        data.materials.emplace_back(::std::move(cisdiMaterial));
    }

    FbxNode* lRootNode = lScene->GetRootNode();
    if (lRootNode) {
        ProcessNode(lRootNode, -1, data, flipYZ, tmpVertices);
    }

    // Generate model bounding box
    for (auto const& mesh : data.meshes) {
        UpdateAABB(data.boundingBox, mesh.boundingBox);
    }

    // TODO: process keyframes

    DestroySdkObjects(lSdkManager);

    return data;
}

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
