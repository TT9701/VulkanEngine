#include "FBX_Importer.h"

#include <cassert>

#include <codecvt>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include "CISDI_3DModelData.h"
#include "Source/Common/Common.h"

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

::std::map<FbxSurfaceMaterial*, uint32_t> materialIdxMap {};

}  // namespace

Importer::Importer(std::pmr::memory_resource* pMemPool, const char* path,
                   bool flipYZ, CISDI_3DModel& outData,
                   Type_InternalMeshDatas& tmpVertices,
                   Type_Indices& outIndices, bool meshData)
    : pMemPool(pMemPool), mTmpVertices(tmpVertices), mOutIndices(outIndices) {
    InitializeSdkObjects();

    ImportScene(path);
    if (meshData)
        ModifyGeometry();
    InitializeData(outData, path);
    ExtractMaterials(outData);
    ProcessNode(outData, mScene->GetRootNode(), -1, flipYZ, meshData);

    outData.header.meshCount = (uint32_t)mTmpVertices.size();
    outData.meshes.resize(outData.header.meshCount);
}

Importer::~Importer() {
    if (mSdkManager)
        mSdkManager->Destroy();
}

void Importer::InitializeSdkObjects() {
    mSdkManager = FbxManager::Create();
    if (!mSdkManager) {
        FBXSDK_printf("Error: Unable to create FBX Manager!\n");
        exit(1);
    } else {
        FBXSDK_printf("Using Autodesk FBX SDK version %s\n",
                      mSdkManager->GetVersion());
    }

    FbxIOSettings* ios = FbxIOSettings::Create(mSdkManager, IOSROOT);
    mSdkManager->SetIOSettings(ios);

    mScene = ::FbxScene::Create(mSdkManager, "");
    if (!mScene) {
        FBXSDK_printf("Error: Unable to create FBX scene!\n");
        exit(1);
    }
}

void Importer::ImportScene(const char* path) {
    FbxImporter* lImporter = FbxImporter::Create(mSdkManager, "");

    if (!lImporter->Initialize(path, -1, mSdkManager->GetIOSettings())) {
        printf("Call to FbxImporter::Initialize() failed.\n");
        printf("Error returned: %s\n\n",
               lImporter->GetStatus().GetErrorString());
        exit(-1);
    }

    if (lImporter->Import(mScene)) {
        // Check the scene integrity!
        FbxStatus status;
        FbxArray<FbxString*> details;
        FbxSceneCheckUtility sceneCheck(FbxCast<FbxScene>(mScene), &status,
                                        &details);
        bool lNotify =
            (!sceneCheck.Validate(FbxSceneCheckUtility::eCkeckData)
             && details.GetCount() > 0)
            || (lImporter->GetStatus().GetCode() != FbxStatus::eSuccess);
        if (lNotify) {
            printf("FbxImporter:\n");
            if (details.GetCount()) {
                printf(
                    "Scene integrity verification failed with the following "
                    "errors:\n");

                for (int i = 0; i < details.GetCount(); i++)
                    printf("   %s\n", details[i]->Buffer());

                FbxArrayDelete<FbxString*>(details);
            }

            if (lImporter->GetStatus().GetCode() != FbxStatus::eSuccess) {
                printf("\nWARNING:\n");
                printf(
                    "   The importer was able to read the file but with "
                    "errors.\n");
                printf("   Loaded scene may be incomplete.\n\n");
                printf("   Last error message:'%s'\n",
                       lImporter->GetStatus().GetErrorString());
            }
        }
    }

    lImporter->Destroy();
}

void Importer::InitializeData(ModelData::CISDI_3DModel& data,
                              const char* path) {
    data.header = {CISDI_3DModel_HEADER_UINT64, CISDI_3DModel_VERSION,
                   (uint32_t)mScene->GetNodeCount(),
                   (uint32_t)mScene->GetGeometryCount(),
                   (uint32_t)mScene->GetMaterialCount()};

    data.name = ::std::filesystem::path(path).stem().string();

    data.nodes.reserve(data.header.nodeCount);

    mTmpVertices.reserve(data.header.meshCount);

    data.materials.reserve(data.header.materialCount);
}

void Importer::ModifyGeometry() {
    FbxAxisSystem SceneAxisSystem = mScene->GetGlobalSettings().GetAxisSystem();
    FbxAxisSystem OurAxisSystem(FbxAxisSystem::eYAxis,
                                FbxAxisSystem::eParityOdd,
                                FbxAxisSystem::eRightHanded);
    if (SceneAxisSystem != OurAxisSystem) {
        OurAxisSystem.ConvertScene(mScene);
    }

    FbxSystemUnit SceneSystemUnit = mScene->GetGlobalSettings().GetSystemUnit();
    if (SceneSystemUnit.GetScaleFactor() != 1.0) {
        FbxSystemUnit::cm.ConvertScene(mScene);
    }

    // Convert mesh, NURBS and patch into triangle mesh
    FbxGeometryConverter lGeomConverter(mSdkManager);
    try {
        lGeomConverter.Triangulate(mScene, true);
    } catch (std::runtime_error&) {
        FBXSDK_printf("Scene integrity verification failed.\n");
    }
}

void Importer::ExtractMaterials(CISDI_3DModel& data) {
    materialIdxMap.clear();

    for (int i = 0; i < mScene->GetMaterialCount(); ++i) {
        FbxSurfaceMaterial* lMaterial = mScene->GetMaterial(i);
        materialIdxMap.emplace(lMaterial, i);

        CISDI_Material cisdiMaterial {pMemPool};
        cisdiMaterial.name = lMaterial->GetName();

        if (lMaterial->GetClassId().Is(FbxSurfacePhong::ClassId)) {
            cisdiMaterial.data.shadingModel =
                CISDI_Material::ShadingModel::Phong;

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
                CISDI_Material::ShadingModel::Lambert;

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
}

int Importer::ProcessNode(CISDI_3DModel& data, FbxNode* pNode,
                          int parentNodeIdx, bool flipYZ, bool meshData) {
    static int meshIdx {0};

    int nodeIdx = (int)data.nodes.size();
    int childCount = pNode->GetChildCount();

    CISDI_Node cisdiNode {pMemPool};
    cisdiNode.name = pNode->GetName();
    cisdiNode.parentIdx = parentNodeIdx;
    cisdiNode.childCount = childCount;
    cisdiNode.childrenIdx.reserve(childCount);

    if (FbxNodeAttribute* lNodeAttribute = pNode->GetNodeAttribute()) {
        switch (lNodeAttribute->GetAttributeType()) {
            case FbxNodeAttribute::eMesh:
                if (FbxMesh* mesh = pNode->GetMesh())
                    cisdiNode.meshIdx = ProcessMesh(mesh, flipYZ, meshData);
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
        ref.childrenIdx.emplace_back(
            ProcessNode(data, pNode->GetChild(i), nodeIdx, flipYZ, meshData));
    }

    return nodeIdx;
}

int Importer::ProcessMesh(FbxMesh* pMesh, bool flipYZ, bool meshData) {
    FbxVector4* lControlPoints = pMesh->GetControlPoints();

    int triangleCount = pMesh->GetPolygonCount();

    if (triangleCount == 0)
        return -1;

    int vertCount = triangleCount * 3;

    int meshIdx = (int)mTmpVertices.size();
    auto& tmpMeshVertices = mTmpVertices.emplace_back();

    if (!meshData)
        return meshIdx;

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

            // UVs
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

void Importer::ProcessUserDefinedProperties(FbxNode const* pNode,
                                            CISDI_Node& cisdiNode) {
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

    if (cisdiNode.userPropertyCount > 0) {
        cisdiNode.userProperties.emplace("_NumProperties_",
                                         cisdiNode.userPropertyCount);
        cisdiNode.userPropertyCount++;
    }
}

}  // namespace FBXSDK
}  // namespace IntelliDesign_NS::ModelImporter
