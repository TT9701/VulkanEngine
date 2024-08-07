#pragma once

#include <assimp/scene.h>

#include "Mesh.hpp"

#define CISDI_3DModel_Subfix ::std::string(".cisdi")

constexpr uint64_t CISDI_3DModel_HEADER_UINT64 = 0x1111111111111111ui64;

struct CISDI_3DModelDataVersion {
    uint8_t  major;
    uint8_t  minor;
    uint16_t patch;
};

constexpr CISDI_3DModelDataVersion CISDI_3DModel_VERSION = {0ui8, 0ui8, 1ui16};

struct CISDI_3DModelData {
    struct Header {
        uint64_t                 header;
        CISDI_3DModelDataVersion version;
        uint32_t                 meshCount;
    } header;

    struct CISDI_Mesh {
        // TODO: 
        uint32_t vertexCount;
        uint32_t indexCount;

        struct Vertex {
            glm::vec3 position;
            glm::vec3 normal;
        };

        ::std::vector<Vertex>   vertices;
        ::std::vector<uint32_t> indices;
    };

    ::std::vector<CISDI_Mesh> meshes;
};

class CISDI_3DModelDataConverter {
public:
    CISDI_3DModelDataConverter(::std::string const& path, ::std::string const& ,
                               bool                 flipYZ = true);

    ::std::string Execute();

    ::std::vector<Mesh> LoadCISDIModelData(::std::string const& path);

private:
    void ProcessNode(::std::ofstream& out, aiNode* node, const aiScene* scene);

    void ProcessMesh(::std::ofstream& out, aiMesh* mesh);

    uint32_t CalcMeshCount(aiNode* node);

private:
    bool mFlipYZ;

    ::std::string mPath;
    ::std::string mDirectory;
    ::std::string mName;
};