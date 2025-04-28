#include <filesystem>
#include <vector>

#include <Windows.h>

#include "CISDI_3DModelData.h"

void GenerateModel(std::pmr::memory_resource* pMemPool,
                   ::std::vector<::std::filesystem::path> const& modelPathes,
                   ::std::filesystem::path const& outPath) {
    int count = modelPathes.size();

    for (int i = 0; i < count; ++i) {
        if (!::std::filesystem::exists(modelPathes[i])) {
            printf("Model file %s does not exist.\n",
                   modelPathes[i].string().c_str());
            continue;
        }

        IntelliDesign_NS::ModelData::CISDI_3DModel model {pMemPool};

        Convert(&model, modelPathes[i].string().c_str(), false, pMemPool,
                outPath.string().c_str());

        DWORD pid = GetCurrentProcessId();

        printf("[Pid: %ld] CISDI Model %d/%d: %s successfully generated.\n ",
               pid, i + 1, count, modelPathes[i].string().c_str());
    }
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, ".utf8");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    if (argc < 3) {
        printf("Usage: %s <model_file_path 1> ... <out_path>\n", argv[0]);
        return 0;
    }

    ::std::vector<::std::filesystem::path> modelPathes(argc - 2);

    for (int i = 1; i < argc - 1; ++i) {
        modelPathes[i - 1] = ::std::filesystem::path {argv[i]};
    }

    auto outPath = ::std::filesystem::path {argv[argc - 1]};

    std::pmr::memory_resource* pool {::std::pmr::get_default_resource()};

    GenerateModel(pool, modelPathes, outPath);

    return 0;
}