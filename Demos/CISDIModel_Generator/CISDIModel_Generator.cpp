#include <filesystem>
#include <vector>

#include <Windows.h>

#include "CISDI_3DModelData.h"

void GenerateModel(std::pmr::memory_resource* pMemPool,
                   ::std::vector<const char*> const& modelPathes) {
    int count = modelPathes.size();

    for (int i = 0; i < count; ++i) {
        if (!::std::filesystem::exists(modelPathes[i])) {
            printf("Model file %s does not exist.\n", modelPathes[i]);
            continue;
        }

        IntelliDesign_NS::ModelData::CISDI_3DModel model {pMemPool};

        Convert(&model, modelPathes[i], false, pMemPool);

        DWORD pid = GetCurrentProcessId();

        printf("[Pid: %ld] CISDI Model %d/%d: %s successfully generated.\n ",
               pid, i + 1, count, modelPathes[i]);
    }
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, ".utf8");
    // SetConsoleOutputCP(CP_UTF8);
    // SetConsoleCP(CP_UTF8);

    if (argc < 2) {
        printf("Usage: %s <model file path>\n", argv[0]);
        return 0;
    }

    ::std::vector<const char*> modelPathes(argc - 1);

    for (int i = 1; i < argc; ++i) {
        modelPathes[i - 1] = argv[i];
    }

    std::pmr::memory_resource* pool {::std::pmr::get_default_resource()};

    GenerateModel(pool, modelPathes);

    return 0;
}