#include <filesystem>
#include <vector>

#include "CISDI_3DModelData.h"

void GenerateModel(std::pmr::memory_resource* pMemPool,
                   ::std::vector<const char*> const& modelPathes) {
    int count = modelPathes.size();

    for (int i = 0; i < count; ++i) {
        if (!::std::filesystem::exists(modelPathes[i])) {
            printf("Model file %s does not exist.\n", modelPathes[i]);
            continue;
        }

        printf("Generating CISDI Model %d/%d from: %s. \n", i + 1, count,
               modelPathes[i]);

        IntelliDesign_NS::ModelData::Convert(modelPathes[i], false, pMemPool);
    }

    printf("Model data generation completed.\n");
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