#include <filesystem>
#include <vector>

#include <Windows.h>

#include <Core/System/MemoryPool/MemoryPool.h>
#include <Win32/DbgMemLeak.h>

#include "CISDI_3DModelData.h"

using Type_VecTempObject = ::std::vector<
    IntelliDesign_NS::ModelData::CISDI_3DModel::TempObject::Type_PInstance>;

Type_VecTempObject GenerateModel(
    std::pmr::memory_resource* pMemPool,
    ::std::vector<::std::filesystem::path> const& modelPathes,
    ::std::filesystem::path const& outPath) {
    Type_VecTempObject vecTmpObj {};

    int count = modelPathes.size();

    for (int i = 0; i < count; ++i) {
        if (!::std::filesystem::exists(modelPathes[i])) {
            printf("Model file %s does not exist.\n",
                   modelPathes[i].u8string().c_str());
            continue;
        }

        IntelliDesign_NS::ModelData::CISDI_3DModel model {pMemPool};

        auto ret = Convert(
            &model,
            reinterpret_cast<char const*>(modelPathes[i].u8string().c_str()),
            false, pMemPool,
            reinterpret_cast<char const*>(outPath.u8string().c_str()));
        vecTmpObj.emplace_back(::std::move(ret));

        DWORD pid = GetCurrentProcessId();
    }

    return vecTmpObj;
}

int wmain(int argc, wchar_t* argv[], wchar_t* envp[]) {
    INTELLI_DS_SetDbgFlag();

    setlocale(LC_ALL, ".utf8");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    if (argc < 3) {
        wprintf(L"Usage: %s <model_file_path 1> ... <out_path>\n", argv[0]);
        return 0;
    }

    ::std::vector<::std::filesystem::path> modelPathes(argc - 2);

    for (int i = 1; i < argc - 1; ++i) {
        modelPathes[i - 1] = ::std::filesystem::path {argv[i]};
    }

    auto outPath = ::std::filesystem::path {argv[argc - 1]};

    std::pmr::memory_resource* pool {::std::pmr::get_default_resource()};

    auto vecTmpObj = GenerateModel(pool, modelPathes, outPath);

    exit(0);

    return 0;
}