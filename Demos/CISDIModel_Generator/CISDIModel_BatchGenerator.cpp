#include <omp.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <Core/System/GameTimer.h>
#include <Core/System/MemoryPool/MemoryPool.h>
#include <Win32/DbgMemLeak.h>
#include <Core/System/StringHelper.hpp>
#include <JSON/NlohmannJSON_v3_11_3/json.hpp>

#include "CISDIModel_BatchGenerator.h"

using namespace IntelliDesign_NS::Core::MemoryPool;

constexpr const char* GENERATOR_EXECUTABLE_NAME =
    "\"CISDIModel_Generator.exe\"";

namespace {
void WaitAllProcess(
    ::std::vector<CISDIModel_GeneratorProcess> const& processes) {
    ::std::vector<HANDLE> procHandles;
    for (auto const& proc : processes) {
        if (proc.mCreateSucess)
            procHandles.push_back(proc.mProcessInfo.hProcess);
    }

    auto processCount = procHandles.size();

    printf("%zu processes executing...\n", processCount);
    WaitForMultipleObjects(processCount, procHandles.data(), TRUE, INFINITE);
    printf("Finished.\n");
}

void WaitSingleProcess(HANDLE hProc) {
    WaitForSingleObjectEx(hProc, INFINITE, false);
}

void ReadModelNameFromDirectory(const ::std::filesystem::path& path,
                                ::std::vector<Type_STLString>& modelPathes) {
    for (const auto& entry : ::std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".fbx") {
            modelPathes.push_back(entry.path().string().c_str());
        }
    }
}

void ReadModelNameFromJSON(const ::std::filesystem::path& jsonFile,
                           ::std::vector<Type_STLString>& inModelPathes,
                           Type_STLString& outModelPath) {
    ::std::ifstream ifs(jsonFile);

    using json = nlohmann::json;
    json j;

    ifs >> j;

    for (uint32_t i = 0; i < j["input_pathes"].size(); ++i) {
        auto pathStr = j["input_pathes"][i].get<Type_STLString>();
        auto path = ::std::filesystem::path(pathStr.c_str());
        if (::std::filesystem::is_directory(path)) {
            ReadModelNameFromDirectory(path, inModelPathes);
        } else {
            inModelPathes.push_back(path.string().c_str());
        }
    }

    outModelPath = j["output_path"].get<Type_STLString>();
}
}  // namespace

CISDIModel_GeneratorProcess::CISDIModel_GeneratorProcess(
    const char* commandLine)
    : mCommandLine((char*)commandLine) {
    ZeroMemory(&mStartupInfo, sizeof(mStartupInfo));
    mStartupInfo.cb = sizeof(mStartupInfo);
    ZeroMemory(&mProcessInfo, sizeof(mProcessInfo));

    auto cmdLineW =
        IntelliDesign_NS::Core::StringHelper::StringViewToWString(commandLine);

    mCreateSucess = CreateProcessW(
        NULL, const_cast<LPWSTR>(cmdLineW.c_str()),
        NULL,   //_In_opt_    LPSECURITY_ATTRIBUTES lpProcessAttributes,
        NULL,   //_In_opt_    LPSECURITY_ATTRIBUTES lpThreadAttributes,
        FALSE,  //_In_        BOOL                  bInheritHandles,
        0,
        NULL,            //_In_opt_    LPVOID                lpEnvironment,
        NULL,            //_In_opt_    LPCTSTR               lpCurrentDirectory,
        &mStartupInfo,   //_In_        LPSTARTUPINFO         lpStartupInfo,
        &mProcessInfo);  //_Out_       LPPROCESS_INFORMATION lpProcessInformation

    if (!mCreateSucess) {
        std::cout << "Create Process error!\n";
    }
}

CISDIModel_GeneratorProcess::~CISDIModel_GeneratorProcess() {
    Wait();
}

DWORD CISDIModel_GeneratorProcess::Wait() const {
    DWORD retCode {};  //用于保存子程进的返回值;

    if (mCreateSucess) {
        WaitForSingleObject(mProcessInfo.hProcess, INFINITE);
        GetExitCodeProcess(mProcessInfo.hProcess, &retCode);
    }

    return retCode;
}

static constexpr int ProgressWidth = 30;
static constexpr const char* ProgressStr = "##############################";

void PrintProgress(int offset, int total, const char* msg) {

    int val = offset * 100 / total;

    int lpad = offset * ProgressWidth / total;

    int rpad = ProgressWidth - lpad;

    printf("\r\033[K");

    printf("\r%3d%% [%.*s%*s] [%d/%d]", val, lpad, ProgressStr, rpad, "",
           offset, total);

    printf(": %s.\r", msg);

    fflush(stdout);
}

int main(int argc, char* argv[]) {
    INTELLI_DS_SetDbgFlag();

    ::std::filesystem::path pathApp {argv[0]};

    ::std::filesystem::path pathJSONANSI {};
    if (argc >= 2)
        pathJSONANSI = argv[1];

    unsigned int numProcsMax = 0xffffffff;
    if (argc >= 3)
        numProcsMax = atoi(argv[2]);
    /**************************************************/
    setlocale(LC_ALL, ".utf8");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    /**************************************************/
    ::std::vector<Type_STLString> modelPathes;
    Type_STLString outModelPath;

    if (argc < 2) {
        printf("No model path specified.\n");
        printf("usage: %s <model_path.json>\n", pathApp.u8string().c_str());
        printf(
            "\t .json example: \n\t{\n\t\t\"input_pathes\": "
            "[\"model1.fbx\",\"model_dir\"], \n\t\t\"output_path\": "
            "\"output_dir\"\n\t}\n");
        return -1;
    } else {
        ::std::filesystem::path path = pathJSONANSI;
        if (::std::filesystem::exists(path)) {
            ReadModelNameFromJSON(path, modelPathes, outModelPath);

            printf("Output path: %s\n", outModelPath.c_str());
            if (!::std::filesystem::exists(outModelPath.c_str())) {
                printf(
                    "Output path %s does not exist. Create a new one.\n",
                    outModelPath
                        .c_str());  // Updated to use string() for correct output

                // Create the output directory if it doesn't exist
                ::std::filesystem::create_directories(outModelPath.c_str());
            }

            printf("Number of models found: %zu\n", modelPathes.size());
            /*for (auto&& modelPath : modelPathes) {
                printf("Model path: %s\n", modelPath.c_str());
            }*/
        } else {
            printf("JSON file not found: %s\n", path.string().c_str());
            return -1;
        }
    }

    auto maxProcCount =
        __min(::std::thread::hardware_concurrency(), numProcsMax);

    int processCount = __min(maxProcCount, modelPathes.size());
    auto numTsksTotal = modelPathes.size();

    ::std::atomic<size_t> numTsksFetched {0};
    ::std::atomic<size_t> numTsksCompleted {0};

    GameTimer timer0 {};
    INTELLI_DS_MEASURE_DURATION_MS_START(timer0) {
#pragma omp parallel for
        for (int i = 0; i < processCount; ++i) {
            while (true) {
                size_t idxTsk = numTsksFetched++;
                if (idxTsk < numTsksTotal) {
                    auto commandLine =
                        Type_STLString {GENERATOR_EXECUTABLE_NAME} + " \""
                        + modelPathes[idxTsk] + "\"" + " \"" + outModelPath
                        + "\"";
                    CISDIModel_GeneratorProcess {commandLine.c_str()};

                    auto modelFileName =
                        ::std::filesystem::path {modelPathes[idxTsk].c_str()}
                            .filename()
                            .string();
                    auto idxTskCompleted = numTsksCompleted++;

                    // printf("Conversion completed (%zu/%zu): %s\n",
                    //        idxTskCompleted + 1, numTsksTotal,
                    //        modelFileName.c_str());

                    PrintProgress(static_cast<int>(idxTskCompleted + 1),
                                  static_cast<int>(numTsksTotal),
                                  modelFileName.c_str());

                } else
                    break;
            }
        }
    }

    printf("\r\033[K");
    INTELLI_DS_MEASURE_DURATION_MS_END_PRINT(timer0, "FBX conversion");

    return 0;
}