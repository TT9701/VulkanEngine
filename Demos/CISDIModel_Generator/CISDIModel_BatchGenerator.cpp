#include "CISDIModel_BatchGenerator.h"

#include "JSON//NlohmannJSON_v3_11_3/json.hpp"
#include "Core/System/MemoryPool/MemoryPool.h"

#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace IntelliDesign_NS::Core::MemoryPool;

constexpr const char* GENERATOR_EXECUTABLE_NAME = "CISDIModel_Generator.exe";

constexpr int MAX_PROCESS_COUNT = 8;

void WaitAllProcess(
    ::std::vector<CISDIModel_GeneratorProcess> const& processes) {
    ::std::vector<HANDLE> procHandles;
    for (auto const& proc : processes) {
        if (proc.mCreateSucess)
            procHandles.push_back(proc.mProcessInfo.hProcess);
    }

    auto processCount = procHandles.size();

    printf("%d processes executing...\n", processCount);
    WaitForMultipleObjects(processCount, procHandles.data(), TRUE, INFINITE);
    printf("Finished.\n");
}

CISDIModel_GeneratorProcess::CISDIModel_GeneratorProcess(
    const char* commandLine)
    : mCommandLine((char*)commandLine) {
    ZeroMemory(&mStartupInfo, sizeof(mStartupInfo));
    mStartupInfo.cb = sizeof(mStartupInfo);
    ZeroMemory(&mProcessInfo, sizeof(mProcessInfo));

    mCreateSucess = CreateProcess(
        NULL, mCommandLine,
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
    if (mCreateSucess) {
        // CloseHandle(mProcessInfo.hThread);
        // CloseHandle(mProcessInfo.hProcess);
    }
}

DWORD CISDIModel_GeneratorProcess::Wait() {
    if (mCreateSucess) {
        WaitForSingleObject(mProcessInfo.hProcess, INFINITE);
        GetExitCodeProcess(mProcessInfo.hProcess, &mReturnCode);
    }
    return mReturnCode;
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

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, ".utf8");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    ::std::vector<Type_STLString> modelPathes;
    Type_STLString outModelPath;

    if (argc != 2) {
        printf("No model path specified.\n");
        printf("usage: %s <model_path.json>\n", argv[0]);
        printf(
            "\t .json example: \n\t{\n\t\t\"input_pathes\": "
            "[\"model1.fbx\",\"model_dir\"], \n\t\t\"output_path\": "
            "\"output_dir\"\n\t}\n");
        return -1;
    } else {
        ::std::filesystem::path path(argv[1]);
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
            for (auto&& modelPath : modelPathes) {
                printf("Model path: %s\n", modelPath.c_str());
            }
        } else {
            printf("File not found: %s\n", path.string().c_str());
            return -1;
        }
    }

    int processCount = modelPathes.size() > MAX_PROCESS_COUNT
                         ? MAX_PROCESS_COUNT
                         : modelPathes.size();

    ::std::vector<::std::vector<Type_STLString>> modelPathesList(processCount);

    for (int i = 0; i < modelPathes.size(); ++i) {
        modelPathesList[i % processCount].push_back(modelPathes[i]);
    }

    ::std::vector<CISDIModel_GeneratorProcess> processes;

    for (int i = 0; i < processCount; ++i) {
        Type_STLString commandLine = GENERATOR_EXECUTABLE_NAME;
        for (auto&& path : modelPathesList[i]) {
            commandLine = commandLine + " ";
            commandLine += path;
        }
        commandLine = commandLine + " ";
        commandLine += outModelPath;
        processes.emplace_back(commandLine.c_str());
    }

    WaitAllProcess(processes);

    return 0;
}