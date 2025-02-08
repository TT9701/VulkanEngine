#include "CISDIModel_BatchGenerator.h"

#include <filesystem>
#include <iostream>
#include <vector>

constexpr const char8_t* GENERATOR_EXECUTABLE_NAME = u8"CISDIModel_Generator.exe";
constexpr int MAX_PROCESS_COUNT = 8;

void WaitAllProcess(
    ::std::vector<CISDIModel_GeneratorProcess> const& processes) {
    ::std::vector<HANDLE> procHandles;
    for (auto const& proc : processes) {
        if (proc.mCreateSucess)
            procHandles.push_back(proc.mProcessInfo.hProcess);
    }

    printf("Wait all process executing...\n");
    WaitForMultipleObjects(procHandles.size(), procHandles.data(), TRUE,
                           INFINITE);
    printf("All process finished.\n");
}

CISDIModel_GeneratorProcess::CISDIModel_GeneratorProcess(
    const char8_t* commandLine)
    : mCommandLine((char*)commandLine) {
    ZeroMemory(&mStartupInfo, sizeof(mStartupInfo));
    mStartupInfo.cb = sizeof(mStartupInfo);
    ZeroMemory(&mProcessInfo, sizeof(mProcessInfo));

    mCreateSucess = CreateProcess(  
        NULL,  
        mCommandLine,   
        NULL,           //_In_opt_    LPSECURITY_ATTRIBUTES lpProcessAttributes,
        NULL,           //_In_opt_    LPSECURITY_ATTRIBUTES lpThreadAttributes,
        FALSE,          //_In_        BOOL                  bInheritHandles,
        0,              
        NULL,           //_In_opt_    LPVOID                lpEnvironment,
        NULL,           //_In_opt_    LPCTSTR               lpCurrentDirectory,
        &mStartupInfo,  //_In_        LPSTARTUPINFO         lpStartupInfo,
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

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, ".utf8");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    if (argc < 2) {
        printf(
            "Usage: %s <model_path1>|<directory1> <model_path2>"
            "|<directory2> ...\n",
            argv[0]);
        return 1;
    }

    ::std::vector<::std::u8string> modelPathes;

    for (int i = 1; i < argc; ++i) {
        ::std::filesystem::path path(argv[i]);
        if (std::filesystem::is_directory(path)) {
            for (const auto& entry :
                 std::filesystem::directory_iterator(path)) {
                if (entry.path().extension() == ".fbx") {
                    modelPathes.push_back(entry.path().u8string());
                }
            }
        } else {
            modelPathes.push_back(path.u8string());
        }
    }

    int processCount = modelPathes.size() > MAX_PROCESS_COUNT
                         ? MAX_PROCESS_COUNT
                         : modelPathes.size();

    ::std::vector<::std::vector<::std::u8string>> modelPathesList(processCount);

    for (int i = 0; i < modelPathes.size(); ++i) {
        modelPathesList[i % processCount].push_back(modelPathes[i]);
    }

    ::std::vector<CISDIModel_GeneratorProcess> processes;

    for (int i = 0; i < processCount; ++i) {
        ::std::u8string commandLine = GENERATOR_EXECUTABLE_NAME;
        for (auto && path : modelPathesList[i]) {
            commandLine += u8" ";
            commandLine += path;
        }
        processes.emplace_back(commandLine.c_str());
    }

    WaitAllProcess(processes);

    printf("Sucess.\n");

    return 0;
}