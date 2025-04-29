#pragma once
#include <Windows.h>

struct CISDIModel_GeneratorProcess {
    CISDIModel_GeneratorProcess(const char* commandLine);
    ~CISDIModel_GeneratorProcess();

    DWORD Wait() const;

    STARTUPINFOW mStartupInfo = {sizeof(STARTUPINFOW)};  //子进程的窗口相关信息
    PROCESS_INFORMATION mProcessInfo {};               //子进程的ID/线程相关信息
    LPSTR mCommandLine;                                //测试命令行参数一
    BOOL mCreateSucess {false};
};