#pragma once
#include <Windows.h>

struct CISDIModel_GeneratorProcess {
    CISDIModel_GeneratorProcess(const char8_t* commandLine);
    ~CISDIModel_GeneratorProcess();

    DWORD Wait();

    STARTUPINFO mStartupInfo = {sizeof(STARTUPINFO)};  //子进程的窗口相关信息
    PROCESS_INFORMATION mProcessInfo {};               //子进程的ID/线程相关信息
    DWORD mReturnCode {};                              //用于保存子程进的返回值;
    LPSTR mCommandLine;                                //测试命令行参数一
    BOOL mCreateSucess {false};
};