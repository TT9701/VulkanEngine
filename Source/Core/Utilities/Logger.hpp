#pragma once

#include <Core/System/AP_ErrorLogger.h>

inline IntelliDesign_NS::Core::AP_ErrorLogger* GetLogger() {
    static ::std::unique_ptr<IntelliDesign_NS::Core::AP_ErrorLogger>
        pErrorLogger {new IntelliDesign_NS::Core::AP_ErrorLogger};
    return pErrorLogger.get();
}

#ifndef NDEBUG
#define DBG_LOG_INFO(...) GetLogger()->Log(__VA_ARGS__)
#else
#define DBG_LOG_INFO(...)
#endif
