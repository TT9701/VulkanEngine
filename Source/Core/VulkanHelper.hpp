#pragma once

#include <vulkan/vulkan_to_string.hpp>

#include <iostream>

#include <Core/System/AP_ErrorLogger.h>

inline IntelliDesign_NS::Core::AP_ErrorLogger* GetLogger() {
    static ::std::unique_ptr<IntelliDesign_NS::Core::AP_ErrorLogger>
        pErrorLogger {new IntelliDesign_NS::Core::AP_ErrorLogger};
    return pErrorLogger.get();
}

#define VK_CHECK(x)                                                        \
    {                                                                      \
        vk::Result err = x;                                                \
        if (err != vk::Result::eSuccess) {                                 \
            ::std::cout << "Detected Vulkan error: " << vk::to_string(err) \
                        << "\n";                                           \
            abort();                                                       \
        }                                                                  \
    }

#ifdef DEBUG
#define DBG_LOG_INFO(...) GetLogger()->Log(__VA_ARGS__)
#else
#define DBG_LOG_INFO(...)
#endif
