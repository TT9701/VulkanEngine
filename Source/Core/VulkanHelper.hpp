#pragma once

#include <vulkan/vulkan_to_string.hpp>
#include "Core/System/MemoryPool/MemoryPool.h"

#include <iostream>

#define VK_CHECK(x)                                                        \
    {                                                                      \
        vk::Result err = x;                                                \
        if (err != vk::Result::eSuccess) {                                 \
            ::std::cout << "Detected Vulkan error: " << vk::to_string(err) \
                        << "\n";                                           \
            abort();                                                       \
        }                                                                  \
    }

#define USING_UNIQUE_PTR_TYPE(name, T) \
    using name = IntelliDesign_NS::Core::MemoryPool::Type_UniquePtr<T>

#define USING_SHARED_PTR_TYPE(name, T) \
    using name = IntelliDesign_NS::Core::MemoryPool::Type_SharedPtr<T>

#define USING_PTR_TYPE(uniquePtrName, sharedPtrName, T) \
    USING_UNIQUE_PTR_TYPE(uniquePtrName, T);            \
    USING_SHARED_PTR_TYPE(sharedPtrName, T)

#define USING_TEMPLATE_UNIQUE_PTR_TYPE(name) \
    template <class T>                       \
    using name = IntelliDesign_NS::Core::MemoryPool::Type_UniquePtr<T>

#define USING_TEMPLATE_SHARED_PTR_TYPE(name) \
    template <class T>                       \
    using name = IntelliDesign_NS::Core::MemoryPool::Type_SharedPtr<T>

#define USING_TEMPLATE_PTR_TYPE(uniquePtrName, sharedPtrName) \
    USING_TEMPLATE_UNIQUE_PTR_TYPE(uniquePtrName);            \
    USING_TEMPLATE_SHARED_PTR_TYPE(sharedPtrName)

#include "Utilities/Logger.hpp"