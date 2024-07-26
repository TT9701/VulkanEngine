#pragma once

#include "Core/System/MemoryPool/MemoryPool.h"
#include "Defines.hpp"

class MemoryPoolInstance {
public:
    MemoryPoolInstance() : mResource(::std::pmr::get_default_resource()) {}

    ~MemoryPoolInstance() = default;
    MOVABLE_ONLY(MemoryPoolInstance);

public:
    static MemoryPoolInstance* Get() {
        static MemoryPoolInstance inst;
        return &inst;
    }

    ::std::pmr::memory_resource* GetMemPoolResource() const {
        return mResource;
    }

private:
    ::std::pmr::memory_resource* mResource;
};

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