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

template <typename T>
using UniquePtr = IntelliDesign_NS::Core::MemoryPool::Type_UniquePtr<T>;

template <typename T>
using SharedPtr = IntelliDesign_NS::Core::MemoryPool::Type_SharedPtr<T>;

template <typename T, typename... Types>
UniquePtr<T> MakeUnique(Types&&... val) {
    return IntelliDesign_NS::Core::MemoryPool::New_Unique<T>(
        MemoryPoolInstance::Get()->GetMemPoolResource(),
        ::std::forward<Types>(val)...);
}

template <typename T, typename... Types>
SharedPtr<T> MakeShared(Types&&... val) {
    return IntelliDesign_NS::Core::MemoryPool::New_Shared<T>(
        MemoryPoolInstance::Get()->GetMemPoolResource(),
        ::std::forward<Types>(val)...);
}