#pragma once

static constexpr uint64_t WAIT_NEXT_IMAGE_TIME_OUT = 1000000000;
static constexpr uint64_t DEFAULT_FENCE_TIME_OUT = 1000000000;

// #define MAX_BINDLESS_DESCRIPTOR_COUNT (1048576ui32)
#define MAX_BINDLESS_DESCRIPTOR_COUNT (64ui32)

#define SHADER_PATH(name)                                                 \
    (IntelliDesign_NS::Core::MemoryPool::Type_STLString("../../Shaders/") \
     + name)
#define SHADER_PATH_CSTR(name) SHADER_PATH(name).c_str()

#define MODEL_PATH(name)                                                    \
    (IntelliDesign_NS::Core::MemoryPool::Type_STLString("../../../Models/") \
     + name)
#define MODEL_PATH_CSTR(name) MODEL_PATH(name).c_str()

#define NV_PREFERRED_MESH_SHADER_MAX_VERTICES 64
#define NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES 124

#define TASK_SHADER_INVOCATION_COUNT 32
#define MESH_SHADER_INVOCATION_COUNT 32

#define VE_ASSERT(expr, message) \
    {                            \
        void(message);           \
        assert(expr);            \
    }

#define CLASS_NO_COPY(CLASS_NAME)           \
    CLASS_NAME(const CLASS_NAME&) = delete; \
    CLASS_NAME& operator=(const CLASS_NAME&) = delete

#define CLASS_NO_MOVE(CLASS_NAME)               \
    CLASS_NAME(CLASS_NAME&&) noexcept = delete; \
    CLASS_NAME& operator=(CLASS_NAME&&) noexcept = delete

#define CLASS_NO_COPY_MOVE(CLASS_NAME) \
    CLASS_NO_COPY(CLASS_NAME);         \
    CLASS_NO_MOVE(CLASS_NAME)

#define CLASS_MOVABLE_ONLY(CLASS_NAME)                 \
    CLASS_NAME(const CLASS_NAME&) = delete;            \
    CLASS_NAME& operator=(const CLASS_NAME&) = delete; \
    CLASS_NAME(CLASS_NAME&&) noexcept = default;       \
    CLASS_NAME& operator=(CLASS_NAME&&) noexcept = default

#define VE_STRINGIFY_MACRO(x) #x