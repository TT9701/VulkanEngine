#pragma once

#define SHADER_PATH(name) (::std::string("../../Shaders/") + name)
#define SHADER_PATH_CSTR(name) SHADER_PATH(name).c_str()

#define MODEL_PATH(name) (::std::string("../../../Models/") + name)
#define MODEL_PATH_CSTR(name) MODEL_PATH(name).c_str()

using Type_ShaderMacros = ::std::unordered_map<::std::string, ::std::string>;

#define NV_PREFERRED_MESH_SHADER_MAX_VERTICES 64
#define NV_PREFERRED_MESH_SHADER_MAX_PRIMITIVES 124

#define TASK_SHADER_INVOCATION_COUNT 32
#define MESH_SHADER_INVOCATION_COUNT 32

#define VE_ASSERT(expr, message) \
    {                            \
        void(message);           \
        assert(expr);            \
    }

#define MOVABLE_ONLY(CLASS_NAME)                       \
    CLASS_NAME(const CLASS_NAME&) = delete;            \
    CLASS_NAME& operator=(const CLASS_NAME&) = delete; \
    CLASS_NAME(CLASS_NAME&&) noexcept = default;       \
    CLASS_NAME& operator=(CLASS_NAME&&) noexcept = default