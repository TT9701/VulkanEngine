#pragma once

#include "Core/Vulkan/Manager/VulkanContext.h"

#include "Core/Utilities/MemoryPool.h"
#include "Core/Vulkan/Native/Shader.h"

// #define EXPAND(x) x
//
// #define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, \
//                   _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26,  \
//                   _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38,  \
//                   _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50,  \
//                   _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62,  \
//                   _63, _64, Name, ...)                                         \
//     Name
//
// #define GET_MACRO_NUM5(_1, _2, _3, _4, _5, Name, ...) Name
//
// #define GET_ARG_COUNT(...)                                                    \
//     EXPAND(GET_MACRO(__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, \
//                      53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40,  \
//                      39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,  \
//                      25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,  \
//                      11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
//
// #define PAIR(x) PARE x  // PAIR((double) x) => PARE(double) x => double x
// #define PARE(...) __VA_ARGS__
//
// #define EAT(...)
// #define STRIP(x) EAT x  // STRIP((double) x) => EAT(double) x => x
//
// #define STRING(x) STR(x)
// #define STR(x) #x
//
// #define PASTE(x, y) CONCATE(x, y)
// #define CONCATE(x, y) x##y
//
// #define FOR_EACH_1(func, i, arg) func(i, arg);
// #define FOR_EACH_2(func, i, arg, ...) \
//     func(i, arg);                     \
//     EXPAND(FOR_EACH_1(func, i + 1, __VA_ARGS__))
// #define FOR_EACH_3(func, i, arg, ...) \
//     func(i, arg);                     \
//     EXPAND(FOR_EACH_2(func, i + 1, __VA_ARGS__))
// #define FOR_EACH_4(func, i, arg, ...) \
//     func(i, arg);                     \
//     EXPAND(FOR_EACH_3(func, i + 1, __VA_ARGS__))
// #define FOR_EACH_5(func, i, arg, ...) \
//     func(i, arg);                     \
//     EXPAND(FOR_EACH_4(func, i + 1, __VA_ARGS__))
//
// #define INTELLI_DS_MACRO_BATCH_NUM_5(func, i, ...)                         \
//     EXPAND(GET_MACRO_NUM5(__VA_ARGS__, FOR_EACH_5, FOR_EACH_4, FOR_EACH_3, \
//                           FOR_EACH_2, FOR_EACH_1)(func, i, __VA_ARGS__))
//
// #define FIELD_EACH(i, arg)                                          \
// public:                                                             \
//     PAIR(arg);                                                      \
//                                                                     \
// private:                                                            \
//     template <typename T>                                           \
//     struct FIELD<T, i> {                                            \
//         T& obj;                                                     \
//         FIELD(T& obj) : obj(obj) {}                                 \
//         auto value() -> decltype(auto) {                            \
//             return (obj.STRIP(arg));                                \
//         }                                                           \
//         static constexpr uint32_t offset {offsetof(T, STRIP(arg))}; \
//         using Type = ::std::decay_t<decltype(obj.STRIP(arg))>;      \
//     }
//
// #define DEFINE_STRUCT_INTERNAL(st, ...)                                    \
//     struct st {                                                            \
//         static constexpr size_t _field_count_ =                            \
//             EXPAND(GET_ARG_COUNT(__VA_ARGS__));                            \
//                                                                            \
//     private:                                                               \
//         template <typename, size_t>                                        \
//         struct FIELD;                                                      \
//         EXPAND(INTELLI_DS_MACRO_BATCH_NUM_5(FIELD_EACH, 0, __VA_ARGS__))   \
//     public:                                                                \
//         template <size_t Idx>                                              \
//         decltype(auto) Get() {                                             \
//             return typename st::template FIELD<st, Idx>(*this).value();    \
//         }                                                                  \
//         template <size_t Idx>                                              \
//         static uint32_t GetOffset() {                                      \
//             return FIELD<st, Idx>::offset;                                 \
//         }                                                                  \
//         template <typename F, size_t... Is>                                \
//         inline constexpr void forEach(F&& f, std::index_sequence<Is...>) { \
//             using TDECAY = std::decay_t<decltype(*this)>;                  \
//             (void(f(typename st::template FIELD<st, Is>(*this).value())),  \
//              ...);                                                         \
//         }                                                                  \
//         template <typename F>                                              \
//         inline constexpr void forEach(F&& f) {                             \
//             forEach(std::forward<F>(f),                                    \
//                     std::make_index_sequence<_field_count_> {});           \
//         }                                                                  \
//         template <size_t Idx>                                              \
//         struct Type_Element {                                              \
//             using Type = typename st::template FIELD<st, Idx>::Type;       \
//         };                                                                 \
//         template <size_t Idx>                                              \
//         using Type_Element_t = typename Type_Element<Idx>::Type;           \
//     }
//
// #define DEFINE_STRUCT_COMPUTE_true(st, ...) \
//     DEFINE_STRUCT_INTERNAL(st, __VA_ARGS__, \
//                            (vk::DispatchIndirectCommand)_dispatchCommand_)
//
// #define DEFINE_STRUCT_COMPUTE_false(st, ...) \
//     DEFINE_STRUCT_INTERNAL(                  \
//         st, __VA_ARGS__,                     \
//         (vk::DrawIndirectCountIndirectCommandEXT)_drawCommand_)
//
// #define DEFINE_STRUCT_MULTIPIPELINE_true(st, bIsCompute, ...) \
//     DEFINE_STRUCT_COMPUTE_##bIsCompute(st, (uint32_t)_pipelineIdx_, __VA_ARGS__)
//
// #define DEFINE_STRUCT_MULTIPIPELINE_false(st, bIsCompute, ...) \
//     DEFINE_STRUCT_COMPUTE_##bIsCompute(st, __VA_ARGS__)
//
// #define DEFINE_STRUCT(st, bMultiPipeline, bIsCompute, ...) \
//     DEFINE_STRUCT_MULTIPIPELINE_##bMultiPipeline(st, bIsCompute, __VA_ARGS__)

template <bool IsCompute>
struct DrawCommandContainer;

template <>
struct DrawCommandContainer<true> {
    using _Type_ = vk::DispatchIndirectCommand;
};

template <>
struct DrawCommandContainer<false> {
    using _Type_ = vk::DrawIndirectCountIndirectCommandEXT;
};

template <bool IsCompute>
using DrawCommandContainer_t = typename DrawCommandContainer<IsCompute>::_Type_;

template <bool UsePipeline, uint32_t ShaderCount = 1>
struct ExecutionSetInfo;

/**
 *  Execution Set use pipeline index
 */
template <>
struct ExecutionSetInfo<true> {
    static constexpr bool _UsePipeline_ = true;

    using _Type_ES_ = uint32_t;
};

/**
 *  Execution Set use shader object index
 */
template <uint32_t ShaderCount>
struct ExecutionSetInfo<false, ShaderCount> {
    static constexpr bool _UsePipeline_ = false;
    static constexpr uint32_t _ShaderCount_ = ShaderCount;

    using _Type_ES_ = ::std::array<uint32_t, ShaderCount>;
};

#define DECL_INTERNAL_INFO2(IsCompute, UseES, TPushConstant) \
    static constexpr bool _IsCompute_ = IsCompute;           \
    static constexpr bool _UseExecutionSet_ = UseES;         \
    using _Type_PushConstant_ = TPushConstant

template <bool IsCompute, class ExecutionSetInfo = void,
          class TPushConstant = void>
struct SequenceTemplate2;

template <bool IsCompute>
struct SequenceTemplate2<IsCompute, void, void> {
    DECL_INTERNAL_INFO2(IsCompute, false, void);

    DrawCommandContainer_t<IsCompute> command;
};

template <bool IsCompute, class TPushConstant>
struct SequenceTemplate2<IsCompute, void, TPushConstant> {
    DECL_INTERNAL_INFO2(IsCompute, false, TPushConstant);

    TPushConstant pushConstant;
    DrawCommandContainer_t<IsCompute> command;
};

template <bool IsCompute, class ExecutionSetInfo>
struct SequenceTemplate2<IsCompute, ExecutionSetInfo, void> {
    DECL_INTERNAL_INFO2(IsCompute, true, void);
    static constexpr bool _UsePipeline_ = ExecutionSetInfo::_UsePipeline_;

    typename ExecutionSetInfo::_Type_ES_ index;
    DrawCommandContainer_t<IsCompute> command;
};

template <bool IsCompute, class ExecutionSetInfo>
concept UsePipeline = ExecutionSetInfo::_UsePipeline_;

template <bool IsCompute, class ExecutionSetInfo>
concept GraphicsShaderCountGreaterThanOne =
    !IsCompute && !UsePipeline<IsCompute, ExecutionSetInfo>
    && ExecutionSetInfo::_ShaderCount_ > 1;

template <bool IsCompute, class ExecutionSetInfo>
concept ComputeShaderCountEqualsOne =
    IsCompute && !UsePipeline<IsCompute, ExecutionSetInfo>
    && ExecutionSetInfo::_ShaderCount_ == 1;

template <bool IsCompute, class ExecutionSetInfo>
concept ValidExecutionSetInfo =
    UsePipeline<IsCompute, ExecutionSetInfo>
    || GraphicsShaderCountGreaterThanOne<IsCompute, ExecutionSetInfo>
    || ComputeShaderCountEqualsOne<IsCompute, ExecutionSetInfo>;

template <bool IsCompute, class ExecutionSetInfo, class TPushConstant>
    requires ValidExecutionSetInfo<IsCompute, ExecutionSetInfo>
struct SequenceTemplate2<IsCompute, ExecutionSetInfo, TPushConstant> {
    DECL_INTERNAL_INFO2(IsCompute, true, TPushConstant);
    static constexpr bool _UsePipeline_ = ExecutionSetInfo::_UsePipeline_;

    typename ExecutionSetInfo::_Type_ES_ index;
    TPushConstant pushConstant;
    DrawCommandContainer_t<IsCompute> command;
};

enum class DGCExecutionSetType { None, Pipeline, Shader_Dispatch, Shader_Draw };

template <bool IsCompute, DGCExecutionSetType ESType,
          class TPushConstant = void>
struct DGCSequenceTemplate;

template <bool IsCompute, class TPushConstant>
struct DGCSequenceTemplate<IsCompute, DGCExecutionSetType::None, TPushConstant>
    : SequenceTemplate2<IsCompute, void, TPushConstant> {};

template <bool IsCompute, class TPushConstant>
struct DGCSequenceTemplate<IsCompute, DGCExecutionSetType::Pipeline,
                           TPushConstant>
    : SequenceTemplate2<IsCompute, ExecutionSetInfo<true>, TPushConstant> {};

template <class TPushConstant>
struct DGCSequenceTemplate<true, DGCExecutionSetType::Shader_Dispatch,
                           TPushConstant>
    : SequenceTemplate2<true, ExecutionSetInfo<false>, TPushConstant> {};

template <class TPushConstant>
struct DGCSequenceTemplate<false, DGCExecutionSetType::Shader_Draw,
                           TPushConstant>
    : SequenceTemplate2<false, ExecutionSetInfo<false, 3>, TPushConstant> {};

namespace IntelliDesign_NS::Vulkan::Core {

class SequenceLayout {
public:
    explicit SequenceLayout(VulkanContext& context);
    ~SequenceLayout();

    vk::IndirectCommandsLayoutEXT GetHandle() const;

private:
    template <class TDGCSequenceTemplate>
    friend Type_UniquePtr<SequenceLayout> CreateLayout(
        VulkanContext& context, vk::PipelineLayout pipelineLayout,
        bool unorderedSequence, bool explicitPreprocess);

private:
    VulkanContext& mContext;

    vk::IndirectCommandsLayoutEXT mHandle;
};

template <class TDGCSequenceTemplate>
Type_UniquePtr<SequenceLayout> CreateLayout(VulkanContext& context,
                                            vk::PipelineLayout pipelineLayout,
                                            bool unorderedSequence = false,
                                            bool explicitPreprocess = false) {
    constexpr bool hasPushConstant =
        !::std::is_same_v<typename TDGCSequenceTemplate::_Type_PushConstant_,
                          void>;
    constexpr bool isCompute = TDGCSequenceTemplate::_IsCompute_;
    constexpr bool useES = TDGCSequenceTemplate::_UseExecutionSet_;

    auto layout = MakeUnique<SequenceLayout>(context);

    constexpr vk::ShaderStageFlags stages =
        isCompute ? vk::ShaderStageFlagBits::eCompute
                  : vk::ShaderStageFlagBits::eTaskEXT
                        | vk::ShaderStageFlagBits::eMeshEXT
                        | vk::ShaderStageFlagBits::eFragment;

    Type_STLVector<vk::IndirectCommandsLayoutTokenEXT> tokenDatas {};

    vk::IndirectCommandsExecutionSetTokenEXT esToken {};
    if constexpr (useES) {
        if constexpr (TDGCSequenceTemplate::_UsePipeline_) {
            esToken.setType(vk::IndirectExecutionSetInfoTypeEXT::ePipelines);
        } else {
            esToken.setType(
                vk::IndirectExecutionSetInfoTypeEXT::eShaderObjects);
        }
        esToken.setShaderStages(stages);

        vk::IndirectCommandsTokenDataEXT esTokenData {};
        esTokenData.setPExecutionSet(&esToken);

        vk::IndirectCommandsLayoutTokenEXT token {};
        token.setType(vk::IndirectCommandsTokenTypeEXT::eExecutionSet)
            .setOffset(offsetof(TDGCSequenceTemplate, index))
            .setData(esTokenData);

        tokenDatas.push_back(token);
    }

    vk::IndirectCommandsPushConstantTokenEXT pcToken {};
    if constexpr (hasPushConstant) {
        pcToken.setUpdateRange(
            {stages, 0,
             sizeof(typename TDGCSequenceTemplate::_Type_PushConstant_)});

        vk::IndirectCommandsTokenDataEXT pcTokenData {};
        pcTokenData.setPPushConstant(&pcToken);

        vk::IndirectCommandsLayoutTokenEXT token {};
        token.setType(vk::IndirectCommandsTokenTypeEXT::ePushConstant)
            .setOffset(offsetof(TDGCSequenceTemplate, pushConstant))
            .setData(pcTokenData);

        tokenDatas.push_back(token);
    }

    vk::IndirectCommandsLayoutTokenEXT idCmdtoken {};
    idCmdtoken
        .setType(isCompute
                     ? vk::IndirectCommandsTokenTypeEXT::eDispatch
                     : vk::IndirectCommandsTokenTypeEXT::eDrawMeshTasksCount)
        .setOffset(offsetof(TDGCSequenceTemplate, command));
    tokenDatas.push_back(idCmdtoken);

    vk::IndirectCommandsLayoutCreateInfoEXT info {};
    info.setFlags((explicitPreprocess
                       ? vk::IndirectCommandsLayoutUsageFlagBitsEXT::
                             eExplicitPreprocess
                       : vk::IndirectCommandsLayoutUsageFlagBitsEXT {0})
                  | (unorderedSequence
                         ? vk::IndirectCommandsLayoutUsageFlagBitsEXT::
                               eUnorderedSequences
                         : vk::IndirectCommandsLayoutUsageFlagBitsEXT {0}))
        .setTokens(tokenDatas)
        .setShaderStages(stages)
        .setPipelineLayout(pipelineLayout)
        .setIndirectStride(sizeof(TDGCSequenceTemplate));

    layout->mHandle =
        context.GetDevice()->createIndirectCommandsLayoutEXT(info);

    return layout;
}

}  // namespace IntelliDesign_NS::Vulkan::Core