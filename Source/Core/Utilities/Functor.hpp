#pragma once

#include "MemoryPool.h"

namespace IntelliDesign_NS::Vulkan::Core {

template <typename Type_Fn>
    requires(!::std::is_fundamental_v<Type_Fn>)
struct FunctionTraits {
    using FunctionTraitsForward =
        FunctionTraits<decltype(::std::function {::std::declval<Type_Fn>()})>;

    using Type_TpRet = typename FunctionTraitsForward::Type_TpRet;
    using Type_TpArgs = typename FunctionTraitsForward::Type_TpArgs;
};

template <typename Ret, typename... Args>
struct FunctionTraits<Ret (*)(Args...)> {
    using Type_TpRet = Ret;
    using Type_TpArgs = ::std::tuple<Args...>;
};

template <typename Ret, typename... Args>
struct FunctionTraits<::std::function<Ret(Args...)>> {
    using Type_TpRet = Ret;
    using Type_TpArgs = ::std::tuple<Args...>;
};

template <typename Ret, typename... Args>
struct Functor {
    virtual Ret operator()(Args const&... args) const = 0;
};

template <typename Type_Fn, typename Ret, typename... Args>
struct Function : Functor<Ret, Args...> {
    Function(Type_Fn&& f) : func {::std::move(f)} {}

    Ret operator()(Args const&... args) const override { return func(args...); }

    Type_Fn func;
};

template <class Ret, class Type_Fn, class... TArgs>
auto MakeFunc(Type_Fn&& f, ::std::tuple<TArgs...>*) {
    return MakeShared<Function<Type_Fn, Ret, TArgs...>>(
        ::std::forward<Type_Fn>(f));
}

template <typename Type_Fn>
auto WrapSharedFuncPtr(Type_Fn&& f) {
    using Type_TpRet = typename FunctionTraits<Type_Fn>::Type_TpRet;
    using Type_TpArgs = typename FunctionTraits<Type_Fn>::Type_TpArgs;
    return MakeFunc<Type_TpRet>(::std::forward<Type_Fn>(f),
                                static_cast<Type_TpArgs*>(nullptr));
}

template <class Ty>
struct SingleVar_IsUint32 {
    bool value = false;
};

template <>
struct SingleVar_IsUint32<uint32_t> {
    bool value = true;
};

template <class ...Ty>
struct DoubleVar_IsUint32 {
    bool value = false;
};

template<>
struct DoubleVar_IsUint32<uint32_t, uint32_t> {
    bool value = true;
};

template <class Ty>
inline constexpr bool SingleVarIsUint32_v = SingleVar_IsUint32<Ty>::value;

template <class ...Ty>
inline constexpr bool DoubleVarIsUint32_v = DoubleVar_IsUint32<Ty>::value;

}  // namespace IntelliDesign_NS::Vulkan::Core