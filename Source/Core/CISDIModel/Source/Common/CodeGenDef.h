/**
 * @file CodeGenDef.h
 * @author 
 * @brief 宏递归展开
 * @version 0.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#define INTELLI_DS_MACRO_EXPAND(x) x

#define INTELLI_DS_GET_MACRO(                                                  \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,     \
    _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, \
    _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, \
    _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, \
    _62, _63, _64, Name, ...)                                                  \
    Name

#define INTELLI_DS_MACRO_BATCH(func, ...)                                  \
    INTELLI_DS_MACRO_EXPAND(INTELLI_DS_GET_MACRO(                          \
        __VA_ARGS__, INTELLI_DS_MACRO_BATCH_64, INTELLI_DS_MACRO_BATCH_63, \
        INTELLI_DS_MACRO_BATCH_62, INTELLI_DS_MACRO_BATCH_61,              \
        INTELLI_DS_MACRO_BATCH_60, INTELLI_DS_MACRO_BATCH_59,              \
        INTELLI_DS_MACRO_BATCH_58, INTELLI_DS_MACRO_BATCH_57,              \
        INTELLI_DS_MACRO_BATCH_56, INTELLI_DS_MACRO_BATCH_55,              \
        INTELLI_DS_MACRO_BATCH_54, INTELLI_DS_MACRO_BATCH_53,              \
        INTELLI_DS_MACRO_BATCH_52, INTELLI_DS_MACRO_BATCH_51,              \
        INTELLI_DS_MACRO_BATCH_50, INTELLI_DS_MACRO_BATCH_49,              \
        INTELLI_DS_MACRO_BATCH_48, INTELLI_DS_MACRO_BATCH_47,              \
        INTELLI_DS_MACRO_BATCH_46, INTELLI_DS_MACRO_BATCH_45,              \
        INTELLI_DS_MACRO_BATCH_44, INTELLI_DS_MACRO_BATCH_43,              \
        INTELLI_DS_MACRO_BATCH_42, INTELLI_DS_MACRO_BATCH_41,              \
        INTELLI_DS_MACRO_BATCH_40, INTELLI_DS_MACRO_BATCH_39,              \
        INTELLI_DS_MACRO_BATCH_38, INTELLI_DS_MACRO_BATCH_37,              \
        INTELLI_DS_MACRO_BATCH_36, INTELLI_DS_MACRO_BATCH_35,              \
        INTELLI_DS_MACRO_BATCH_34, INTELLI_DS_MACRO_BATCH_33,              \
        INTELLI_DS_MACRO_BATCH_32, INTELLI_DS_MACRO_BATCH_31,              \
        INTELLI_DS_MACRO_BATCH_30, INTELLI_DS_MACRO_BATCH_29,              \
        INTELLI_DS_MACRO_BATCH_28, INTELLI_DS_MACRO_BATCH_27,              \
        INTELLI_DS_MACRO_BATCH_26, INTELLI_DS_MACRO_BATCH_25,              \
        INTELLI_DS_MACRO_BATCH_24, INTELLI_DS_MACRO_BATCH_23,              \
        INTELLI_DS_MACRO_BATCH_22, INTELLI_DS_MACRO_BATCH_21,              \
        INTELLI_DS_MACRO_BATCH_20, INTELLI_DS_MACRO_BATCH_19,              \
        INTELLI_DS_MACRO_BATCH_18, INTELLI_DS_MACRO_BATCH_17,              \
        INTELLI_DS_MACRO_BATCH_16, INTELLI_DS_MACRO_BATCH_15,              \
        INTELLI_DS_MACRO_BATCH_14, INTELLI_DS_MACRO_BATCH_13,              \
        INTELLI_DS_MACRO_BATCH_12, INTELLI_DS_MACRO_BATCH_11,              \
        INTELLI_DS_MACRO_BATCH_10, INTELLI_DS_MACRO_BATCH_9,               \
        INTELLI_DS_MACRO_BATCH_8, INTELLI_DS_MACRO_BATCH_7,                \
        INTELLI_DS_MACRO_BATCH_6, INTELLI_DS_MACRO_BATCH_5,                \
        INTELLI_DS_MACRO_BATCH_4, INTELLI_DS_MACRO_BATCH_3,                \
        INTELLI_DS_MACRO_BATCH_2,                                          \
        INTELLI_DS_MACRO_BATCH_1)(func, __VA_ARGS__))

#define INTELLI_DS_MACRO_BATCH_1(func, arg1) func(arg1)

#define INTELLI_DS_MACRO_BATCH_2(func, arg1, arg2) \
    INTELLI_DS_MACRO_BATCH_1(func, arg1), INTELLI_DS_MACRO_BATCH_1(func, arg2)

#define INTELLI_DS_MACRO_BATCH_3(func, arg1, arg2, arg3) \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                \
        INTELLI_DS_MACRO_BATCH_2(func, arg2, arg3)

#define INTELLI_DS_MACRO_BATCH_4(func, arg1, arg2, arg3, arg4) \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                      \
        INTELLI_DS_MACRO_BATCH_3(func, arg2, arg3, arg4)

#define INTELLI_DS_MACRO_BATCH_5(func, arg1, arg2, arg3, arg4, arg5) \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                            \
        INTELLI_DS_MACRO_BATCH_4(func, arg2, arg3, arg4, arg5)

#define INTELLI_DS_MACRO_BATCH_6(func, arg1, arg2, arg3, arg4, arg5, arg6) \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                  \
        INTELLI_DS_MACRO_BATCH_5(func, arg2, arg3, arg4, arg5, arg6)

#define INTELLI_DS_MACRO_BATCH_7(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                 arg7)                                     \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                  \
        INTELLI_DS_MACRO_BATCH_6(func, arg2, arg3, arg4, arg5, arg6, arg7)

#define INTELLI_DS_MACRO_BATCH_8(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                 arg7, arg8)                               \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                  \
        INTELLI_DS_MACRO_BATCH_7(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                 arg8)

#define INTELLI_DS_MACRO_BATCH_9(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                 arg7, arg8, arg9)                         \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                  \
        INTELLI_DS_MACRO_BATCH_8(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                 arg8, arg9)

#define INTELLI_DS_MACRO_BATCH_10(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10)                  \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_9(func, arg2, arg3, arg4, arg5, arg6, arg7,  \
                                 arg8, arg9, arg10)

#define INTELLI_DS_MACRO_BATCH_11(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11)           \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_10(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11)

#define INTELLI_DS_MACRO_BATCH_12(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12)    \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_11(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12)

#define INTELLI_DS_MACRO_BATCH_13(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13)                                    \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_12(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13)

#define INTELLI_DS_MACRO_BATCH_14(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14)                             \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_13(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14)

#define INTELLI_DS_MACRO_BATCH_15(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14, arg15)                      \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_14(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14, arg15)

#define INTELLI_DS_MACRO_BATCH_16(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14, arg15, arg16)               \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_15(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14, arg15, arg16)

#define INTELLI_DS_MACRO_BATCH_17(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14, arg15, arg16, arg17)        \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_16(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14, arg15, arg16, arg17)

#define INTELLI_DS_MACRO_BATCH_18(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14, arg15, arg16, arg17, arg18) \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_17(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14, arg15, arg16, arg17, arg18)

#define INTELLI_DS_MACRO_BATCH_19(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19)                   \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_18(func, arg2, arg3, arg4, arg5, arg6, arg7,   \
                                  arg8, arg9, arg10, arg11, arg12, arg13,     \
                                  arg14, arg15, arg16, arg17, arg18, arg19)

#define INTELLI_DS_MACRO_BATCH_20(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20)            \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_19(func, arg2, arg3, arg4, arg5, arg6, arg7,   \
                                  arg8, arg9, arg10, arg11, arg12, arg13,     \
                                  arg14, arg15, arg16, arg17, arg18, arg19,   \
                                  arg20)

#define INTELLI_DS_MACRO_BATCH_21(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21)     \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_20(func, arg2, arg3, arg4, arg5, arg6, arg7,   \
                                  arg8, arg9, arg10, arg11, arg12, arg13,     \
                                  arg14, arg15, arg16, arg17, arg18, arg19,   \
                                  arg20, arg21)

#define INTELLI_DS_MACRO_BATCH_22(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14, arg15, arg16, arg17, arg18, \
                                  arg19, arg20, arg21, arg22)               \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_21(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14, arg15, arg16, arg17, arg18, arg19, \
                                  arg20, arg21, arg22)

#define INTELLI_DS_MACRO_BATCH_23(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14, arg15, arg16, arg17, arg18, \
                                  arg19, arg20, arg21, arg22, arg23)        \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_22(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14, arg15, arg16, arg17, arg18, arg19, \
                                  arg20, arg21, arg22, arg23)

#define INTELLI_DS_MACRO_BATCH_24(func, arg1, arg2, arg3, arg4, arg5, arg6, \
                                  arg7, arg8, arg9, arg10, arg11, arg12,    \
                                  arg13, arg14, arg15, arg16, arg17, arg18, \
                                  arg19, arg20, arg21, arg22, arg23, arg24) \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                   \
        INTELLI_DS_MACRO_BATCH_23(func, arg2, arg3, arg4, arg5, arg6, arg7, \
                                  arg8, arg9, arg10, arg11, arg12, arg13,   \
                                  arg14, arg15, arg16, arg17, arg18, arg19, \
                                  arg20, arg21, arg22, arg23, arg24)

#define INTELLI_DS_MACRO_BATCH_25(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25)                                               \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_24(func, arg2, arg3, arg4, arg5, arg6, arg7,   \
                                  arg8, arg9, arg10, arg11, arg12, arg13,     \
                                  arg14, arg15, arg16, arg17, arg18, arg19,   \
                                  arg20, arg21, arg22, arg23, arg24, arg25)

#define INTELLI_DS_MACRO_BATCH_26(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26)                                        \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_25(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26)

#define INTELLI_DS_MACRO_BATCH_27(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27)                                 \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_26(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27)

#define INTELLI_DS_MACRO_BATCH_28(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28)                          \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_27(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28)

#define INTELLI_DS_MACRO_BATCH_29(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29)                   \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_28(func, arg2, arg3, arg4, arg5, arg6, arg7,   \
                                  arg8, arg9, arg10, arg11, arg12, arg13,     \
                                  arg14, arg15, arg16, arg17, arg18, arg19,   \
                                  arg20, arg21, arg22, arg23, arg24, arg25,   \
                                  arg26, arg27, arg28, arg29)

#define INTELLI_DS_MACRO_BATCH_30(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30)            \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_29(func, arg2, arg3, arg4, arg5, arg6, arg7,   \
                                  arg8, arg9, arg10, arg11, arg12, arg13,     \
                                  arg14, arg15, arg16, arg17, arg18, arg19,   \
                                  arg20, arg21, arg22, arg23, arg24, arg25,   \
                                  arg26, arg27, arg28, arg29, arg30)

#define INTELLI_DS_MACRO_BATCH_31(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31)     \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_30(func, arg2, arg3, arg4, arg5, arg6, arg7,   \
                                  arg8, arg9, arg10, arg11, arg12, arg13,     \
                                  arg14, arg15, arg16, arg17, arg18, arg19,   \
                                  arg20, arg21, arg22, arg23, arg24, arg25,   \
                                  arg26, arg27, arg28, arg29, arg30, arg31)

#define INTELLI_DS_MACRO_BATCH_32(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32)                                                                    \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_31(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32)

#define INTELLI_DS_MACRO_BATCH_33(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33)                                                             \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_32(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33)

#define INTELLI_DS_MACRO_BATCH_34(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34)                                                      \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_33(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34)

#define INTELLI_DS_MACRO_BATCH_35(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35)                                               \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_34(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35)

#define INTELLI_DS_MACRO_BATCH_36(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36)                                        \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_35(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36)

#define INTELLI_DS_MACRO_BATCH_37(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37)                                 \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_36(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37)

#define INTELLI_DS_MACRO_BATCH_38(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38)                          \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_37(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38)

#define INTELLI_DS_MACRO_BATCH_39(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39)                   \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_38(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39)

#define INTELLI_DS_MACRO_BATCH_40(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40)            \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_39(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40)

#define INTELLI_DS_MACRO_BATCH_41(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41)     \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_40(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41)

#define INTELLI_DS_MACRO_BATCH_42(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42)                                                                    \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_41(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42)

#define INTELLI_DS_MACRO_BATCH_43(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43)                                                             \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_42(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43)

#define INTELLI_DS_MACRO_BATCH_44(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44)                                                      \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_43(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44)

#define INTELLI_DS_MACRO_BATCH_45(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45)                                               \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_44(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45)

#define INTELLI_DS_MACRO_BATCH_46(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46)                                        \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_45(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46)

#define INTELLI_DS_MACRO_BATCH_47(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47)                                 \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_46(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47)

#define INTELLI_DS_MACRO_BATCH_48(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48)                          \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_47(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48)

#define INTELLI_DS_MACRO_BATCH_49(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49)                   \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_48(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49)

#define INTELLI_DS_MACRO_BATCH_50(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50)            \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_49(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50)

#define INTELLI_DS_MACRO_BATCH_51(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51)     \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_50(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51)

#define INTELLI_DS_MACRO_BATCH_52(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52)                                                                    \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_51(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52)

#define INTELLI_DS_MACRO_BATCH_53(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53)                                                             \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_52(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53)

#define INTELLI_DS_MACRO_BATCH_54(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54)                                                      \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_53(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54)

#define INTELLI_DS_MACRO_BATCH_55(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55)                                               \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_54(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55)

#define INTELLI_DS_MACRO_BATCH_56(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56)                                        \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_55(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56)

#define INTELLI_DS_MACRO_BATCH_57(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57)                                 \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_56(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57)

#define INTELLI_DS_MACRO_BATCH_58(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57, arg58)                          \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_57(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57, arg58)

#define INTELLI_DS_MACRO_BATCH_59(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59)                   \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_58(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57, arg58, arg59)

#define INTELLI_DS_MACRO_BATCH_60(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60)            \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_59(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57, arg58, arg59, arg60)

#define INTELLI_DS_MACRO_BATCH_61(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60, arg61)     \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_60(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57, arg58, arg59, arg60, arg61)

#define INTELLI_DS_MACRO_BATCH_62(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60, arg61,     \
    arg62)                                                                    \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_61(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57, arg58, arg59, arg60, arg61, arg62)

#define INTELLI_DS_MACRO_BATCH_63(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60, arg61,     \
    arg62, arg63)                                                             \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_62(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57, arg58, arg59, arg60, arg61, arg62, arg63)

#define INTELLI_DS_MACRO_BATCH_64(                                            \
    func, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, \
    arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21,     \
    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31,     \
    arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41,     \
    arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51,     \
    arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60, arg61,     \
    arg62, arg63, arg64)                                                      \
    INTELLI_DS_MACRO_BATCH_1(func, arg1),                                     \
        INTELLI_DS_MACRO_BATCH_63(                                            \
            func, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,      \
            arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,    \
            arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28,    \
            arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37,    \
            arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46,    \
            arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55,    \
            arg56, arg57, arg58, arg59, arg60, arg61, arg62, arg63, arg64)