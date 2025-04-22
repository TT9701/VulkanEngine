/**
 * @file PMR_Def.h
 * @author 
 * @brief   Macros in this file will be useful for defining the PMR container
 *        elements which also have PMR container elements.
 * @details For example, if you have a struct like this:
 *            struct MyStruct { ::std::pmr::vector<int> vec; };
 *        and you use this struct like ::std::pmr::vector<MyStruct>, you can use
 *        the macros in this file to define the PMR elements(MyStruct) to use
 *        the allocator of the outer container(::std::pmr::vector<MyStruct>)
 *        without any extra effort.

 *        The macros in this file are designed to be used in class/struct.
 *
 *      Example:
 *        1. no member variables need to be initialized in the constructors
 *            struct MyStruct1 {
                INTELLI_DS_PMR_ELEMENT_DEFAULT_CTORS(MyStruct1, name, vec);

                ::std::pmr::string name;
                ::std::pmr::vector<int> vec;
              };

          2. member variables need to be initialized in the constructors
              struct MyStruct2 {
                INTELLI_DS_PMR_ELEMENT_CTOR_SIG(MyStruct2, int n)
                    : count(n),
                      INTELLI_DS_PMR_ELEMENT_CTOR_DEFAULT_INIT_LIST(name, vec) {}

                INTELLI_DS_PMR_ELEMENT_COPY_CTOR(MyStruct2, name, vec),
                    count(_other_.count) {}

                INTELLI_DS_PMR_ELEMENT_MOVE_CTOR(MyStruct2, name, vec),
                    count(::std::move(_other_.count)) {}

                int count;
                ::std::pmr::string name;
                ::std::pmr::vector<int> vec;
              };
 *
 * @version 0.1
 * @date 2025-01-24
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#ifdef INTELLI_DS_EXPORT_PMR_API
#define INTELLI_DS_PMR_API __declspec(dllexport)
#else
#define INTELLI_DS_PMR_API __declspec(dllimport)
#endif

#include "CodeGenDef.h"

/*
 * PMR Element constructors MACROs
 */
#define INTELLI_DS_PMR_ELEMENT_INIT(Name) Name(_allocator_)

#define INTELLI_DS_PMR_ELEMENT_DEFAULT_CTOR_SIG(Name) \
    INTELLI_DS_PMR_API explicit Name(allocator_type _allocator_ = {})

#define INTELLI_DS_PMR_ELEMENT_CTOR_SIG(Name, ...) \
    INTELLI_DS_PMR_ELEMENT_DECL_ALLOCATOR_TYPE;    \
    INTELLI_DS_PMR_API explicit Name(__VA_ARGS__,  \
                                     allocator_type _allocator_ = {})

#define INTELLI_DS_PMR_ELEMENT_CTOR_DEFAULT_INIT_LIST(...) \
    INTELLI_DS_MACRO_EXPAND(                               \
        INTELLI_DS_MACRO_BATCH(INTELLI_DS_PMR_ELEMENT_INIT, __VA_ARGS__))

#define INTELLI_DS_PMR_ELEMENT_DEFAULT_CTOR(Name, ...) \
    INTELLI_DS_PMR_ELEMENT_DECL_ALLOCATOR_TYPE;        \
    INTELLI_DS_PMR_ELEMENT_DEFAULT_CTOR_SIG(Name)      \
        : INTELLI_DS_PMR_ELEMENT_CTOR_DEFAULT_INIT_LIST(__VA_ARGS__) {}

/*
 * PMR Element copy constructors MACROs
 */
#define INTELLI_DS_PMR_ELEMENT_COPY_INIT(Name) \
    Name {                                     \
        _other_.Name, _allocator_              \
    }

#define INTELLI_DS_PMR_ELEMENT_COPY_CTOR_SIG(Name) \
    INTELLI_DS_PMR_API Name(Name const& _other_, allocator_type _allocator_)

#define INTELLI_DS_PMR_ELEMENT_COPY_CTOR_DEFAULT_INIT_LIST(...) \
    INTELLI_DS_MACRO_EXPAND(                                    \
        INTELLI_DS_MACRO_BATCH(INTELLI_DS_PMR_ELEMENT_COPY_INIT, __VA_ARGS__))

#define INTELLI_DS_PMR_ELEMENT_COPY_CTOR(Name, ...) \
    INTELLI_DS_PMR_ELEMENT_COPY_CTOR_SIG(Name)      \
        : INTELLI_DS_PMR_ELEMENT_COPY_CTOR_DEFAULT_INIT_LIST(__VA_ARGS__)

#define INTELLI_DS_PMR_ELEMENT_DEFAULT_COPY_CTOR(Name, ...) \
    INTELLI_DS_PMR_ELEMENT_COPY_CTOR(Name, __VA_ARGS__) {}

/*
 * PMR Element move constructors MACROs
 */
#define INTELLI_DS_PMR_ELEMENT_MOVE_INIT(Name) \
    Name {                                     \
        ::std::move(_other_.Name), _allocator_ \
    }

#define INTELLI_DS_PMR_ELEMENT_MOVE_CTOR_SIG(Name) \
    INTELLI_DS_PMR_API Name(Name&& _other_, allocator_type _allocator_)

#define INTELLI_DS_PMR_ELEMENT_MOVE_CTOR_DEFAULT_INIT_LIST(...) \
    INTELLI_DS_MACRO_EXPAND(                                    \
        INTELLI_DS_MACRO_BATCH(INTELLI_DS_PMR_ELEMENT_MOVE_INIT, __VA_ARGS__))

#define INTELLI_DS_PMR_ELEMENT_MOVE_CTOR(Name, ...) \
    INTELLI_DS_PMR_ELEMENT_MOVE_CTOR_SIG(Name)      \
        : INTELLI_DS_PMR_ELEMENT_MOVE_CTOR_DEFAULT_INIT_LIST(__VA_ARGS__)

#define INTELLI_DS_PMR_ELEMENT_DEFAULT_MOVE_CTOR(Name, ...) \
    INTELLI_DS_PMR_ELEMENT_MOVE_CTOR(Name, __VA_ARGS__) {}

/*
 * PMR Element allocator type MACROs
 */
#define INTELLI_DS_PMR_ELEMENT_DECL_ALLOCATOR_TYPE \
    using allocator_type = ::std::pmr::polymorphic_allocator<>

/*
 * PMR Element combined MACROs
 */
#define INTELLI_DS_PMR_ELEMENT_DEFAULT_CTORS(Name, ...)         \
    INTELLI_DS_PMR_ELEMENT_DEFAULT_CTOR(Name, __VA_ARGS__)      \
    INTELLI_DS_PMR_ELEMENT_DEFAULT_COPY_CTOR(Name, __VA_ARGS__) \
    INTELLI_DS_PMR_ELEMENT_DEFAULT_MOVE_CTOR(Name, __VA_ARGS__)