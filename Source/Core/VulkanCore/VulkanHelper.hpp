#pragma once

#include <vulkan/vulkan_to_string.hpp>

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

#include "Core/Utilities/Logger.hpp"