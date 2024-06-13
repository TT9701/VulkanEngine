#pragma once

#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>

namespace Utils {

inline ::std::vector<::std::string> FilterStringList(::std::vector<::std::string> available,
                                                     ::std::vector<::std::string> request) {
    ::std::ranges::sort(available);
    ::std::ranges::sort(request);
    ::std::vector<::std::string> result {};
    ::std::ranges::set_intersection(available, request,
                            ::std::back_inserter(result));
    return result;
}

}  // namespace Utils