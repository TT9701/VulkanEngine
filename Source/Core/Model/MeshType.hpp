#pragma once

#include <glm/glm.hpp>

namespace IntelliDesign_NS::Vulkan::Core {

struct Vertex {
    glm::vec4 position {};   // w - empty
    glm::vec2 normal {};     // normal.yz
    glm::vec2 texcoords {};
};

}  // namespace IntelliDesign_NS::Vulkan::Core