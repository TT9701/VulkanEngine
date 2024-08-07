#pragma once

#include <glm/glm.hpp>

struct Vertex {
    glm::vec4 position {};   // w - empty
    glm::vec4 normal {};     // w - empty
    glm::vec2 texcoords {};  // z, w - empty
    glm::vec2 paddings {};
    glm::vec4 tangent {};    // w - empty
    glm::vec4 bitangent {};  // w - empty
};