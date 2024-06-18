#pragma once

#include <glm/glm.hpp>

struct Vertex {
    glm::vec3 position {};
    float     uvX {};
    glm::vec3 normal {};
    float     uvY {};
    glm::vec4 color {};
};