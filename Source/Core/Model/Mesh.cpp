#include "Mesh.hpp"

namespace IntelliDesign_NS::Vulkan::Core {

Mesh::Mesh(std::vector<Vertex> const& vertices,
           std::vector<uint32_t> const& indices)
    : mVertices(vertices), mIndices(indices) {}

}  // namespace IntelliDesign_NS::Vulkan::Core